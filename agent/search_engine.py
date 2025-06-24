from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.mixture import GaussianMixture

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore, BaseNode, MetadataMode, IndexNode, ImageNode, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from data_process.vl_embedding import VL_Embedding
from utils.format_converter import nodefile2node,nodes2dict

# 定义基础目录前缀
base_dir_prefix = '/home/huangjiayu/MFlexRAG/data_process/data'

def gmm(recall_result: list[NodeWithScore], input_length: int=20, max_valid_length: int=10, min_valid_length: int=5) -> List[NodeWithScore]:
    """
    使用高斯混合模型(GMM)对检索结果进行聚类和筛选

    参数:
        recall_result: 检索结果列表
        input_length: 输入GMM的最大长度
        max_valid_length: 输出的最大有效长度
        min_valid_length: 输出的最小有效长度
    返回:
        经过GMM处理后的检索结果列表
    """
    # 提取检索结果的分数
    scores = [node.score for node in recall_result[:input_length]]

    # 将分数转换为numpy数组并重塑为2D数组
    scores = np.array(scores)
    scores = scores.reshape(-1, 1)
    # 使用2个组件的高斯混合模型进行聚类
    gmm = GaussianMixture(n_components=2, n_init=1,random_state=0)
    gmm.fit(scores)
    labels = gmm.predict(scores)

    # 将分数和检索结果按聚类标签分组
    scores = scores.flatten()
    scores = [scores[labels == label] for label in np.unique(labels)]
    recall_result = [np.array(recall_result[:input_length])[labels == label].tolist() for label in np.unique(labels)]

    # 获取每个聚类的最大分数并排序
    max_values = np.array([np.max(p) for p in scores])
    sorted_indices = np.argsort(-max_values)

    # 如果只有一个聚类，直接返回前max_valid_length个结果
    if len(sorted_indices) == 1:
        valid_recall_result = recall_result[0]
        valid_recall_result = valid_recall_result[:max_valid_length]
        for node in valid_recall_result:
            node.score = None
        return valid_recall_result

    # 获取分数最高的两个聚类
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]

    valid_recall_result = recall_result[max_index]

    # 根据长度要求调整结果
    if len(valid_recall_result) > max_valid_length:
        valid_recall_result = valid_recall_result[:max_valid_length]
    elif len(valid_recall_result) < min_valid_length:
        second_valid_recall_result_len = min_valid_length - len(valid_recall_result)
        valid_recall_result.extend(recall_result[second_max_index][:second_valid_recall_result_len])

    # 清除所有节点的分数
    for node in valid_recall_result:
        node.score = None

    return valid_recall_result

class SearchEngine:
    """
    搜索引擎类，用于处理文档检索，是封闭领域问答，需要针对特定文档下的图片进行检索
    """
    def __init__(self, dataset, doc_name, node_dir_prefix=None, embed_model_name='vidore/colqwen2-v1.0'):
        """
        初始化搜索引擎

        参数:
            dataset: 数据集名称
            doc_name: 文档名称（子目录名），用于封闭领域检索
            node_dir_prefix: 节点目录前缀
            embed_model_name: 嵌入模型名称
        """
        Settings.llm = None
        self.gmm = False
        self.gmm_candidate_length = False
        self.return_raw = False
        self.input_gmm = 20
        self.max_output_gmm = 10
        self.min_output_gmm = 5
        self.dataset = dataset
        self.dataset_dir = os.path.join(base_dir_prefix, dataset)
        self.img_dir = os.path.join(self.dataset_dir, 'img')

        # 设置文档名称（必需参数）
        if doc_name is None:
            raise ValueError("doc_name is required for closed-domain document search")
        self.doc_name = doc_name

        if node_dir_prefix is None:
            if 'bge' in embed_model_name:
                node_dir_prefix = 'bge_ingestion'
            elif 'NV-Embed' in embed_model_name:
                node_dir_prefix = 'nv_ingestion'
            elif 'colqwen' in embed_model_name:
                node_dir_prefix = 'colqwen_ingestion'
            elif 'openbmb' in embed_model_name:
                node_dir_prefix = 'visrag_ingestion'
            elif 'colpali' in embed_model_name:
                node_dir_prefix = 'colpali_ingestion'
            else:
                raise ValueError('Please specify the node_dir_prefix')

        if node_dir_prefix in ['colqwen_ingestion','visrag_ingestion','colpali_ingestion']:
            self.vl_ret = True # 是否使用视觉检索
        else:
            self.vl_ret = False

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.rag_dataset_path = os.path.join(self.dataset_dir, 'rag_dataset.json')
        self.workers = 1
        self.embed_model_name = embed_model_name
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name:
            if self.vl_ret:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
            else:
                self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='text')
        else:
            self.vector_embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name, embed_batch_size=10, max_length=512, trust_remote_code=True, device='cuda')
        self.recall_num = 100

        # 初始化节点和查询引擎
        self.nodes = []
        self.embedding_img = []
        self.query_engine = None

        # 加载指定文档的节点
        self.load_document_nodes()

        self.output_dir = os.path.join(self.dataset_dir, 'search_output')
        # os.makedirs(self.output_dir, exist_ok=True)

    def _process_search_result(self, node):
        """
        统一处理检索结果节点
        Args:
            node: 检索结果节点
        Returns:
            处理后的文档和页面信息
        """
        # 获取文件路径
        if isinstance(node, dict):
            if 'image_path' in node['node']:
                file_path = node['node']['image_path']
            else:
                file_path = node['node']['metadata'].get('filename', '')
        else:
            if hasattr(node, 'image_path'):
                file_path = node.image_path
            else:
                file_path = node.metadata.get('filename', '')

        # 提取文件名（不含扩展名）
        file_name = '.'.join(os.path.basename(file_path).split('.')[:-1])

        # 解析文档名和页码
        doc_parts = file_name.split('_')
        if len(doc_parts) > 1:
            doc = '_'.join(doc_parts[:-1])
            page = doc_parts[-1]
        else:
            doc = file_name
            page = '0'

        return doc, page


    def load_document_nodes(self):
        """
        加载指定文档目录下的所有节点文件

        返回:
            解析后的节点列表
        """
        if self.doc_name is None:
            raise ValueError("doc_name must be specified for document-specific loading")

        # 构建文档特定的节点目录路径
        doc_node_dir = os.path.join(self.node_dir, self.doc_name)

        if not os.path.exists(doc_node_dir):
            raise ValueError(f"Document directory not found: {doc_node_dir}")

        print(f'Loading nodes from document: {self.doc_name}')
        self.nodes = self._load_nodes_from_directory(doc_node_dir)

        # 根据模型类型处理节点，设置检索器，先只支持vidore的视觉检索
        if self.vl_ret and 'vidore' in self.embed_model_name:
            self.embedding_img = [torch.tensor(node.embedding).view(-1,128).bfloat16() for node in self.nodes]
            self.embedding_img = [tensor.to(self.vector_embed_model.embed_model.device) for tensor in self.embedding_img]

    def change_document(self, doc_name):
        """
        切换文档领域，重新加载指定文档的节点

        参数:
            doc_name: 新的文档名称
        """
        if doc_name == self.doc_name:
            print(f"Already loaded document: {doc_name}")
            return

        self.doc_name = doc_name
        print(f"Switching to document: {doc_name}")

        # 清空之前的节点和查询引擎
        self.nodes = []
        self.embedding_img = []
        self.query_engine = None

        # 重新加载新文档的节点
        self.load_document_nodes()
        print(f"Successfully loaded {len(self.nodes)} nodes from document: {doc_name}")

    def _load_nodes_from_directory(self, directory):
        """
        从指定目录加载节点文件

        参数:
            directory: 节点文件目录
        返回:
            解析后的节点列表
        """
        if not os.path.exists(directory):
            return []

        files = os.listdir(directory)
        parsed_files = []
        max_workers = 10

        if max_workers == 1:
            for file in tqdm(files, desc=f"Loading nodes from {os.path.basename(directory)}"):
                input_file = os.path.join(directory, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'json':
                    continue
                nodes = nodefile2node(input_file)
                # 确保节点类型正确
                for node in nodes:
                    if isinstance(node, (TextNode, ImageNode)):
                        parsed_files.append(node)
        else:
            def parse_file(file, node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'json':
                    return []
                nodes = nodefile2node(input_file)
                # 确保节点类型正确
                return [node for node in nodes if isinstance(node, (TextNode, ImageNode))]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(parse_file, files, [directory]*len(files)),
                                  total=len(files),
                                  desc=f"Loading nodes from {os.path.basename(directory)}"))
            for result in results:
                parsed_files.extend(result)
        return parsed_files

    def search(self, query):
        """
        执行搜索

        参数:
            query: 查询文本
        返回:
            检索结果字典
        """
        # 检查是否已加载节点
        if not self.nodes:
            if self.doc_name is not None:
                print(f"No nodes loaded for document: {self.doc_name}")
                return nodes2dict([])
            else:
                print("No nodes loaded")
                return nodes2dict([])

        if self.vl_ret and 'vidore' in self.embed_model_name:
            query_embedding = self.vector_embed_model.embed_text(query)
            scores = self.vector_embed_model.processor.score(query_embedding,self.embedding_img)
            k = min(100, scores[0].numel())
            values, indices = torch.topk(scores[0], k=k)
            recall_results = [self.nodes[i] for i in indices]
            for node in recall_results:
                node.embedding = None
            recall_results = [NodeWithScore(node=node, score=score) for node, score in zip(recall_results, values)]
            recall_results_output = recall_results

        if self.gmm:
            recall_results_output = gmm(recall_results,self.input_gmm,self.max_output_gmm,self.min_output_gmm)
        if self.return_raw:
            return recall_results_output
        if self.gmm_candidate_length:
            candidate_length = [1,2,4,6,9,12,16,20]
            current_length = len(recall_results_output)
            target_length = min([num for num in candidate_length if num > current_length])
            recall_results_output = recall_results[:target_length]
        # 处理检索结果
        result_docs = {}
        for node in recall_results_output:
            doc, page = self._process_search_result(node)
            if doc not in result_docs:
                result_docs[doc] = [int(page)]
            elif int(page) not in result_docs[doc]:
                result_docs[doc].append(int(page))

        return nodes2dict(recall_results_output)

    def search_example(self,example):
        """
        搜索示例

        参数:
            example: 示例数据
        返回:
            添加检索结果的示例数据
        """
        query = example['query']
        recall_result =  self.search(query)
        example['recall_result'] = recall_result
        return example

    def search_multi_session(self,output_file='search_result.json'):
        """
        多会话搜索，根据rag_dataset_path中的数据进行搜索，并保存到output_file中

        参数:
            output_file: 输出文件名
        """
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.rag_dataset_path, 'r') as f:
            dataset = json.load(f)
        data = dataset['examples']
        results = []
        if self.workers == 1:
            for example in tqdm(data):
                results.append(self.search_example(example))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {executor.submit(self.search_example, example): example for example in data}
                for future in tqdm(as_completed(future_to_file), total=len(data), desc='Processing files'):
                    results.append(future.result())
        with open(os.path.join(self.output_dir, output_file), 'w') as json_file:
            json.dump(results, json_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    datasets = ['test']
    for dataset in datasets:
        # 示例1: 文档特定的搜索引擎
        print("=== 示例1: 文档特定的搜索引擎 ===")
        search_engine = SearchEngine(
            dataset=dataset,
            doc_name='123',  # 指定文档名称
            node_dir_prefix='colqwen_ingestion',
            embed_model_name='vidore/colqwen2-v1.0'
        )

        # 执行搜索
        recall_results = search_engine.search('where is Figure 2: Data Construction pipeline?')
        recall_results_dict = nodes2dict(recall_results) if not isinstance(recall_results, dict) else recall_results

        print(f"\nFound {len(recall_results_dict.get('source_nodes', []))} results from document '123':")
        # 打印前5个结果的文件名
        for i, result in enumerate(recall_results_dict['source_nodes'][:5]):
            print(f"File name: {result['node']['metadata'].get('file_name', 'N/A')}")

        # 切换到另一个文档
        print("\n=== 切换到文档 '456' ===")
        search_engine.change_document('456')

        # 在新文档中搜索
        recall_results = search_engine.search('where is Figure 3. A structured taxonomy of synthesizing RAG and Reasoning?')
        recall_results_dict = nodes2dict(recall_results) if not isinstance(recall_results, dict) else recall_results

        print(f"\nFound {len(recall_results_dict.get('source_nodes', []))} results from document '456':")
        # 打印前5个结果的文件名
        for result in recall_results_dict['source_nodes'][:5]:
            print(f"File name: {result['node']['metadata'].get('file_name', 'N/A')}")
