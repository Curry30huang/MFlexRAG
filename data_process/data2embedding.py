import os
import sys
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode
from llama_index.core import SimpleDirectoryReader
from PIL import Image
import time

project_path = os.getcwd()
base_dir_prefix = f"{project_path}/data_process/data"

from data_process.vl_embedding import VL_Embedding

class Ingestion:
    """
    数据向量化处理类
    用于将图片和文本数据转换为向量表示，并保存为独立的节点文件
    """
    def __init__(self, dataset_dir, input_prefix='img', output_prefix='colqwen_ingestion', embed_model_name='vidore/colqwen2-v1.0'):
        """
        初始化向量化处理类

        参数:
            dataset_dir: 数据集根目录
            input_prefix: 输入数据目录名
            output_prefix: 输出向量目录名
            embed_model_name: 使用的向量化模型名称
        """
        # 转换为绝对路径
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.input_dir = os.path.abspath(os.path.join(self.dataset_dir, input_prefix))
        self.output_dir = os.path.abspath(os.path.join(self.dataset_dir, output_prefix))
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")

        self.chunk_size = 1024  # 文本分块大小
        self.overlap_size = 0   # 文本分块重叠大小
        self.workers = 5        # 并行处理的工作进程数
        self.reader = FlatReader()
        self.embed_model_name = embed_model_name

        # 根据模型类型初始化处理管道
        if 'vidore' in embed_model_name or 'openbmb' in embed_model_name:
            if input_prefix == 'img':
                # 图片处理管道
                self.embed_model = VL_Embedding(model=embed_model_name, mode='image')
            # else:
            #     # 文本处理管道
            #     self.pipeline = IngestionPipeline(
            #         transformations=[
            #             SimpleFileNodeParser(),
            #             SentenceSplitter(
            #                 include_metadata=True,
            #                 include_prev_next_rel=True,
            #                 chunk_size=self.chunk_size,
            #                 chunk_overlap=self.overlap_size,
            #                 separator=' ',
            #                 paragraph_separator='\n\n',
            #                 secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'
            #             ),
            #             VL_Embedding(model=embed_model_name, mode='text')
            #         ],
            #     )
        else:
            # 使用HuggingFace模型的处理管道
            self.pipeline = IngestionPipeline(
                transformations=[
                    SimpleFileNodeParser(),
                    SentenceSplitter(
                        include_metadata=True,
                        include_prev_next_rel=True,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.overlap_size,
                        separator=' ',
                        paragraph_separator='\n\n\n',
                        secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'
                    ),
                    HuggingFaceEmbedding(model_name=self.embed_model_name, trust_remote_code=True)
                ],
            )

    def process_single_node(self, node, output_file):
        """
        处理并保存单个节点到文件

        参数:
            node: 要处理的节点
            output_file: 输出文件路径
        """
        if isinstance(node, ImageNode):
            # 对于图片节点，确保包含必要的元数据
            if not node.metadata:
                node.metadata = {}

            # 安全地获取图片路径
            image_path = getattr(node, 'image_path', None)
            if image_path is None:
                image_path = getattr(node, 'image', None)

            # 更新元数据
            metadata_update = {
                "file_type": "image/jpeg",
                "node_type": "image"
            }

            # 只有在有图片路径时才添加文件名
            if image_path is not None:
                metadata_update["file_name"] = os.path.basename(str(image_path))

            # 更新元数据
            for key, value in metadata_update.items():
                node.metadata[key] = value

        else:
            # 对于文本节点，确保包含必要的元数据
            if not node.metadata:
                node.metadata = {}
            node.metadata["node_type"] = "text"

        # 保存节点到文件
        node_json = node.to_dict()
        with open(output_file, 'w') as json_file:
            json.dump(node_json, json_file, indent=2, ensure_ascii=False)
        return True

    def ingestion_example(self, input_file, output_dir):
        """
        处理单个输入文件，将其转换为向量并保存为独立的节点文件

        参数:
            input_file: 输入文件路径
            output_dir: 输出目录路径
        """
        # 转换为绝对路径
        input_file = os.path.abspath(input_file)
        output_dir = os.path.abspath(output_dir)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件名（不含扩展名）
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        print(f"\nProcessing file: {input_file}")

        # 根据文件类型和输入目录前缀处理不同类型的文件
        if self.input_dir.endswith('img'):
            # 处理图片文件
            if input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print("Processing image file...")
                # 创建图片节点并设置元数据
                node = ImageNode(image_path=input_file)
                node.metadata = {
                    "file_path": input_file,
                    "img_path": input_file,
                    "file_type": "image/jpeg",
                    "node_type": "image",
                    "file_name": os.path.basename(input_file)
                }

                if hasattr(self, 'embed_model'):
                    try:
                        node = self.embed_model([node])[0]
                    except Exception as e:
                        print(f"Error generating embedding: {str(e)}")
                        return False

                output_file = os.path.join(output_dir, f"{base_filename}.json")
                return self.process_single_node(node, output_file)
        # else:
        #     # 处理文本文件
        #     if input_file.endswith('.json'):
        #         print("Loading JSON file...")
        #         with open(input_file, 'r', encoding='utf-8') as f:
        #             content_list = json.load(f)

        #         # 处理每个内容块
        #         for idx, content in enumerate(content_list):
        #             # 根据类型确定要向量化的文本内容
        #             text_to_embed = None
        #             if content['type'] in ['text', 'equation']:
        #                 text_to_embed = content.get('text', '')
        #             elif content['type'] in ['image', 'table']:
        #                 # 对于图片和表格，使用caption作为向量化内容
        #                 if content['type'] == 'image':
        #                     text_to_embed = ' '.join(content.get('img_caption', []))
        #                 else:  # table
        #                     text_to_embed = ' '.join(content.get('table_caption', []))

        #             # 如果文本为空，跳过向量化
        #             if not text_to_embed or text_to_embed.strip() == '':
        #                 print(f"Skipping empty content at index {idx}")
        #                 continue

        #             # 创建文本节点
        #             node = TextNode(text=text_to_embed)

        #             # 设置元数据
        #             node.metadata = {
        #                 "file_path": input_file,
        #                 "node_type": content['type'],
        #                 "file_name": os.path.basename(input_file),
        #                 "page_idx": content.get('page_idx', 0),
        #                 "text_level": content.get('text_level', None),
        #                 "text_format": content.get('text_format', None),
        #                 "text": content.get('text', ''),  # 存储原始文本
        #                 "img_path": content.get('img_path', None),
        #                 "img_caption": content.get('img_caption', []),
        #                 "img_footnote": content.get('img_footnote', []),
        #                 "table_caption": content.get('table_caption', []),
        #                 "table_footnote": content.get('table_footnote', []),
        #                 "table_body": content.get('table_body', None)
        #             }

        #             # 确保元数据被正确序列化
        #             node.excluded_embed_metadata_keys = []
        #             node.excluded_llm_metadata_keys = []

        #             if hasattr(self, 'pipeline'):
        #                 try:
        #                     node = self.pipeline.run(nodes=[node], show_progress=False)[0]
        #                 except Exception as e:
        #                     print(f"Error generating embedding: {str(e)}")
        #                     continue

        #             # 保存节点到文件
        #             output_file = os.path.join(output_dir, f"{base_filename}_node_{idx}.json")
        #             self.process_single_node(node, output_file)
                # return True

        print(f"Unsupported file type: {input_file}")
        return False

    def ingestion_multi_session(self):
        """
        批量处理多个文件，支持并行处理
        针对 input_dir 下的每个文档子目录，对其下所有图片文件进行向量化，结果存储到 output_dir/文档名/ 下
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # 只遍历 input_dir 下的文档子目录
        for doc_name in os.listdir(self.input_dir):
            doc_path = os.path.join(self.input_dir, doc_name)
            if not os.path.isdir(doc_path):
                continue  # 跳过非目录

            doc_output_dir = os.path.join(self.output_dir, doc_name)
            os.makedirs(doc_output_dir, exist_ok=True)

            # 检查该文档是否已经处理过
            if os.path.exists(doc_output_dir) and os.listdir(doc_output_dir):
                print(f"文档 {doc_name} 已经处理过，跳过")
                continue

            print(f"处理文档: {doc_name}")

            # 收集该文档下所有图片文件
            file_to_process = []
            for file in os.listdir(doc_path):
                input_file = os.path.join(doc_path, file)
                if not os.path.isfile(input_file):
                    continue
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                base_filename = os.path.splitext(os.path.basename(input_file))[0]
                existing_nodes = [f for f in os.listdir(doc_output_dir) if f.startswith(f"{base_filename}")]
                if not existing_nodes:
                    file_to_process.append(input_file)

            if not file_to_process:
                print(f"文档 {doc_name} 中的所有图片都已处理过")
                continue

            # 单进程处理
            if self.workers == 1:
                for input_file in tqdm(file_to_process, desc=f'处理文档 {doc_name}'):
                    self.ingestion_example(input_file, doc_output_dir)
            else:
                # 多进程并行处理
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    future_to_file = {
                        executor.submit(self.ingestion_example, input_file, doc_output_dir): input_file
                        for input_file in file_to_process
                    }
                    for future in tqdm(as_completed(future_to_file), total=len(file_to_process), desc=f'并行处理文档 {doc_name}'):
                        result_type = future.result()

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='数据向量化处理工具')
    parser.add_argument('--dataset_name', type=str, default='LongDocURL',
                       help='数据集名称 (默认: LongDocURL)')
    parser.add_argument('--input_prefix', type=str, default='img',
                       help='输入数据目录名 (默认: img)')
    parser.add_argument('--output_prefix', type=str, default='colqwen_ingestion',
                       help='输出向量目录名 (默认: colqwen_ingestion)')
    parser.add_argument('--embed_model_name', type=str, default='vidore/colqwen2-v1.0',
                       help='向量化模型名称 (默认: vidore/colqwen2-v1.0)')
    parser.add_argument('--workers', type=int, default=5,
                       help='并行处理的工作进程数 (默认: 5)')

    # 解析命令行参数
    args = parser.parse_args()

    # 设置数据集路径
    dataset_name = args.dataset_name
    print(f"使用数据集: {dataset_name}")

    # 处理每个数据集
    dataset_dir = os.path.join(base_dir_prefix, dataset_name)

    # 选择向量化模型
    ingestion = Ingestion(
        dataset_dir,
        input_prefix=args.input_prefix,
        output_prefix=args.output_prefix,
        embed_model_name=args.embed_model_name
    )

    # 设置工作进程数
    ingestion.workers = args.workers

    # 执行向量化处理
    ingestion.ingestion_multi_session()