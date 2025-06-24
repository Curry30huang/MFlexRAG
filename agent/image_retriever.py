from agent.search_engine import SearchEngine
from utils.format_converter import nodes2dict
from typing import List, Optional, Dict, Any

class ImageRetrieverAgent():
    """
    图像检索代理Agent，根据query 检索指定的文档页面
    """
    def __init__(self, dataset: str, node_dir_prefix: str, embed_model_name: str, gmm: bool = False, input_gmm: int = 20, max_output_gmm: int = 10, min_output_gmm: int = 5) -> None:
        """
        初始化检索代理

        Args:
            dataset: 数据集名称
            node_dir_prefix: 节点目录前缀
            embed_model_name: 嵌入模型名称
            gmm: 是否使用gmm
            input_gmm: gmm输入长度
            max_output_gmm: gmm最大输出长度
            min_output_gmm: gmm最小输出长度
        """
        # 初始时不加载search_engine，因为doc_name 可能不固定
        self.search_engine = None
        self.dataset = dataset
        self.node_dir_prefix = node_dir_prefix
        self.embed_model_name = embed_model_name
        self.gmm = gmm
        self.input_gmm = input_gmm
        self.max_output_gmm = max_output_gmm
        self.min_output_gmm = min_output_gmm


    def search(self, doc_name: str, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        根据query 检索指定的文档页面
        Args:
            doc_name: 文档名称
            query: 查询语句
        Returns:
            image_node_list: 图像节点列表
        """
        # 在第一次搜索时加载search_engine
        if self.search_engine is None:
            self.search_engine = SearchEngine(self.dataset, doc_name, self.node_dir_prefix, self.embed_model_name)
            if self.gmm:
                self.search_engine.gmm = True
                self.search_engine.input_gmm = self.input_gmm
                self.search_engine.max_output_gmm = self.max_output_gmm
                self.search_engine.min_output_gmm = self.min_output_gmm
        else:
            self.search_engine.change_document(doc_name)
        nodes = self.search_engine.search(query)
        nodes_dict = nodes2dict(nodes) if not isinstance(nodes,dict) else nodes
        return nodes_dict.get('source_nodes',[])
