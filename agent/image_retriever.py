from agent.search_engine import SearchEngine
from utils.format_converter import nodes2dict

class ImageRetrieverAgent():
    """
    图像检索代理Agent，根据query 检索指定的文档页面
    """
    def __init__(self, dataset, node_dir_prefix, embed_model_name,gmm=False,input_gmm=20,max_output_gmm=10,min_output_gmm=5):
        """
        初始化检索代理

        Args:
            args: 配置参数对象，包含模型配置、路径等信息
            **kwargs: 其他传递给父类的参数
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


    def search(self, doc_name, query):
        """
        根据query 检索指定的文档页面
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
