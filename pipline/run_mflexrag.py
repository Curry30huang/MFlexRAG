from agent.image_retriever import ImageRetrieverAgent
from agent.image_resort import ImageResortAgent
from agent.document_summary import DocumentSummaryAgent
from agent.reasoner import ReasonerAgent
from typing import List, Optional, Dict, Any

class MflexRAG:
    """
    MflexRAG 的pipeline ，编排agent的执行顺序和流程
    """
    def __init__(self, dataset: str, node_dir_prefix: str, embed_model_name: str, api_key: str, model_name: str, model_base_url: str, gmm: bool = False, input_gmm: int = 20, max_output_gmm: int = 10, min_output_gmm: int = 5) -> None:
        # 初始化参数
        self.dataset = dataset # 数据集名称
        self.node_dir_prefix = node_dir_prefix # 节点目录前缀
        self.embed_model_name = embed_model_name # 嵌入模型名称
        self.gmm = gmm # 是否使用gmm
        self.input_gmm = input_gmm # gmm输入长度
        self.max_output_gmm = max_output_gmm # gmm最大输出长度
        self.min_output_gmm = min_output_gmm # gmm最小输出长度

        # 初始化各种agent
        self.image_retriever_agent = ImageRetrieverAgent(dataset, node_dir_prefix, embed_model_name,gmm,input_gmm,max_output_gmm,min_output_gmm)
        self.image_resort_agent = ImageResortAgent(api_key,model_name,model_base_url)
        self.document_summary_agent = DocumentSummaryAgent(api_key,model_name,model_base_url)
        self.reasoner_agent = ReasonerAgent(api_key,model_name,model_base_url)

    def run_one_doc(self, doc_name: str, query: str) -> None:
        """
        执行MflexRAG 的pipeline，对单个文档进行处理
        """

        # 最大循环次数
        max_loop_times = 3

        # 初始化循环次数
        loop_times = 0

        # 现在认为每次迭代，只有query是变化的，其他变量都需要重新计算
        original_query:str = query # 原始查询语句

        while loop_times < max_loop_times:
            # 1. 图像检索
            image_node_list = self.image_retriever_agent.search(doc_name, query)
            if image_node_list is None or len(image_node_list) == 0:
                print(f"图像检索失败，跳出循环")
                break

            # 将image_node_list转换为image_path_list，获取对应的图像路径
            image_path_list = [node.get('node',{}).get('metadata',{}).get('file_path') for node in image_node_list]
            if image_path_list is None or len(image_path_list) == 0:
                print(f"图像路径获取失败，跳出循环")
                break
            # 只取前面几个图像，启动模型是限制了最大读取图像数量为10
            image_path_list = image_path_list[:5]

            print(f"图像检索成功，图像数量: {len(image_path_list)}")
            print(f"初始图像路径: {image_path_list}")

            # 2. 图像重排序
            resort_result = self.image_resort_agent.resort(doc_name, query, image_path_list)
            if resort_result is None:
                print(f"图像重排序失败，跳出循环")
                break

            # 将筛选并重排序之后的图片选择出来
            selected_image_index_list:List[int] = resort_result.get('selected_images',[])
            selected_image_path_list = [image_path_list[i] for i in selected_image_index_list]
            image_path_list = selected_image_path_list
            # 选择选择图像的原因
            selected_image_reason_list = [resort_result.get('image_analysis',{}).get(f'image {i}',f'image {i} is not relevant') for i in selected_image_index_list]


            print(f"图像重排序成功，图像数量: {len(image_path_list)}")
            print(f"重排序后的图像路径: {image_path_list}")
            print(f"重排序后的图像原因: {selected_image_reason_list}")

            # 3. 文档摘要
            summary_result = self.document_summary_agent.summary(doc_name, query, image_path_list,selected_image_reason_list)
            if summary_result is None:
                print(f"文档摘要失败，跳出循环")
                break

            summary_content_temp = summary_result.get('document_summary',"")
            summary_content = summary_content_temp if summary_content_temp is not None else ""

            print(f"文档摘要成功，摘要内容: {summary_content}")

            # 4. 推理
            reasoning_result = self.reasoner_agent.reason(doc_name, query, image_path_list,selected_image_reason_list,summary_content)
            if reasoning_result is None:
                print(f"推理失败，跳出循环")
                break

            print(f"推理成功，推理结果: {reasoning_result}")

            # 判断如果推理结果的response_type为answer，则则结束循环
            if reasoning_result.get('response_type') == 'answer':
                print(f"推理结果为answer，结束循环")
                break

            # 判断如果推理结果的response_type为not_answerable，则结束循环
            if reasoning_result.get('response_type') == 'not_answerable':
                print(f"推理结果为not_answerable，结束循环")
                break

            # 判断如果推理结果的response_type为query_update，则更新查询语句
            if reasoning_result.get('response_type') == 'query_update':
                print(f"推理结果为query_update，更新查询语句")
                query_update = reasoning_result.get('query_update',"")
                if query_update is not None:
                    query = query_update
                else:
                    print(f"推理结果为query_update，但query_update为空，跳出循环")
                    break

            loop_times += 1


if __name__ == "__main__":
    # 初始化MflexRAG
    mflexrag = MflexRAG(dataset="test", node_dir_prefix="colqwen_ingestion", embed_model_name="vidore/colqwen2-v1.0",api_key="EMPTY",model_name="Qwen/Qwen2.5-VL-7B-Instruct",model_base_url="http://localhost:8001/v1")

    print("初始化MflexRAG完成")

    # 执行MflexRAG
    mflexrag.run_one_doc(doc_name="123", query="where is Figure 2: Data Construction pipeline?")

