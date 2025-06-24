from utils.openai_helper import initialize_client
import os
import base64
from page_map_dict_normal import page_map_dict_normal
from typing import Any, List


class ReasonerAgent():
    """
    推理Agent，利用MLLM对选择的全部信息进行推理
    """

    def __init__(self,api_key_file,model_name,model_base_url):
        # 加载OpenAI标准的客户端
        self.client = initialize_client(api_key_file,model_name,model_base_url)
        self.page_map = page_map_dict_normal  # 多图像模式映射,最多支持20张图像
        # 认为执行的工作目录为项目根目录
        self.prompt_file = os.path.join(os.getcwd(),'prompt','reasoner.txt')
        # 读取prompt文件
        with open(self.prompt_file, 'r') as f:
            self.prompt = f.read()

    def reason(self,doc_name,query,image_refined_path_list,document_summary):
        """
        对选择的全部信息进行推理
        """

        # 读取图像，并将其转换为base64格式的列表
        images = []
        for image_path in image_refined_path_list:
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_str = f"data:image/jpeg;base64,{image_data}"
                images.append(image_str)

        # TODO: 后续加上最大token数量限制，防止token数量过多

        # 构建prompt
        prompt = self.prompt.format(doc_name=doc_name,query=query,page_map=self.page_map[len(images)],document_summary=document_summary)

        try:
            messages_content:List[Any] = [
                {"type": "text", "text": prompt}
            ]

            # 追加图像
            for image in images:
                messages_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                        "detail": "low" # low 或者 high
                    },
                })

            response = self.client.create(
                messages=[
                    {"role": "user", "content": messages_content}
                ],

            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
            return None

        # TODO: 解析answer，保证格式正确，然后返回正确结果

        return answer