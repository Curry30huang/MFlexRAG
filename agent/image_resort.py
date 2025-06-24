from utils.openai_helper import initialize_client
import os
import base64
from typing import Any, List
from page_map_dict_normal import page_map_dict_normal

class ImageResortAgent():
    """
    图像重排序Agent，利用MLLM根据query对检索出来的图像进行重排序
    """

    def __init__(self,api_key_file,model_name,model_base_url):
        # 加载OpenAI标准的客户端
        self.client = initialize_client(api_key_file,model_name,model_base_url)
        # 认为执行的工作目录为项目根目录
        self.prompt_file = os.path.join(os.getcwd(),'prompt','image_resort.txt')
        self.page_map = page_map_dict_normal  # 多图像模式映射,最多支持20张图像
        # 读取prompt文件
        with open(self.prompt_file, 'r') as f:
            self.prompt = f.read()

    def resort(self,doc_name,query,image_path_list):
        """
        根据query对检索出来的图像进行重排序
        """
        # 使用LLM对页面进行重排序
        answer = self.resort_by_llm(doc_name,query,image_path_list)
        return answer

    def resort_by_llm(self,doc_name,query,image_path_list):
        """
        使用LLM对页面进行重排序
        """


        # 读取图像，并将其转换为base64格式的列表
        images = []
        for image_path in image_path_list:
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                image_str = f"data:image/jpeg;base64,{image_data}"
                images.append(image_str)

        # TODO: 后续加上最大token数量限制，防止token数量过多

        # 构建prompt
        prompt = self.prompt.format(query=query,page_map=self.page_map[len(images)])

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
            print(f"图像重排序过程中出现错误: {e}")
            return None

        # TODO: 解析answer，保证格式正确，然后返回正确结果

        return answer



