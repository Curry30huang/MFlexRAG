from openai import OpenAI
import os
import base64
from typing import Any, List, Optional, Dict
from agent.page_map_dict_normal import page_map_dict_normal
import json
import re

class ImageResortAgent():
    """
    图像重排序Agent，利用MLLM根据query对检索出来的图像进行重排序
    """

    def __init__(self, api_key: str, model_name: str, model_base_url: str) -> None:
        # 加载OpenAI标准的客户端
        self.client = OpenAI(api_key=api_key, base_url=model_base_url)
        self.model_name = model_name
        # 认为执行的工作目录为项目根目录
        self.prompt_file = os.path.join(os.getcwd(),'prompt','image_resort.txt')
        self.page_map = page_map_dict_normal  # 多图像模式映射,最多支持20张图像
        # 读取prompt文件
        with open(self.prompt_file, 'r') as f:
            self.prompt = f.read()
        # 最大重试次数
        self.max_retries = 3

    def resort(self, doc_name: str, query: str, image_path_list: List[str]) -> Optional[Dict[str, Any]]:
        """
        根据query使用LLM对页面进行重排序
        Args:
            doc_name: 文档名称
            query: 查询语句
            image_path_list: 图像路径列表
        Returns:
            resort_result: 重排序结果,包含字段:
                image_analysis: 图像分析
                selected_images: 选择的图像索引列表
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
        prompt = self.prompt.format(
            IMAGE_LAYOUT_MAPPING=self.page_map[len(images)],
            USER_QUERY=query
        )

        # 实现错误重试机制
        for attempt in range(self.max_retries):
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
                            "detail": "high" # low 或者 high
                        },
                    })

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": messages_content}
                    ],
                    stream=False
                )
                answer = response.choices[0].message.content

                # 解析answer，提取关键信息标签
                if answer is None:
                    print(f"图像重排序失败，返回结果为空 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return None
                    continue

                parsed_result = self._parse_resort_response(answer)

                # 检查解析结果是否有效
                if parsed_result and parsed_result.get('image_analysis') is not None:
                    return parsed_result
                else:
                    print(f"解析结果无效，尝试重试 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return parsed_result
                    continue

            except Exception as e:
                print(f"图像重排序过程中出现错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise e
                continue

        return None

    def _parse_resort_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析图像重排序响应，提取关键信息标签
        标签包括：image_analysis, selected_images
        """
        result: Dict[str, Any] = {
            'image_analysis': None,
            'selected_images': []
        }

        try:
            # 尝试解析JSON格式的响应
            # 查找JSON代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_json = json.loads(json_str)
            else:
                # 如果没有找到JSON代码块，尝试直接解析整个响应
                parsed_json = json.loads(response_text)

            # 提取image_analysis字段 - 现在是一个JSON对象
            image_analysis = parsed_json.get('image_analysis')
            if isinstance(image_analysis, dict):
                result['image_analysis'] = image_analysis
            elif isinstance(image_analysis, str):
                # 如果image_analysis是字符串，尝试解析为JSON对象
                try:
                    result['image_analysis'] = json.loads(image_analysis)
                except json.JSONDecodeError:
                    # 如果解析失败，保持为字符串
                    result['image_analysis'] = image_analysis
            else:
                result['image_analysis'] = image_analysis

            # 处理selected_images字段 - 可能是列表或字符串
            selected_images = parsed_json.get('selected_images')
            if isinstance(selected_images, list):
                result['selected_images'] = selected_images
            elif isinstance(selected_images, str):
                # 如果是字符串，尝试解析为列表
                try:
                    # 移除方括号并分割
                    clean_str = selected_images.strip('[]')
                    if clean_str:
                        result['selected_images'] = [int(x.strip()) for x in clean_str.split(',')]
                    else:
                        result['selected_images'] = []
                except:
                    result['selected_images'] = []
            else:
                result['selected_images'] = []

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response_text}")
            # 如果JSON解析失败，尝试使用正则表达式作为后备方案
            self._fallback_parse(response_text, result)

            # 检查后备解析是否成功
            if result['image_analysis'] is None:
                raise ValueError("JSON解析失败且后备解析也失败")

        except Exception as e:
            print(f"解析响应时出现错误: {e}")
            self._fallback_parse(response_text, result)

            # 检查后备解析是否成功
            if result['image_analysis'] is None:
                raise ValueError("解析响应失败且后备解析也失败")

        return result

    def _fallback_parse(self, response_text: str, result: Dict[str, Any]):
        """
        后备解析方法，使用正则表达式提取字段
        """
        # 尝试提取image_analysis - 现在是一个JSON对象
        image_analysis_match = re.search(r'"image_analysis":\s*(\{[^}]*\})', response_text, re.DOTALL)
        if image_analysis_match:
            try:
                image_analysis_json = image_analysis_match.group(1)
                result['image_analysis'] = json.loads(image_analysis_json)
            except json.JSONDecodeError:
                # 如果JSON解析失败，尝试提取字符串格式
                image_analysis_str_match = re.search(r'"image_analysis":\s*"([^"]*)"', response_text, re.DOTALL)
                if image_analysis_str_match:
                    result['image_analysis'] = image_analysis_str_match.group(1).strip()

        # 尝试提取selected_images
        selected_images_match = re.search(r'"selected_images":\s*\[([^\]]*)\]', response_text)
        if selected_images_match:
            try:
                indices_str = selected_images_match.group(1)
                if indices_str.strip():
                    result['selected_images'] = [int(x.strip()) for x in indices_str.split(',')]
                else:
                    result['selected_images'] = []
            except:
                result['selected_images'] = []



