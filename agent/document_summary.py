import os
from openai import OpenAI
import base64
from agent.page_map_dict_normal import page_map_dict_normal
from typing import Any, List, Optional, Dict
import json
import re

class DocumentSummaryAgent():
    """
    文档摘要Agent，利用MLLM对选择的全部信息进行文档级别摘要
    """

    def __init__(self, api_key: str, model_name: str, model_base_url: str) -> None:
        # 加载OpenAI标准的客户端
        self.client = OpenAI(api_key=api_key, base_url=model_base_url)
        self.model_name = model_name
        self.page_map = page_map_dict_normal  # 多图像模式映射,最多支持20张图像
        # 认为执行的工作目录为项目根目录
        self.prompt_file = os.path.join(os.getcwd(),'prompt','document_summary.txt')
        # 读取prompt文件
        with open(self.prompt_file, 'r') as f:
            self.prompt = f.read()
        # 最大重试次数
        self.max_retries = 3

    def summary(self, doc_name: str, query: str, image_refined_path_list: List[str], image_analysis_list: List[str]) -> Optional[Dict[str, Optional[str]]]:
        """
        对选择的全部信息进行文档级别摘要

        Args:
            doc_name: 文档名称
            query: 查询语句
            image_refined_path_list: 精调后的图像路径列表
            image_analysis_list: 图像分析list,根据图片下标排序的
        Returns:
            summary_result: 摘要结果,包含字段:
                document_summary: 文档摘要内容
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
        # 将image_analysis_list转换为IMAGE_ANALYSIS，每个描述前需要加上 image i: 前缀才行
        image_analysis_str = "\n".join([f"image {i}: {analysis}" for i, analysis in enumerate(image_analysis_list)])

        prompt = self.prompt.format(
            IMAGE_LAYOUT_MAPPING=self.page_map[len(images)],
            USER_QUERY=query,
            IMAGE_ANALYSIS=image_analysis_str,
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
                            "detail": "low" # low 或者 high
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
                    print(f"文档摘要失败，返回结果为空 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return None
                    continue

                parsed_result = self._parse_summary_response(answer)

                # 检查解析结果是否有效
                if parsed_result and parsed_result.get('document_summary') is not None:
                    return parsed_result
                else:
                    print(f"解析结果无效，尝试重试 (尝试 {attempt + 1}/{self.max_retries})")
                    if attempt == self.max_retries - 1:
                        return parsed_result
                    continue

            except Exception as e:
                print(f"文档摘要过程中出现错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise e
                continue

        return None

    def _parse_summary_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        解析文档摘要响应，提取关键信息标签
        标签包括：document_summary
        """
        result: Dict[str, Optional[str]] = {
            'document_summary': None
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

            result['document_summary'] = parsed_json.get('document_summary')

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response_text}")
            # 如果JSON解析失败，尝试使用正则表达式作为后备方案
            self._fallback_parse(response_text, result)

            # 检查后备解析是否成功
            if result['document_summary'] is None:
                raise ValueError("JSON解析失败且后备解析也失败")

        except Exception as e:
            print(f"解析响应时出现错误: {e}")
            self._fallback_parse(response_text, result)

            # 检查后备解析是否成功
            if result['document_summary'] is None:
                raise ValueError("解析响应失败且后备解析也失败")

        return result

    def _fallback_parse(self, response_text: str, result: Dict[str, Optional[str]]):
        """
        后备解析方法，使用正则表达式提取字段
        """
        document_summary_match = re.search(r'"document_summary":\s*"([^"]*)"', response_text, re.DOTALL)
        if document_summary_match:
            result['document_summary'] = document_summary_match.group(1).strip()