from openai import OpenAI
import os
import base64
from agent.page_map_dict_normal import page_map_dict_normal
from typing import Any, List, Optional, Dict
import json
import re


class ReasonerAgent():
    """
    推理Agent，利用MLLM对选择的全部信息进行推理
    """

    def __init__(self, api_key: str, model_name: str, model_base_url: str) -> None:
        # 加载OpenAI标准的客户端
        self.client = OpenAI(api_key=api_key, base_url=model_base_url)
        self.model_name = model_name
        self.page_map = page_map_dict_normal  # 多图像模式映射,最多支持20张图像
        # 认为执行的工作目录为项目根目录
        self.prompt_file = os.path.join(os.getcwd(),'prompt','reasoner.txt')
        # 读取prompt文件
        with open(self.prompt_file, 'r') as f:
            self.prompt = f.read()

    def reason(self, doc_name: str, query: str, image_refined_path_list: List[str], image_analysis_list: List[str], document_summary: str) -> Optional[Dict[str, Optional[str]]]:
        """
        对选择的全部信息进行推理

        Args:
            doc_name: 文档名称
            query: 查询语句
            image_refined_path_list: 精调后的图像路径列表
            image_analysis_list: 图像分析list,根据图片下标排序的
            document_summary: 文档摘要
        Returns:
            reasoning_result: 推理结果,包含字段:
                scratchpad: 分析过程
                response_type: 响应类型 (answer|not_answerable|query_update)
                answer: 答案内容 (仅当response_type为answer时)
                not_answerable: 不可回答说明 (仅当response_type为not_answerable时)
                query_update: 查询更新 (仅当response_type为query_update时)
                notes: 备注说明 (仅当response_type为query_update时)
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
            DOCUMENT_SUMMARY=document_summary,
            IMAGE_ANALYSIS=image_analysis_str,
        )

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
        except Exception as e:
            print(f"推理过程中出现错误: {e}")
            return None

        # 解析answer，提取关键信息标签
        if answer is None:
            print(f"推理失败，返回结果为空")
            return None

        parsed_result = self._parse_reasoning_response(answer)
        return parsed_result

    def _parse_reasoning_response(self, response_text: str) -> Dict[str, Optional[str]]:
        """
        解析推理响应，提取关键信息标签
        标签包括：scratchpad, response_type, answer, not_answerable, query_update, notes
        """
        result: Dict[str, Optional[str]] = {
            'scratchpad': None,
            'response_type': None,
            'answer': None,
            'not_answerable': None,
            'query_update': None,
            'notes': None
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

            # 提取所有字段
            result['scratchpad'] = parsed_json.get('scratchpad')
            result['response_type'] = parsed_json.get('response_type')
            result['answer'] = parsed_json.get('answer')
            result['not_answerable'] = parsed_json.get('not_answerable')
            result['query_update'] = parsed_json.get('query_update')
            result['notes'] = parsed_json.get('notes')

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response_text}")
            # 如果JSON解析失败，尝试使用正则表达式作为后备方案
            self._fallback_parse(response_text, result)
        except Exception as e:
            print(f"解析响应时出现错误: {e}")
            self._fallback_parse(response_text, result)

        return result

    def _fallback_parse(self, response_text: str, result: Dict[str, Optional[str]]):
        """
        后备解析方法，使用正则表达式提取字段
        """
        # 尝试提取scratchpad
        scratchpad_match = re.search(r'"scratchpad":\s*"([^"]*)"', response_text, re.DOTALL)
        if scratchpad_match:
            result['scratchpad'] = scratchpad_match.group(1).strip()

        # 尝试提取response_type
        response_type_match = re.search(r'"response_type":\s*"([^"]*)"', response_text)
        if response_type_match:
            result['response_type'] = response_type_match.group(1).strip()

        # 尝试提取answer
        answer_match = re.search(r'"answer":\s*"([^"]*)"', response_text, re.DOTALL)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()

        # 尝试提取not_answerable
        not_answerable_match = re.search(r'"not_answerable":\s*"([^"]*)"', response_text, re.DOTALL)
        if not_answerable_match:
            result['not_answerable'] = not_answerable_match.group(1).strip()

        # 尝试提取query_update
        query_update_match = re.search(r'"query_update":\s*"([^"]*)"', response_text, re.DOTALL)
        if query_update_match:
            result['query_update'] = query_update_match.group(1).strip()

        # 尝试提取notes
        notes_match = re.search(r'"notes":\s*"([^"]*)"', response_text, re.DOTALL)
        if notes_match:
            result['notes'] = notes_match.group(1).strip()