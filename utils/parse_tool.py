import json
import re


def parse_tool_output(output):
    """
    解析工具输出的JSON数据
    Args:
        output: 工具输出的字符串
    Returns:
        parsed_tools: 解析后的工具数据列表
    """
    tool_pattern = r"<action>\{(.*?)\}</action>"
    matches = re.findall(tool_pattern, output, re.DOTALL)
    parsed_tools = []
    for match in matches:
        try:
            # 尝试解析JSON数据
            tool_data = json.loads(match.strip())
            parsed_tools.append(tool_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return parsed_tools

def extract_json(select_response):
    """
    从响应中提取JSON数据
    Args:
        select_response: 包含JSON的响应字符串
    Returns:
        json_data: 解析后的JSON数据
    """
    select_response = select_response.replace('```json', '').replace('```', '')
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        return json.loads(json_str)
    else:
        return json.loads(select_response)