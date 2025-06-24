"""
OpenAI助手模块
提供用于初始化和配置OpenAI客户端的工具函数
"""

from autogen import OpenAIWrapper

def initialize_client(api_key_file, model_name, base_url, cache_seed=42):
    """
    初始化OpenAI客户端包装器并配置参数列表

    参数:
        - api_key_file (str): API密钥文件路径
        - model (str): 模型名称
        - base_url (str): API基础URL
        - cache_seed (int): 缓存种子

    返回:
        OpenAIWrapper: 配置好的OpenAI客户端包装器实例，如果初始化失败则返回None

    说明:
        此函数负责从文件中读取API密钥，并创建配置好的OpenAI客户端包装器。
        支持自定义基础URL和模型名称，便于在不同环境中使用不同的配置。
    """

        # 初始化API密钥变量
    api_key = ""

    try:
        # 从指定文件中读取API密钥
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
    except Exception as e:
        # 如果读取API密钥文件失败，打印错误信息并返回None
        print(f"Error reading API key file: {e}")
        return None

        # 创建配置列表，包含模型、基础URL、API密钥等信息
    config_list = [
        {
            "model": model_name,  # 使用自定义模型名或默认模型名
            "base_url": base_url,  # 使用自定义URL或默认URL
            "api_key": api_key,  # API密钥
            "api_type": "openai",  # API类型
            "price": [0.08/1000, 0.24/1000]  # 价格配置（输入和输出token的价格）
        }
    ]

    # 创建并返回OpenAI客户端包装器实例
    return OpenAIWrapper(config_list=config_list, cache_seed=cache_seed)


