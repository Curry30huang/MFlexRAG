from openai import OpenAI
from dotenv import load_dotenv
import os

project_dir = os.getcwd()

def load_environment_variables():
    """加载.env文件中的环境变量"""
    # 尝试从项目根目录加载.env文件
    env_path = os.path.join(project_dir, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"已从 {env_path} 加载环境变量")
    else:
        # 如果项目根目录没有.env文件，尝试从当前目录加载
        load_dotenv()
        print("已从当前目录的.env文件加载环境变量（如果存在）")

load_environment_variables()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")

response = client.chat.completions.create(
    model="openai/gpt-4.1",
    messages=[{"role": "user", "content": "Who are you? GPT or Claude?"}],
    stream=False
)

print(response.choices[0].message.content)