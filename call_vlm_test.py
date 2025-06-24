from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": "Give me a short introduction to large language models."}],
    stream=False
)

print(response.choices[0].message.content)