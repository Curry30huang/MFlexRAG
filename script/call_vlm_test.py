from openai import OpenAI
import base64
import os
from typing import List, Any

project_dir = os.getcwd()
image_file = f"{project_dir}/images/1.jpg"

with open(image_file, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')
    image_str = f"data:image/jpeg;base64,{image_data}"

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

messages_content:List[Any] = [
    {"type": "text", "text": "Give me a short introduction to the image."}
]

messages_content.append({
    "type": "image_url",
    "image_url": {
        "url": image_str,
        "detail": "high"
    },
})

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{"role": "user", "content": messages_content}],
    stream=False
)

print(response.choices[0].message.content)