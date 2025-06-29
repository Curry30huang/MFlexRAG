from openai import OpenAI
import base64
import os
from typing import List, Any

project_dir = os.getcwd()
# image_file = f"{project_dir}/images/1.jpg"

image_file = "/home/huangjiayu/MFlexRAG/data_process/data/LongDocURL/img/4117236/79.jpg"

with open(image_file, 'rb') as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')
    image_str = f"data:image/jpeg;base64,{image_data}"

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

prompt = """
Based on the question, please carefully examine each image and provide descriptions from the perspective of solving the problem.

*Question: Which street features 'Sugar Maple' in 'Good' condition and 'Signs of Stress' in observations?*

The number of images is 1.

Please focus on information relevant to the question and avoid irrelevant descriptions to help answer the question more accurately.

IMPORTANT: Only describe the images that are actually provided to you. Do not fabricate descriptions for images that were not included in the input.

Please format your response as follows:
Image 0: [Question-based description of the first image]
Image 1: [Question-based description of the second image]
...and so on for each image provided.
"""

messages_content:List[Any] = [
    {"type": "text", "text": prompt}
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