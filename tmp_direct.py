from openai import OpenAI
from typing import List, Any, Dict, Union
import base64

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

images_path_list = [
    '/home/huangjiayu/MFlexRAG/data_process/data/LongDocURL/img/4106951/72.jpg',
    '/home/huangjiayu/MFlexRAG/data_process/data/LongDocURL/img/4106951/73.jpg',
    '/home/huangjiayu/MFlexRAG/data_process/data/LongDocURL/img/4106951/74.jpg',
    '/home/huangjiayu/MFlexRAG/data_process/data/LongDocURL/img/4106951/75.jpg',
]

images = []
for image_path in images_path_list:
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        image_str = f"data:image/jpeg;base64,{image_data}"
        images.append(image_str)

# 构建包含文本和图像的内容
content: List[Dict[str, Any]] = [
    {"type": "text", "text": "Which section best matches the follwing description: \n<description>This text primarily discusses David Lack's scientific research on robins, as published in \"The Life of the Robin.\" The passage details robin behavior, including territorial habits, mating patterns, and survival rates. It explains how male robins maintain territories, their courtship behaviors, and how females choose mates. The research includes interesting experiments with stuffed birds and describes mortality rates and causes of death among robins. The text also notes that most scientific work is published in specialized journals, and concludes by encouraging more people to engage in similar scientific observation, emphasizing that valuable research can be conducted even by non-professionals in their spare time.</description>\nSelect titles from the doc that best answer the question, do not alter or analyze the titles themselves."}
]

# 追加图像
for image in images:
    content.append({
        "type": "image_url",
        "image_url": {
            "url": image,
            "detail": "high"
        },
    })

messages = [
    {"role": "user", "content": content},  # type: ignore
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=messages,  # type: ignore
    stream=False
)

print(response.choices[0].message.content)