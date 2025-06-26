import os
import argparse
import json
import base64
from typing import List, Dict, Any, Optional
from openai import OpenAI

project_dir = os.getcwd()
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    读取数据集文件的json数组
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    return data_list

def process_data(data_list: List[Dict[str, Any]],dataset_name:str, output_dir: str)->List[Dict[str, Any]]:
    """
    处理数据集，根据数据集名称调用不同的处理函数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = f"{output_dir}/golden_direct_test.jsonl"
    # 读取文件最后一行，将JSON 转换为dict
    last_json = None
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            last_line = f.readlines()[-1]
            last_json = json.loads(last_line)

    if dataset_name == "LongDocURL":
        # 判断已经完成的JSON，只处理没有做过的题目
        # 在data_list中，定位last_json的question_id，然后从这个下一个开始处理
        if last_json is not None:
            for index, data in enumerate(data_list):
                if data['question_id'] == last_json['question_id']:
                    data_list = data_list[index+1:]
                    break
        return process_longdocurl_data(data_list,dataset_name, output_dir)
    elif dataset_name == "MMLongBench":
        # 判断已经完成的JSON，只处理没有做过的题目
        # 在data_list中，定位last_json的doc_id+question，然后从这个下一个开始处理
        if last_json is not None:
            for index, data in enumerate(data_list):
                if data['doc_id'] == last_json['doc_id'] and data['question'] == last_json['question']:
                    data_list = data_list[index+1:]
                    break
        return process_mmlongbench_data(data_list,dataset_name, output_dir)

    raise ValueError(f"不支持的数据集名称: {dataset_name}")

def call_vlm_model(question:str,image_path_list:List[str])->str:
    """
    调用VLM模型，返回预测答案
    """
    # 将图片全部转换为base64
    images = []
    for image_path in image_path_list:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            image_str = f"data:image/jpeg;base64,{image_data}"
            images.append(image_str)

    messages_content:List[Any] = [
        {"type": "text", "text": question}
    ]

    for image in images:
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": image,
                "detail": "high"
            },
        })

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {"role": "user", "content": messages_content}
        ],
        stream=False
    )
    predict_answer = response.choices[0].message.content
    if predict_answer is None:
        raise ValueError(f"预测答案为空")
    return predict_answer

# 处理 LongDocURL 数据集，直接调用金标签，测试模型直接输出答案的能力
def process_longdocurl_data(data_list: List[Dict[str, Any]],dataset_name:str, output_dir: str)->List[Dict[str, Any]]:
    """
    处理 LongDocURL 数据集，直接调用金标签，测试模型直接输出答案的能力，判断是否核心就在召回。因为7B模型输入详细的上下文描述反而会让模型回答混乱
    """
    result_list = []

    for data in data_list:
        question_id = data['question_id']
        question = data['question']
        answer = data['answer']
        evidence_pages = data['evidence_pages']
        doc_no = data['doc_no']

        # 读取doc_no对应的图片
        image_dir = f"{project_dir}/data_process/data/{dataset_name}/img/{doc_no}"
        # 读取 evidence_pages 对应的图片
        image_files_path_list = []
        for page_no in evidence_pages:
            image_files_path_list.append(f"{image_dir}/{page_no}.jpg")

        # 只能取前10个图片
        predict_answer = call_vlm_model(question,image_files_path_list[:10])

        # 构造结果对象
        result_obj = {
            "question_id": question_id,
            "answer": answer,
            "predict_answer": predict_answer
        }

        # 立即保存单个结果
        success = save_result(result_obj, output_dir)
        if success:
            result_list.append(result_obj)
            print(f"question_id: {question_id} 已经完成并保存")
        else:
            print(f"question_id: {question_id} 保存失败，跳过")

    print(f"所有问题已经完成")
    return result_list

def process_mmlongbench_data(data_list: List[Dict[str, Any]],dataset_name:str, output_dir: str)->List[Dict[str, Any]]:
    """
    处理 MMLongBench 数据集，直接调用金标签，测试模型直接输出答案的能力
    """
    result_list = []
    for data in data_list:
        question = data['question']
        answer = data['answer']
        evidence_pages = data['evidence_pages']
        doc_id = data['doc_id']
        # 这个 evidence_pages 从JSON文件中读取得到的是字符串，需要自己将这个字符串变为列表
        evidence_pages = json.loads(evidence_pages)

        # 读取doc_id对应的图片
        image_dir = f"{project_dir}/data_process/data/{dataset_name}/img/{doc_id}"
        # 读取 evidence_pages 对应的图片
        image_files_path_list = []
        for page_no in evidence_pages:
            image_files_path_list.append(f"{image_dir}/{page_no}.jpg")

        # 只能取前10个图片
        predict_answer = call_vlm_model(question,image_files_path_list[:10])

        # 构造结果对象
        result_obj = {
            "question": question,
            "doc_id": doc_id,
            "answer": answer,
            "predict_answer": predict_answer
        }

        # 立即保存单个结果
        success = save_result(result_obj, output_dir)
        if success:
            result_list.append(result_obj)
            print(f"doc_id: {doc_id} 已经完成并保存")
        else:
            print(f"doc_id: {doc_id} 保存失败，跳过")

    print("所有问题已经完成")
    return result_list

def save_result(result_obj: Dict[str, Any], output_dir: str) -> bool:
    """
    将单个结果对象追加保存为jsonl文件

    Args:
        result_obj: 要保存的单个结果对象
        output_dir: 输出目录路径

    Returns:
        bool: 保存成功返回True，失败返回False
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = f"{output_dir}/golden_direct_test.jsonl"

        # 追加模式写入单个JSON对象
        with open(output_file_path, 'a', encoding='utf-8') as f:
            json.dump(result_obj, f, ensure_ascii=False)
            f.write('\n')  # 添加换行符

        return True

    except Exception as e:
        print(f"保存结果时发生错误: {e}")
        return False

"""
直接采用金标签，测试模型直接输出答案的能力，判断是否核心就在召回。因为7B模型输入详细的上下文描述反而会让模型回答混乱
"""
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='处理JSON文件并转换文档路径')
    parser.add_argument('--dataset_name', type=str, default='LongDocURL',
                       help='数据集名称 (默认: LongDocURL)')
    parser.add_argument('--path_prefix', type=str,
                       default='/home/huangjiayu/Mdocagent-dataset',
                       help='数据集根目录路径 (默认: /home/huangjiayu/Mdocagent-dataset)')

    # 解析命令行参数
    args = parser.parse_args()

    dataset_name = args.dataset_name
    path_prefix = args.path_prefix

    json_file_name = "samples.json"
    input_json_file = f"{path_prefix}/{dataset_name}/{json_file_name}"
    output_dir = f"{project_dir}/result/{dataset_name}"

    # 判断目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据
    print(f"正在加载数据从: {input_json_file}")
    data_list = load_json_data(input_json_file)
    print(f"成功加载 {len(data_list)} 条数据")

    # 处理数据
    print("开始处理数据...")
    # note: FetaTab 和 PaperTab 没有金标签，不能测试
    result_list = process_data(data_list, dataset_name, output_dir)
    print(f"数据处理完成，共处理 {len(result_list)} 条结果")

    print("程序执行完成！")