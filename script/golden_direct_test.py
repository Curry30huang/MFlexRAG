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
    处理数据集，根据数据集名称调用不同的处理函数.直接调用金标签，测试模型直接输出答案的能力
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = f"{output_dir}/golden_direct_test.jsonl"

    # 读取输出文件，获取所有已经回答过的问题
    processed_questions = set()
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 过滤掉空行
            non_empty_lines = [line.strip() for line in lines if line.strip()]

            if non_empty_lines:  # 确保文件不为空且有有效内容
                try:
                    for line in non_empty_lines:
                        item = json.loads(line)
                        doc_id = item.get('doc_id', '')
                        question = item.get('question', '')
                        # 使用doc_id+question作为唯一标识
                        question_key = f"{doc_id}_{question}"
                        processed_questions.add(question_key)
                    print(f"从输出文件中读取到 {len(processed_questions)} 个已处理的问题")
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"解析输出文件时出错: {e}，从头开始处理")
                    processed_questions = set()
            else:
                print("输出文件为空，从头开始处理")
                processed_questions = set()
    else:
        print("输出文件不存在，从头开始处理")
        processed_questions = set()

    # 过滤data_list，剔除已处理的问题
    filtered_data_list = []
    for data in data_list:
        doc_id = data.get('doc_id', '')
        question = data.get('question', '')
        question_key = f"{doc_id}_{question}"
        if question_key not in processed_questions:
            filtered_data_list.append(data)

    print(f"原始数据 {len(data_list)} 个问题，已处理 {len(processed_questions)} 个，剩余 {len(filtered_data_list)} 个待处理")

    # 使用过滤后的数据
    data_list = filtered_data_list

    result_list = []

    for data in data_list:
        question = data['question']
        answer = data['answer']
        evidence_pages = data['evidence_pages']
        doc_id = data['doc_id']
        # doc_id 是文件名，需要去掉后缀，使用os.path.splitext完成严谨的后缀处理
        doc_no = os.path.splitext(doc_id)[0]
        # 判断如果 evidence_pages 是字符串，就需要转换为列表
        if isinstance(evidence_pages, str):
            evidence_pages = json.loads(evidence_pages)
            # 需要判断因为 MMLongBench 的 evidence_pages 下标是中1开始的，但是其他数据集是0开始的，而我们图片向量化时也是从0开始的，所以需要减1
            # note: 如果evidence_pages为空或者含有0，则直接跳过 ，因为非0都是从1开始的页面，还有部分没有证据的，所以需要跳过
            if dataset_name == 'MMLongBench':
                if evidence_pages is None or len(evidence_pages) == 0 or 0 in evidence_pages:
                    print(f"doc_id: {doc_id}  question: {question}  evidence_pages为空，跳过")
                    continue
                evidence_pages = [page_no - 1 for page_no in evidence_pages]

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
            "question": question,
            "doc_id": doc_id,
            "answer": answer,
            "predict_answer": predict_answer
        }

        # 立即保存单个结果
        success = save_result(result_obj, output_dir)
        if success:
            result_list.append(result_obj)
            print(f"doc_id: {doc_id}  question: {question} 已经完成并保存")
        else:
            print(f"doc_id: {doc_id}  question: {question} 保存失败，跳过")

    print(f"所有问题已经完成")
    return result_list

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
    # nohup python -u -m script.golden_direct_test > golden_direct_test.log 2>&1 &

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