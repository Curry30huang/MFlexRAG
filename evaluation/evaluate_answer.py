"""
文档问答响应评估脚本

这个脚本用于评估文档问答系统中模型生成的答案质量。
主要功能包括：
1. 使用GPT-4.1作为评估器，对比预测答案与真实答案
2. 支持批量处理多个问答对
3. 提供详细的评估指标和统计分析
4. 支持按证据来源和页面数量进行子集分析
5. 支持多种数据集格式（LongDocURL、MMLongBench等）

使用方法：
python evaluate_answer.py --dataset_name LongDocURL --path_prefix /path/to/dataset

该脚本测试，只针对 LongDocURL 数据集，只进行临时测试使用
"""

import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import joblib
from functools import partial
import ast
import numpy as np
from dotenv import load_dotenv

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

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用GPT-4.1评估文档问答响应")

    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="LongDocURL",
                       help="数据集名称 (默认: LongDocURL)")
    parser.add_argument("--path_prefix", type=str,
                       default="/home/huangjiayu/Mdocagent-dataset",
                       help="数据集根目录路径 (默认: /home/huangjiayu/Mdocagent-dataset)")
    parser.add_argument("--results_file_name", type=str, default='golden_direct_test.jsonl',
                       help="要评估的结果文件路径（如果不指定，将根据dataset_name自动生成）")

    # 模型配置参数
    parser.add_argument("--model", type=str, default="openai/gpt-4.1",
                       help="用于评估的模型名称 (默认: openai/gpt-4.1)")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                       help="OpenAI API的基URL (默认: https://openrouter.ai/api/v1)")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="GPT响应的最大token数")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="GPT响应的温度(较低值用于更一致的评估)")
    parser.add_argument("--add_notanswerable", action="store_true",
                        help="向真实答案添加'无法回答'")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="并行运行的作业数量(-1表示使用所有核心)")

    return parser.parse_args()


def get_file_paths(dataset_name, path_prefix, results_file_name):
    """根据数据集名称自动生成文件路径

    Args:
        dataset_name (str): 数据集名称
        path_prefix (str): 数据集根目录路径
        results_file_name (str): 结果文件名称
    Returns:
        tuple: (results_file_path, ground_truth_file_path, output_file_path)
    """

    results_file = f"{project_dir}/result/{dataset_name}/{results_file_name}"

    # 真实数据文件路径
    ground_truth_file = f"{path_prefix}/{dataset_name}/samples.json"

    output_file = f"{project_dir}/result/{dataset_name}/evaluation_results.jsonl"

    return results_file, ground_truth_file, output_file

def load_ground_truth(ground_truth_file):
    """加载真实数据并创建映射字典

    Args:
        ground_truth_file (str): 真实数据JSON文件路径

    Returns:
        dict: 以 'doc_id_question' 为键，答案为值的映射字典
    """
    print("load_ground_truth")
    ground_truth_list = []
    # 读取 json 列表文件
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        ground_truth_list = json.load(f)
    return ground_truth_list

def load_results(results_file):
    """从JSONL文件加载模型预测生成的结果

    Args:
        results_file (str): 结果文件路径

    Returns:
        list: 包含模型生成结果的列表
    """
    print("Loading results")
    results_list = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            results_list.append(item)
    return results_list

def load_combined_data(results_list:list, ground_truth_list:list):
    """加载模型预测生成的结果和真实数据

    Args:
        results_list (list): 模型预测生成的结果
        ground_truth_list (list): 真实数据
    """
    # 检查 results_list 和 ground_truth_list 的结构是否相同，判断前3个元素的question_id是否相同
    if results_list[0]['question_id'] != ground_truth_list[0]['question_id']:
        raise ValueError("results_list 和 ground_truth_list 的结构不同")
    if results_list[1]['question_id'] != ground_truth_list[1]['question_id']:
        raise ValueError("results_list 和 ground_truth_list 的结构不同")
    if results_list[2]['question_id'] != ground_truth_list[2]['question_id']:
        raise ValueError("results_list 和 ground_truth_list 的结构不同")

    combined_data = []
    # results_list 和 ground_truth_list 的结构相同，且顺序相同，可以直接组合
    for idx, result in enumerate(results_list):
        tmp = result.copy()
        tmp['ground_truth'] = ground_truth_list[idx]['answer']
        tmp['question'] = ground_truth_list[idx]['question']
        tmp['doc_id'] = ground_truth_list[idx]['doc_id']
        combined_data.append(tmp)
    return combined_data

def evaluate_response(client:OpenAI, model:str, predicted_answer:str, ground_truth:str, question:str, max_tokens:int=1024, temperature:float=0.0):
    """
    使用GPT-4.1评估预测答案与真实答案的对比

    Args:
        client: OpenAI客户端实例
        model (str): 用于评估的模型名称
        predicted_answer (str): 模型预测的答案
        ground_truth (str): 真实答案
        question (str): 问题内容
        max_tokens (int): 最大token数
        temperature (float): 温度参数

    Returns:
        dict: 包含评分和解释的字典，格式如 {"score": 1, "explanation": "..."}
    """
    try:
        # 构建评估提示词
        prompt = f"""Question: {question}
Predicted Answer: {predicted_answer}
Ground Truth Answer: {ground_truth}

Please evaluate if the predicted answer is correct compared to the ground truth.
Score the answer on:
Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

Return only a string with these scores in a dictionary and can be parsed by json.loads, e.g. {{"binary_correctness": 1}}"""

        # 调用GPT模型进行评估
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )

        evaluation_text = response.choices[0].message.content
        if evaluation_text is None:
            raise ValueError(f"预测答案为空")
        evaluation_text = evaluation_text.strip()

        try:
            # 解析GPT返回的JSON格式评估结果
            evaluation_dict = json.loads(evaluation_text)
            score = evaluation_dict.get("binary_correctness", 0)
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回-1分
            score = -1,
        return {
            "score": score,
            "explanation": evaluation_text
        }
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {"score": 0, "explanation": f"Evaluation error: {str(e)}"}


def process_item(item:dict, client:OpenAI, model:str, max_tokens:int, temperature:float, add_notanswerable:bool,output_file:str):
    """
    处理单个评估项目

    Args:
        item (dict): 包含单个问答项目的字典
        client: OpenAI客户端实例
        model (str): 模型名称
        max_tokens (int): 最大token数
        temperature (float): 温度参数
        add_notanswerable (bool): 是否包含无法回答的问题

    Returns:
        dict: 评估结果字典，如果跳过则返回None
    """
    try:
        question = item['question']
        doc_id = item['doc_id']
        ground_truth = item['ground_truth']
        predicted_answer = item['predict_answer']

        # 跳过"无法回答"的真实答案（除非明确要求包含）
        if ground_truth == "Not answerable" and not add_notanswerable:
            return None

        # 评估响应
        eval_result = evaluate_response(
            client,
            model,
            predicted_answer,
            ground_truth,
            question,
            max_tokens,
            temperature
        )

        res = {
            "score": eval_result['score'],
            "doc_id": doc_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "explanation": eval_result['explanation'],
            "error": None
        }

        # 将结果写入文件
        with open(output_file, 'a') as f:
            f.write(json.dumps(res) + '\n')

        # 返回评估结果
        return res
    except Exception as e:
        return {
            "doc_id": item.get('doc_id', item.get('question_id', '')),
            "question": item.get('question', ''),
            "error": f"Evaluation error: {str(e)}",
            "score": 0,
            "explanation": f"Exception during evaluation: {str(e)}"
        }


def main():
    """主函数：运行评估流程"""
    # 加载环境变量
    load_environment_variables()

    args = parse_arguments()

    # 获取文件路径
    results_file, ground_truth_file, output_file = get_file_paths(
        args.dataset_name,
        args.path_prefix,
        args.results_file_name
    )

    # 初始化OpenAI客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=args.base_url)
    if client is None:
        print("Exiting script as OpenAI client could not be initialized.")
        return
    print("In the main function")

    # 加载真实数据
    ground_truth_map = load_ground_truth(ground_truth_file)

    # 加载模型生成的结果（包含真实答案和预测答案）
    results_list = load_results(results_file)

    combined_data = load_combined_data(results_list, ground_truth_map)

    # 读取输出文件，需要获取所有已经回答过的问题，从combined_data中剔除
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
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

    # 过滤combined_data，剔除已处理的问题
    filtered_combined_data = []
    for item in combined_data:
        doc_id = item.get('doc_id', '')
        question = item.get('question', '')
        question_key = f"{doc_id}_{question}"
        if question_key not in processed_questions:
            filtered_combined_data.append(item)

    print(f"原始数据 {len(combined_data)} 个问题，已处理 {len(processed_questions)} 个，剩余 {len(filtered_combined_data)} 个待处理")

    # 使用过滤后的数据
    combined_data = filtered_combined_data

    # 创建带有固定参数的偏函数
    process_func = partial(
        process_item,
        client=client,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        add_notanswerable=args.add_notanswerable,
        output_file=output_file
    )

    # 使用joblib并行处理项目
    print(f"Processing {len(combined_data)} items with {args.n_jobs} parallel jobs")
    evaluation_results = joblib.Parallel(n_jobs=args.n_jobs, backend="threading")(
        joblib.delayed(process_func)(item) for item in tqdm(combined_data, desc="Evaluating responses")
    )


if __name__ == "__main__":
    main()