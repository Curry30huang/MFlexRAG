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

TODO: 这个文件现在有问题，问题在于之前自己写的 golden_direct_test.py 中，对于 LongDocURL 数据集，在输出结果的时候，没有输出 doc_id，导致现在无法正确加载真实数据。应该统一标签，将结果中统一记录 doc_id, question, answer , predict_answer 等字段。
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
    parser.add_argument("--model", type=str, default="gpt-4.1",
                       help="用于评估的模型名称 (默认: gpt-4.1)")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",
                       help="OpenAI API的基URL (默认: https://api.openai.com/v1)")
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
        ground_truth_file (str): 真实数据文件路径

    Returns:
        dict: 以 'doc_id_question' 为键，答案为值的映射字典
    """
    print("load_ground_truth")
    # 使用pandas加载真实数据
    ground_truth_df = pd.read_json(ground_truth_file)

    # 注释掉的行：过滤掉"Not answerable"的答案
    # ground_truth_df = ground_truth_df[ground_truth_df.apply(lambda row: row['answer'] != 'Not answerable', axis=1)].reset_index(drop=True)

    # 创建doc_id+question到答案的映射
    ground_truth_map = {}
    for _, row in ground_truth_df.iterrows():
        key = f"{row['doc_id']}_{row['question']}"
        ground_truth_map[key] = row.get('answer', '')

    return ground_truth_map

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
        print("Using evaluate response")

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


def process_item(item:dict, client:OpenAI, model:str, ground_truth_map:dict, max_tokens:int, temperature:float, add_notanswerable:bool, dataset_name:str):
    """
    处理单个评估项目

    Args:
        item (dict): 包含单个问答项目的字典
        client: OpenAI客户端实例
        model (str): 模型名称
        ground_truth_map (dict): 真实答案映射
        max_tokens (int): 最大token数
        temperature (float): 温度参数
        add_notanswerable (bool): 是否包含无法回答的问题
        dataset_name (str): 数据集名称

    Returns:
        dict: 评估结果字典，如果跳过则返回None
    """
    try:
        key = f"{item['doc_id']}_{item['question']}"
        question = item['question']
        doc_id = item['doc_id']


        ground_truth = ground_truth_map.get(key, '')
        # 跳过"无法回答"的真实答案（除非明确要求包含）
        if ground_truth == "Not answerable" and not add_notanswerable:
            return None

        predicted_answer = item.get('predict_answer', '')

        # 跳过"无法回答"的真实答案（除非明确要求包含）
        if ground_truth == "Not answerable" and not add_notanswerable:
            return None

        # 检查是否有真实答案
        if not ground_truth:
            return {
                "doc_id": doc_id,
                "question": question,
                "error": "No ground truth found",
                "score": 0,
                "explanation": "No ground truth answer available for evaluation"
            }

        # 检查是否有预测答案
        if not predicted_answer:
            return {
                "doc_id": doc_id,
                "question": question,
                "error": "No predicted answer found",
                "score": 0,
                "explanation": "No predicted answer available for evaluation"
            }

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

        # 返回评估结果
        return {
            "score": eval_result['score'],
            "doc_id": doc_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "explanation": eval_result['explanation'],
            "error": None
        }
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

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清空输出文件
    with open(output_file, 'w') as f:
        f.write('')

    # 创建带有固定参数的偏函数
    process_func = partial(
        process_item,
        client=client,
        model=args.model,
        ground_truth_map=ground_truth_map,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        add_notanswerable=args.add_notanswerable,
        dataset_name=args.dataset_name
    )

    # 使用joblib并行处理项目
    print(f"Processing {len(combined_data)} items with {args.n_jobs} parallel jobs")
    evaluation_results = joblib.Parallel(n_jobs=args.n_jobs, backend="threading")(
        joblib.delayed(process_func)(item) for item in tqdm(combined_data, desc="Evaluating responses")
    )

    # 过滤掉None结果（跳过的项目）
    evaluation_results = [result for result in evaluation_results if result is not None]

    # 将结果写入文件
    with open(output_file, 'a') as f:
        for evaluation in evaluation_results:
            f.write(json.dumps(evaluation) + '\n')

    # 计算并打印汇总统计信息
    scores = [item['score'] for item in evaluation_results if 'score' in item]
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nEvaluation complete. Average score: {average_score * 100 :.2f} %")
        print(f"Results saved to: {output_file}")

        # 按证据来源计算子集指标
        samples_df = pd.read_json(ground_truth_file)
        samples = samples_df.to_dict('records')
        # 创建(doc_id, question)到样本的映射，用于后续分析
        if args.dataset_name == "LongDocURL":
            sample_map = {(s.get('question_id', ''), s.get('question', '')): s for s in samples}
        else:  # MMLongBench
            sample_map = {(s['doc_id'], s['question']): s for s in samples}

        # 按证据来源分组统计
        subset_by_source = {}
        for result in evaluation_results:
            if args.dataset_name == "LongDocURL":
                key = (result['doc_id'], result['question'])
            else:  # MMLongBench
                key = (result['doc_id'], result['question'])

            sample = sample_map.get(key, {})
            sources = sample.get('evidence_sources', [])
            # 处理evidence_sources可能是字符串的情况
            if not isinstance(sources, list):
                try:
                    sources = ast.literal_eval(sources)
                except:
                    sources = []
            # 将结果按证据来源分组
            for src in sources:
                subset_by_source.setdefault(src, []).append(result)

        print("\nSubset metrics by evidence source:")
        for src, group in subset_by_source.items():
            scores_list = [item['score'] for item in group if 'score' in item]
            accuracy = float(np.mean(scores_list) * 100) if scores_list else 0.0
            print(f"{src}: samples={len(scores_list)}, accuracy={accuracy:.2f}%")

        # 按证据页面长度分组统计
        subset_by_length = {'no_pages': [], 'single_page': [], 'multiple_pages': []}
        for result in evaluation_results:
            if args.dataset_name == "LongDocURL":
                key = (result['doc_id'], result['question'])
            else:  # MMLongBench
                key = (result['doc_id'], result['question'])

            sample = sample_map.get(key, {})
            pages = sample.get('evidence_pages', [])
            # 处理evidence_pages可能是字符串的情况
            if not isinstance(pages, list):
                try:
                    pages = ast.literal_eval(pages)
                except:
                    pages = []
            l = len(pages)
            # 根据页面数量分类：无页面、单页面、多页面
            if l == 0:
                subset_by_length['no_pages'].append(result)
            elif l == 1:
                subset_by_length['single_page'].append(result)
            else:
                subset_by_length['multiple_pages'].append(result)

        print("\nSubset metrics by evidence pages length:")
        for cat, group in subset_by_length.items():
            scores_list = [item['score'] for item in group if 'score' in item]
            accuracy = float(np.mean(scores_list) * 100) if scores_list else 0.0
            print(f"{cat}: samples={len(scores_list)}, accuracy={accuracy:.2f}%")
    else:
        print("\nEvaluation complete, but no valid scores were calculated.")

if __name__ == "__main__":
    main()