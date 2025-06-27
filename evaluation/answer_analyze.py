import json
import os
import argparse

project_dir = os.getcwd()

def load_results(results_file):
    with open(results_file, 'r') as f:
        lines = f.readlines()
        # 去掉空行
        lines = [line for line in lines if line.strip()]
        results = []
        for i, line in enumerate(lines, 1):
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误，第{i}行: {line.strip()}")
                print(f"错误信息: {e}")
                continue
    return results


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分析答案")

    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="LongDocURL",
                       help="数据集名称 (默认: LongDocURL)")
    parser.add_argument("--path_prefix", type=str,
                       default="/home/huangjiayu/Mdocagent-dataset",
                       help="数据集根目录路径 (默认: /home/huangjiayu/Mdocagent-dataset)")
    parser.add_argument("--results_file_name", type=str, default='evaluation_direct_results.jsonl',
                       help="要评估的结果文件路径（如果不指定，将根据dataset_name自动生成）")

    return parser.parse_args()



def main():
    args = parse_arguments()
    input_file = os.path.join(project_dir,  "result", args.dataset_name ,args.results_file_name)
    results = load_results(input_file)

    scores = [item['score'] for item in results if 'score' in item]

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nEvaluation complete. Average score: {average_score * 100 :.2f} %")

if __name__ == "__main__":
    main()