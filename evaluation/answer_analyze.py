import json
import os

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


def main():
    input_file = os.path.join(project_dir,  "result", "LongDocURL" ,"evaluation_results.jsonl")
    results = load_results(input_file)

    scores = [item['score'] for item in results if 'score' in item]

    if scores:
        average_score = sum(scores) / len(scores)
        print(f"\nEvaluation complete. Average score: {average_score * 100 :.2f} %")

if __name__ == "__main__":
    main()