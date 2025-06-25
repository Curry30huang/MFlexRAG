import json
import os

# 获取当前工作路径
project_path = os.getcwd()
# 获取LongDocURL文件路径
longdocurl_path_prefix = f"/home/huangjiayu/long-agent/LongDocURL/data-longdoc"

def convert_path(original_path):
    # 从原始路径中提取关键部分
    parts = original_path.split('/')
    # 获取文件名和目录编号
    filename = parts[-1]
    dir_num = filename[:4]  # 获取前4位数字作为目录名

    # 构建新的路径
    new_path = f"{longdocurl_path_prefix}/mnt/achao/Downloads/ccpdf_zip/4000-4999/{dir_num}/{filename}"
    return new_path

def process_jsonl_file(input_file):
    pdf_paths = set()  # 使用集合来存储唯一的路径

    # 读取jsonl文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'pdf_path' in data:
                    pdf_paths.add(data['pdf_path'])
            except json.JSONDecodeError:
                print(f"无法解析JSON行: {line}")
                continue

    # 转换路径并去重
    converted_paths = {convert_path(path) for path in pdf_paths}

    return converted_paths

def main():
    json_file_name = "LongDocURL_test.jsonl"
    input_file = f"{project_path}/data_process/LongDocURL-extract/{json_file_name}"  # 请替换为你的输入文件路径
    output_file = f"{project_path}/data_process/LongDocURL-extract/converted_paths.txt"  # 输出文件路径

    # 处理文件
    converted_paths = process_jsonl_file(input_file)

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in sorted(converted_paths):
            f.write(f'{path}\n')

    print(f"处理完成！共处理了 {len(converted_paths)} 个唯一路径。")
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
