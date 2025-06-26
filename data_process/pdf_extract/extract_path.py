import json
import os
import argparse

# 获取当前工作路径
project_path = os.getcwd()

def convert_path(doc_id: str, path_prefix: str, dataset_name: str):
    # 拼接路径，得到doc_id的绝对路径
    return f"{path_prefix}/{dataset_name}/documents/{doc_id}"

def process_json_file(input_file, path_prefix, dataset_name):
    doc_id_list = set()  # 使用集合来存储唯一的路径,含有.pdf的文件名字

    # 读取json文件
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)  # 直接加载JSON列表
            for data in data_list:
                if 'doc_id' in data:
                    doc_id_list.add(data['doc_id'])
        except json.JSONDecodeError:
            print(f"无法解析JSON文件: {input_file}")
            return set()

    # 转换路径并去重
    converted_paths = {convert_path(doc_id, path_prefix, dataset_name) for doc_id in doc_id_list}

    return converted_paths

def main():
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
    input_file = f"{path_prefix}/{dataset_name}/{json_file_name}"
    output_file = f"{project_path}/data_process/pdf_extract/{dataset_name}_converted_paths.txt"  # 输出文件路径

    # 处理json文件
    converted_paths = process_json_file(input_file, path_prefix, dataset_name)

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in sorted(converted_paths):
            f.write(f'{path}\n')

    print(f"处理完成！共处理了 {len(converted_paths)} 个唯一路径。")
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
