import os
import shutil
import argparse
from pathlib import Path

# 获取当前工作路径
project_path = os.getcwd()

def clear_directories(dataset_name):
    # 需要清空的目录列表
    directories = [
        f"{project_path}/data_process/data/{dataset_name}/pdf",
        f"{project_path}/data_process/data/{dataset_name}/md",
        f"{project_path}/data_process/data/{dataset_name}/img",
        f"{project_path}/data_process/data/{dataset_name}/colqwen_ingestion",
    ]

    for directory in directories:
        if os.path.exists(directory):
            # 删除目录中的所有内容
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    print(f"已删除: {item_path}")
                except Exception as e:
                    print(f"删除失败 {item_path}: {str(e)}")
        else:
            # 如果目录不存在，创建它
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")

def copy_pdf_files(dataset_name):
    # 目标目录
    target_dir = f"{project_path}/data_process/data/{dataset_name}/pdf"

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 读取converted_paths.txt文件
    with open(f"{project_path}/data_process/pdf_extract/{dataset_name}_converted_paths.txt", "r") as f:
        for line in f:
            # 处理每一行
            pdf_path = line.strip()
            if not pdf_path:
                continue

            # 获取文件名
            pdf_filename = os.path.basename(pdf_path)

            # 构建目标路径
            target_path = os.path.join(target_dir, pdf_filename)

            try:
                # 复制文件
                shutil.copy2(pdf_path, target_path)
                print(f"成功复制: {pdf_filename}")
            except Exception as e:
                print(f"复制失败 {pdf_filename}: {str(e)}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='PDF文件提取和处理工具')
    parser.add_argument('--dataset_name', type=str, default="LongDocURL",
                       help='数据集名称 (默认: LongDocURL)')

    # 解析命令行参数
    args = parser.parse_args()
    dataset_name = args.dataset_name

    print(f"使用数据集: {dataset_name}")

    # 首先清空所有目录
    print("开始清空目录...")
    clear_directories(dataset_name)
    print("目录清空完成")

    # 然后复制文件
    print("开始复制PDF文件...")
    copy_pdf_files(dataset_name)
    print("PDF文件复制完成")
