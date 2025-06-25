import os
import argparse
from tqdm import tqdm
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 获取当前工作路径
project_path = os.getcwd()
base_dir_prefix = f"{project_path}/data_process/data"

def process_pdf(pdf_path, dataset_name):
    """
    处理单个PDF文件，将其转换为图片

    参数:
        pdf_path: PDF文件的路径
        dataset_name: 数据集名称

    返回:
        output_dir: 处理后的输出目录路径
    """
    # 构建基础目录路径
    base_dir = os.path.join(base_dir_prefix, dataset_name)

    # 使用PDF文件名作为文件名（不含扩展名）
    pdf_file_name = os.path.basename(pdf_path)
    name_without_suff = os.path.splitext(pdf_file_name)[0]

    # 构建输出目录结构 - 为每个PDF文件创建子目录
    output_dir = os.path.join(base_dir, "img", name_without_suff)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 将PDF转换为图片列表
        images = convert_from_path(pdf_path)

        # 判断如何目录下jpg文件数量和images长度一致，则跳过
        if len(os.listdir(output_dir)) == len(images):
            print(f"文件 {pdf_file_name} 已存在，跳过")
            return output_dir

        # 将每一页保存为单独的图片
        for i, image in enumerate(images):
            idx = i + 1  # 页码从1开始
            # 保存为JPEG格式，文件名格式为：页码.jpg
            # 判断如果文件存在，则跳过
            output_path = os.path.join(output_dir, f'{idx}.jpg')
            image.save(output_path, 'JPEG')

        return output_dir
    except Exception as e:
        print(f"处理文件 {pdf_file_name} 时出错: {str(e)}")
        return None

def process_dataset(dataset_name, workers=1):
    """
    处理指定数据集目录下的所有PDF文件

    参数:
        dataset_name: 数据集名称，对应data目录下的子目录名
        workers: 并行处理的工作线程数，默认为1
    """
    # 检查目录结构是否正常
    base_dir = os.path.join(base_dir_prefix, dataset_name)
    required_dirs = [
        os.path.join(base_dir, "pdf"),
        os.path.join(base_dir, "img"),
        os.path.join(base_dir, "md"),
        # os.path.join(base_dir, "bge_ingestion"),
        os.path.join(base_dir, "colqwen_ingestion")
    ]

    # 检查并创建所有必需的目录
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"创建目录: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    # 构建PDF文件目录路径
    pdf_dir = os.path.join(base_dir, "pdf")

    # 获取目录中所有的PDF文件
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"警告: 在目录 {pdf_dir} 中没有找到PDF文件")
        return

    # 准备处理任务
    tasks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        tasks.append((pdf_path, dataset_name))

    # 根据工作线程数选择处理模式
    if workers == 1:
        # 单线程顺序处理
        for pdf_path, dataset_name in tqdm(tasks, desc="正在转换PDF为图片"):
            try:
                output_dir = process_pdf(pdf_path, dataset_name)
                if output_dir:
                    print(f"成功处理文件 {os.path.basename(pdf_path)} -> 输出目录: {output_dir}")
            except Exception as e:
                print(f"处理文件 {os.path.basename(pdf_path)} 时出错: {str(e)}")
    else:
        # 多线程并行处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_pdf, pdf_path, dataset_name): (pdf_path, dataset_name)
                for pdf_path, dataset_name in tasks
            }
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="正在并行转换PDF为图片"):
                pdf_path, dataset_name = future_to_file[future]
                try:
                    output_dir = future.result()
                    if output_dir:
                        print(f"成功处理文件 {os.path.basename(pdf_path)} -> 输出目录: {output_dir}")
                except Exception as e:
                    print(f"处理文件 {os.path.basename(pdf_path)} 时出错: {str(e)}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='PDF转图片工具')
    parser.add_argument('--dataset_name', type=str, default='LongDocURL',
                       help='数据集名称 (默认: LongDocURL)')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行处理的工作线程数 (默认: 4)')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    dataset_name = args.dataset_name
    workers = args.workers

    print(f"使用数据集: {dataset_name}")
    print(f"工作线程数: {workers}")

    process_dataset(dataset_name, workers=workers)

