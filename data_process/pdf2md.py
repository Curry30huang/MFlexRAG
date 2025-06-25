import os
import uuid
import argparse
from pathlib import Path

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# 获取当前工作路径
project_path = os.getcwd()
base_dir_prefix = f"{project_path}/data_process/data"

def process_pdf(pdf_path, dataset_name):
    """
    处理单个PDF文件，将其转换为Markdown格式

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

    # 构建以文档为单位的目录结构 - 在md目录下创建子目录
    local_md_dir = os.path.join(base_dir, "md", name_without_suff)  # 每个PDF文档在md目录下有一个子目录
    local_image_dir = os.path.join(local_md_dir, "img")  # 图片存储目录
    image_dir = "img"  # 相对路径，指向md子目录下的img文件夹

    # 确保输出目录存在
    os.makedirs(local_md_dir, exist_ok=True)
    os.makedirs(local_image_dir, exist_ok=True)

    # 初始化文件写入器
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

    # 读取PDF文件内容
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_path)

    # 处理PDF文件
    ds = PymuDocDataset(pdf_bytes)

    # 根据PDF类型选择处理方式（OCR或文本模式）
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # OCR模式处理
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        # 文本模式处理
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # 生成可视化结果
    # 1. 绘制模型结果
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
    # 2. 绘制布局结果
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
    # 3. 绘制文本跨度结果
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    # 输出处理结果
    # 1. 导出Markdown文件
    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
    # 2. 主要核心：导出内容列表（JSON格式）
    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    return local_md_dir

def process_dataset(dataset_name):
    """
    处理指定数据集目录下的所有PDF文件

    参数:
        dataset_name: 数据集名称，对应data目录下的子目录名
    """
    # 检查目录结构是否正常
    base_dir = os.path.join(base_dir_prefix, dataset_name)
    required_dirs = [
        os.path.join(base_dir, "pdf"),
        os.path.join(base_dir, "md"),
        os.path.join(base_dir, "img"),
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

    # 逐个处理PDF文件
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            # 处理单个PDF文件
            output_dir = process_pdf(pdf_path, dataset_name)
            print(f"成功处理文件 {pdf_file} -> 输出目录: {output_dir}")
        except Exception as e:
            print(f"处理文件 {pdf_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='PDF转Markdown工具')
    parser.add_argument('--dataset_name', type=str, default='LongDocURL',
                       help='数据集名称 (默认: LongDocURL)')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    dataset_name = args.dataset_name

    print(f"使用数据集: {dataset_name}")

    process_dataset(dataset_name)