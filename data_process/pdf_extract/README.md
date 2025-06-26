# 执行说明

执行extract_path.py，生成converted_paths.txt文件，将sample.json数据集中的引用的文档PDF，转换为本地PDF路径，并保存到converted_paths.txt文件中

执行extract_pdf.py，读取converted_paths.txt文件，将数据集中的pdf文件复制到data_process/data/test/pdf目录下，方便后面处理

都是在项目根目录运行