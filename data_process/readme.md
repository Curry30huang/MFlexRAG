# 数据处理模块
该模块是数据处理模块，用于处理各种类型的数据集，同时将数据处理之后向量化入库

文件说明：
- pdf2img.py：将PDF文件转换为图片，以一个文件为单位，将图片保存这个文件的目录下面
- pdf2md.py：将PDF文件转换为Markdown文件，里面与板式分析得到的图像和文本跨度信息，也是以文档为单位，将markdown文件保存到这个文件的目录下面
    - magic-pdf.json：配置文件，用于配置OCR模型，需要将这个文件移动到 ~/ 目录下面才可以
- vl_embedding.py：将利用多模态模型将数据转换为向量，支持文本和图像，因为后面需要query转换
- data2embedding.py：将数据处理之后向量化入库，入口文件，以文档为单位创建子目录，方便封闭领域问答区域指定的向量数据
- download_models_hf.py：下载MinerU模型文件

处理数据集合输入输出目录结构：
- 放在data目录下面，data下一层子目录是根据数据集名称划分的，每个数据集目录下面有以下子目录：
- 具体数据集目录结构，直接采用扁平式结构，所有相关文件都放在下面具体的功能目录下面:
    - colqwen_ingestion：多模态数据向量入库目录
    - img：pdf转换的图片数据目录
    - md：markdown数据相关目录
    - pdf：pdf原数据目录

依赖包问题：
- 再使用了 vidorag和mineru两个环境依赖之后，会有一些依赖冲突问题，目前以vidorag为基础
- pip install --upgrade "accelerate>=0.21.0" "peft>=0.4.0"
- pip uninstall bitsandbytes
- pip install langchain langchain-community

这个目录下的代码都需要进入这个目录下面直接执行，不能采用模块方式执行，否则会因为依赖包问题导致报错。

## MinerU 字段说明

输出的`_content_list.json`文件是关键内容，是一个JSON列表，每个元素可能包含的字段含义如下：
- type：文本类型，可能的值为`text`、`image`、`table`、`equation`
- text：文本内容,存在空字符串
- text_level：文本层级，用于标题等层级结构
- page_idx：页码索引，表示内容所在的页码
- img_path：图片路径，当type为image或table时存在
- img_caption：图片说明文字列表，当type为image时存在
- img_footnote：图片脚注列表，当type为image时存在，很多时候是空
- table_caption：表格说明文字列表，当type为table时存在
- table_footnote：表格脚注列表，当type为table时存在,很多时候是空
- table_body：表格内容，当type为table时存在，以HTML格式存储
- text_format：文本格式，当type为equation时存在，表示公式格式（如latex）

## 执行说明

都是需要在项目根目录以模块方式运行，否则会因为依赖包问题导致报错。