# 依赖说明

vllm 对 torch 和 cuda 版本有严格要求，为了防止与agent代码依赖冲突，所以使用两个虚拟环境隔离开。分别运行

# 模块目录说明

- agent: 单agent模块
- data_process: 数据处理模块
- pipline: 流程控制模块
- prompt: 提示词模块
- utils: 工具模块

# 数据处理说明

1. 执行特定数据集的抽取迁移工作
    1. `python -m data_process.LongDocURL-extract.extract_path`
    2. `python -m data_process.LongDocURL-extract.extract_pdf`
2. 数据向量化
    1. PDF转图像 `python -m data_process.pdf2img`
    2. PDF转md `python -m data_process.pdf2md`
    3. 数据向量化 `python -m data_process.data2embedding`
3. TODO: 批量执行测试
