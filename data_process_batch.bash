#!/bin/bash

# 检查是否提供了数据集参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <dataset_name>"
    echo "示例: $0 LongDocURL"
    echo "示例: $0 test"
    exit 1
fi

# 获取数据集名称参数
DATASET_NAME=$1

echo "开始处理数据集: $DATASET_NAME"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 定义日志文件路径
LOG_FILE="data_process_${DATASET_NAME}.log"
PID_FILE="pid_${DATASET_NAME}.log"

# 清除旧日志文件
> "$LOG_FILE"
> "$PID_FILE"

# 记录脚本开始时间
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 数据处理脚本开始执行" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 数据集名称: $DATASET_NAME" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 工作目录: $(pwd)" | tee -a "$LOG_FILE"

# 创建内联脚本内容
SCRIPT_CONTENT="
# 设置工作目录
cd '$SCRIPT_DIR'

# 记录开始处理
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 开始后台处理任务\" >> '$LOG_FILE'

# 检查Python环境
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Python路径: \$(which python)\" >> '$LOG_FILE'
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Python版本: \$(python --version)\" >> '$LOG_FILE'

# 任务1: PDF转图像
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 开始执行任务: PDF转图像\" | tee -a '$LOG_FILE'
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 执行命令: python -m data_process.pdf2img --dataset_name $DATASET_NAME\" | tee -a '$LOG_FILE'
python -m data_process.pdf2img --dataset_name $DATASET_NAME >> '$LOG_FILE' 2>&1
if [ \$? -ne 0 ]; then
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] PDF转图像任务失败，停止后续任务\" >> '$LOG_FILE'
    exit 1
fi
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 任务完成: PDF转图像\" | tee -a '$LOG_FILE'

# 任务2: PDF转md
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 开始执行任务: PDF转md\" | tee -a '$LOG_FILE'
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 执行命令: python -m data_process.pdf2md --dataset_name $DATASET_NAME\" | tee -a '$LOG_FILE'
python -m data_process.pdf2md --dataset_name $DATASET_NAME >> '$LOG_FILE' 2>&1
if [ \$? -ne 0 ]; then
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] PDF转md任务失败，停止后续任务\" >> '$LOG_FILE'
    exit 1
fi
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 任务完成: PDF转md\" | tee -a '$LOG_FILE'

# 任务3: 数据向量化
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 开始执行任务: 数据向量化\" | tee -a '$LOG_FILE'
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 执行命令: python -m data_process.data2embedding --dataset_name $DATASET_NAME\" | tee -a '$LOG_FILE'
python -m data_process.data2embedding --dataset_name $DATASET_NAME >> '$LOG_FILE' 2>&1
if [ \$? -ne 0 ]; then
    echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 数据向量化任务失败\" >> '$LOG_FILE'
    exit 1
fi
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 任务完成: 数据向量化\" | tee -a '$LOG_FILE'

echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] 所有任务已完成\" >> '$LOG_FILE'
"

# 在后台运行脚本内容
echo "$SCRIPT_CONTENT" | nohup bash > /dev/null 2>&1 &
MAIN_PID=$!

# 记录主脚本的PID
echo "主脚本PID: $MAIN_PID" >> "$PID_FILE"

echo "数据处理任务已在后台启动"
echo "正在处理数据集: $DATASET_NAME"
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"
echo "主进程PID: $MAIN_PID"

# 显示如何监控进度
echo ""
echo "监控命令:"
echo "  tail -f $LOG_FILE"
echo "  ps aux | grep $MAIN_PID"
echo "  kill $MAIN_PID  # 停止任务"