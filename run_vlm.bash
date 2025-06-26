#!/bin/bash

# 初始化conda
modelpath=/home/huangjiayu/ViRAGExample/LLM/Qwen2.5-VL-7B-Instruct

# 检查是否已经在运行
if [ -f "pid_vllm.log" ]; then
    pid=$(cat pid_vllm.log)
    if ps -p $pid > /dev/null 2>&1; then
        echo "vllm服务已经在运行，PID: $pid"
        echo "如需重启，请先运行: kill $pid"
        exit 1
    else
        echo "发现旧的PID文件，但进程不存在，将重新启动"
        rm -f pid_vllm.log
    fi
fi

echo "启动vllm服务..."

# 在后台运行vllm，将输出重定向到日志文件，保存PID
nohup vllm serve $modelpath --port 8001 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 2 > run_vlm.log 2>&1 &

# 保存PID到文件
echo $! > pid_vllm.log

echo "vllm服务已启动，PID: $(cat pid_vllm.log)"
echo "日志文件: run_vlm.log"
echo "PID文件: pid_vllm.log"
echo ""
echo "查看日志: tail -f run_vlm.log"
echo "停止服务: kill \$(cat pid_vllm.log)"
