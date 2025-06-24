#!/bin/bash

# 初始化conda

modelpath=/home/huangjiayu/ViRAGExample/LLM/Qwen2.5-VL-7B-Instruct

vllm serve $modelpath --port 8001 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 4
