#!/bin/bash


# conda create -n mflexrag python=3.10
# conda activate mflexrag

pip install -U "magic-pdf[full]" -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements_dataprocess.txt