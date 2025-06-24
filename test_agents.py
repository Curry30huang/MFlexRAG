#!/usr/bin/env python3
"""
测试三个agent的解析功能
"""

import json
from agent.image_resort import ImageResortAgent
from agent.document_summary import DocumentSummaryAgent
from agent.reasoner import ReasonerAgent

def test_image_resort_parsing():
    """测试图像重排序解析功能"""
    print("=== 测试图像重排序解析 ===")

    # 模拟大模型返回的JSON响应
    test_responses = [
        # 正常JSON格式
        '''```json
{
    "image_analysis": "Image 0 shows a data pipeline diagram. Image 1 contains a table with metrics. Both are relevant to the query about data processing.",
    "selected_images": [0, 1],
    "image_roles": "Image 0 provides the overall process flow, while Image 1 shows specific performance metrics."
}
```''',

        # 直接JSON格式
        '''{
    "image_analysis": "Image 0 shows a data pipeline diagram. Image 1 contains a table with metrics. Both are relevant to the query about data processing.",
    "selected_images": [0, 1],
    "image_roles": "Image 0 provides the overall process flow, while Image 1 shows specific performance metrics."
}''',

        # 字符串格式的selected_images
        '''```json
{
    "image_analysis": "Image 0 shows a data pipeline diagram. Image 1 contains a table with metrics. Both are relevant to the query about data processing.",
    "selected_images": "[0, 1]",
    "image_roles": "Image 0 provides the overall process flow, while Image 1 shows specific performance metrics."
}
```'''
    ]

    agent = ImageResortAgent("dummy_key", "dummy_model", "dummy_url")

    for i, response in enumerate(test_responses):
        print(f"\n测试用例 {i+1}:")
        result = agent._parse_resort_response(response)
        print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

def test_document_summary_parsing():
    """测试文档摘要解析功能"""
    print("\n=== 测试文档摘要解析 ===")

    # 模拟大模型返回的JSON响应
    test_responses = [
        # 正常JSON格式
        '''```json
{
    "document_summary": "This document provides a comprehensive overview of the data construction pipeline for multimodal retrieval systems. The strategic approach involves iterative refinement where each stage builds upon the previous one to ensure data quality."
}
```''',

        # 直接JSON格式
        '''{
    "document_summary": "This document provides a comprehensive overview of the data construction pipeline for multimodal retrieval systems. The strategic approach involves iterative refinement where each stage builds upon the previous one to ensure data quality."
}'''
    ]

    agent = DocumentSummaryAgent("dummy_key", "dummy_model", "dummy_url")

    for i, response in enumerate(test_responses):
        print(f"\n测试用例 {i+1}:")
        result = agent._parse_summary_response(response)
        print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

def test_reasoner_parsing():
    """测试推理解析功能"""
    print("\n=== 测试推理解析 ===")

    # 模拟大模型返回的JSON响应
    test_responses = [
        # answer类型
        '''```json
{
    "scratchpad": "Analyzing the provided images and document summary. Image 1 shows the data construction pipeline overview with stages including document collection, query creation, quality review, and multimodal refine.",
    "response_type": "answer",
    "answer": "The data construction pipeline consists of four main stages: 1) Document collecting, 2) Query creation, 3) Quality review, and 4) Multimodal refine."
}
```''',

        # query_update类型
        '''```json
{
    "scratchpad": "The current images show the data construction pipeline but do not contain specific information about the evaluation metrics.",
    "response_type": "query_update",
    "query_update": "evaluation metrics performance benchmarks quality assessment criteria data construction pipeline results",
    "notes": "Current pages show the pipeline structure but lack specific evaluation details. Need to find sections about metrics, benchmarks, or performance assessment."
}
```''',

        # not_answerable类型
        '''```json
{
    "scratchpad": "Analyzing the provided images and document summary. The document does not contain information about the specific topic requested.",
    "response_type": "not_answerable",
    "not_answerable": "The document does not contain the information needed to answer this question."
}
```'''
    ]

    agent = ReasonerAgent("dummy_key", "dummy_model", "dummy_url")

    for i, response in enumerate(test_responses):
        print(f"\n测试用例 {i+1}:")
        result = agent._parse_reasoning_response(response)
        print(f"解析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    test_image_resort_parsing()
    test_document_summary_parsing()
    test_reasoner_parsing()
    print("\n=== 所有测试完成 ===")