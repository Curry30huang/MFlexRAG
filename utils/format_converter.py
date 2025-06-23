import json
import os
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from llama_index.core import Document
from typing import Optional, List, Mapping, Any, Dict

def nodes2dict(nodes) -> Dict[str, Any]:
    """
    将节点列表转换为字典格式
    Args:
        nodes: 节点列表
    Returns:
        包含响应、源节点和元数据的字典，也是检索引擎的返回结果
    """
    resp_dict = {
        "response": None, # 响应内容，默认为None
        "source_nodes": [], # 包含检索到的相关节点
        "metadata": None # 元数据，默认为None
    }
    for node in nodes:
        # node.embedding = None
        resp_dict["source_nodes"].append(node.to_dict())
    return resp_dict

def nodefile2node(input_file):
    """
    从JSON文件中读取并转换为节点列表
    Args:
        input_file: 输入文件路径
    Returns:
        节点列表，包含TextNode和ImageNode
    """
    nodes = []
    try:
        with open(input_file, 'r') as f:
            content = json.load(f)

        # 如果content是列表，遍历处理每个节点
        if isinstance(content, list):
            for doc in content:
                if doc['class_name'] == 'TextNode' and doc.get('text', '') != '':
                    nodes.append(TextNode.from_dict(doc))
                elif doc['class_name'] == 'ImageNode':
                    nodes.append(ImageNode.from_dict(doc))
        # 如果content是单个节点
        elif isinstance(content, dict):
            if content['class_name'] == 'TextNode' and content.get('text', '') != '':
                nodes.append(TextNode.from_dict(content))
            elif content['class_name'] == 'ImageNode':
                nodes.append(ImageNode.from_dict(content))

    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return []

    return nodes

def onlchunkfile2node(input_file):
    """
    将在线分块文件转换为节点列表，并建立节点间的关联关系
    Args:
        input_file: 输入文件路径
    Returns:
        带有前后关联关系的节点列表
    """
    content_json = json.load(open(input_file, 'r'))
    nodes = []
    for data in content_json:
        # 创建包含标题和内容的文本节点
        node = TextNode(text=data['title'] + data.get('hier_title', '') + data['content'], file_name=input_file)
        nodes.append(node)
        # 建立节点之间的前后关联关系
        if len(nodes) > 1:
            nodes[-1].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=nodes[-2].node_id
            )
            nodes[-2].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=nodes[-1].node_id
            )
    return nodes

def transform_idp2markdown(response_json: dict) -> str:
    """
    将IDP格式的JSON转换为Markdown格式的文本
    Args:
        response_json: IDP格式的JSON数据
    Returns:
        Markdown格式的文本字符串
    """
    # 初始化Markdown字符串
    markdown_text = ""
    if 'layouts' in response_json:
        response_json = response_json['layouts']

    # 遍历layouts数组，根据不同类型转换为对应的Markdown格式
    for layout in response_json:
        if layout is None:
            continue
        if not 'subType' in layout:
            layout['subType'] = 'para'
        # 根据类型设置Markdown格式
        if layout["type"] == "title":
            # 文档标题使用一级标题
            markdown_text += "\n\n\n# " + layout["text"] + '\n'
        else:
            # 正文使用段落格式
            markdown_text += layout["text"] + "\n"
    return markdown_text

def documentfile2document(input_file):
    """
    从JSON文件中读取并转换为Document对象列表
    Args:
        input_file: 输入文件路径
    Returns:
        Document对象列表
    """
    documents = []
    for doc in json.load(open(input_file, 'r')):
        document = Document.from_dict(doc)
        documents.append(document)
    return documents

def idpfile2document(input_file):
    """
    将IDP格式的文件转换为Document对象
    Args:
        input_file: 输入文件路径
    Returns:
        包含转换后文本的Document对象列表
    """
    content_json = json.load(open(input_file, 'r'))
    text = transform_idp2markdown(content_json)
    metadata = {"file_name": input_file}
    documents = [Document(text=text, metadata=metadata)]
    return documents

def text2document(input_file):
    """
    将纯文本文件转换为Document对象
    Args:
        input_file: 输入文件路径
    Returns:
        包含文本内容的Document对象列表
    """
    text = open(input_file, 'r').read()
    metadata = {"file_name": input_file}
    documents = [Document(text=text, metadata=metadata)]
    return documents

def idpfile2text(input_file):
    """
    将IDP格式的文件转换为纯文本
    Args:
        input_file: 输入文件路径
    Returns:
        转换后的文本字符串
    """
    content_json = json.load(open(input_file, 'r'))
    text = transform_idp2markdown(content_json)
    return text