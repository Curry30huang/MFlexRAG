import asyncio
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def weighted_mean_pooling(hidden, attention_mask):
    """
    使用加权平均池化方法处理隐藏状态
    Args:
        hidden: 隐藏状态张量
        attention_mask: 注意力掩码
    Returns:
        加权平均后的表示向量
    """
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


class VL_Embedding(MultiModalEmbedding):
    """
    视觉语言多模态嵌入模型类
    支持文本和图像的嵌入表示
    """
    model: str = Field(description="使用的多模态模型名称")

    api_key: Optional[str] = Field(
        default=None,
        description="API密钥",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "输出嵌入向量的维度数。"
            "仅在embedding-3及以后版本支持。embedding-2固定为1024维。"
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="超时时间",
    )

    mode: str = Field(
        default="text",
        description="模型模式，可选'text'或'image'",
    )
    show_progress: bool = Field(
        default=False,
        description="是否显示进度条",
    )

    embed_model: Union[ColQwen2, AutoModel, None] = Field(default=None)
    processor: Optional[Union[ColQwen2Processor, ColPaliProcessor]] = Field(default=None)
    tokenizer: Optional[AutoTokenizer] = Field(default=None)

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = "text",
        **kwargs: Any,
    ) -> None:
        """
        初始化视觉语言嵌入模型
        Args:
            model: 模型名称
            dimensions: 嵌入维度
            timeout: 超时时间
            callback_manager: 回调管理器
            mode: 模型模式
            **kwargs: 其他参数
        """
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )

        self.mode = mode

        # 根据不同的模型类型初始化相应的模型和处理器
        # 目前就对通义相关模型做了适配，cuda选择空闲的卡号码
        if "vidore" in model and "qwen" in model:
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda:4",
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(model)
        elif "vidore" in model and "pali" in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda:4",
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)

    @classmethod
    def class_name(cls) -> str:
        """返回类名"""
        return "VL_Embedding"

    def _process_embeddings(self, embeddings):
        """
        统一处理嵌入向量格式
        Args:
            embeddings: 原始嵌入向量
        Returns:
            处理后的嵌入向量列表
        """
        if isinstance(embeddings, torch.Tensor):
            # 先将BFloat16转换为Float32
            embeddings = embeddings.to(torch.float32)
            embeddings = embeddings.detach().cpu().numpy()

        # 确保向量是2D数组
        if len(embeddings.shape) == 3:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        elif len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # 转换为列表格式
        return embeddings.tolist()

    def embed_img(self, img_path):
        """
        生成图像的嵌入表示
        Args:
            img_path: 图像路径或路径列表
        Returns:
            图像的嵌入向量
        """
        if isinstance(img_path, str):
            img_path = [img_path]

        try:
            if "vidore" in self.model:
                images = [Image.open(img) for img in img_path]
                batch_images = self.processor.process_images(images).to(
                    self.embed_model.device
                )
                with torch.no_grad():
                    image_embeddings = self.embed_model(**batch_images)
            return image_embeddings
        except Exception as e:
            print(f"Error processing images: {str(e)}")
            raise

    def embed_text(self, text):
        """
        生成文本的嵌入表示
        Args:
            text: 输入文本或文本列表
        Returns:
            文本的嵌入向量
        """
        if isinstance(text, str):
            text = [text]
        try:
            if "colqwen" in self.model:
                batch_queries = self.processor.process_queries(text).to(
                    self.embed_model.device
                )
                with torch.no_grad():
                    query_embeddings = self.embed_model(**batch_queries)
            elif "colpali" in self.model:
                batch_queries = self.processor.process_queries(text).to(
                    self.embed_model.device
                )
                with torch.no_grad():
                    query_embeddings = self.embed_model(**batch_queries)
            return query_embeddings
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的嵌入向量列表"""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return self.embed_text(query)[0]

    def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取单个文本的嵌入向量"""
        return self.embed_text(text)[0]

    def _get_image_embedding(self, img_file_path) -> Embedding:
        """获取图像的嵌入向量"""
        embeddings = self.embed_img(img_file_path)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings.flatten().tolist()

    def _aget_image_embedding(self, img_file_path) -> Embedding:
        """异步获取图像的嵌入向量"""
        embeddings = self.embed_img(img_file_path)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings.flatten().tolist()

    def __call__(self, nodes, **kwargs):
        """
        处理节点列表，为每个节点生成嵌入向量
        Args:
            nodes: 节点列表
            **kwargs: 其他参数
        Returns:
            处理后的节点列表
        """
        try:
            if "vidore" in self.model:
                if self.mode == "image":
                    # 为每个节点单独处理
                    for i, node in enumerate(nodes):
                        # 获取图片路径
                        img_path = None
                        if hasattr(node, 'image_path'):
                            img_path = node.image_path
                        elif hasattr(node, 'metadata'):
                            img_path = node.metadata.get('file_path') or node.metadata.get('img_path')

                        if not img_path:
                            print(f"Warning: No image path found for node {i}")
                            continue

                        # 单独处理每个图像
                        try:
                            single_embedding = self.embed_img([img_path])
                            # 使用统一的向量处理方法
                            node_embedding = self._process_embeddings(single_embedding)[0]
                            node.embedding = node_embedding
                        except Exception as e:
                            print(f"Error processing image {img_path}: {str(e)}")
                            continue
                else:
                    # 获取节点文本内容
                    texts = [getattr(node, 'text', '') or getattr(node, 'content', '') for node in nodes]
                    embeddings = self.embed_text(texts)
                    # 使用统一的向量处理方法
                    processed_embeddings = self._process_embeddings(embeddings)

                    for node, embedding in zip(nodes, processed_embeddings):
                        node.embedding = embedding

            return nodes
        except Exception as e:
            print(f"Error in __call__: {str(e)}")
            raise

    def score(self, image_embeddings, text_embeddings):
        """
        计算图像和文本嵌入向量之间的相似度分数
        Args:
            image_embeddings: 图像嵌入向量
            text_embeddings: 文本嵌入向量
        Returns:
            相似度分数
        """
        if "vidore" in self.model:
            score = self.processor.score_multi_vector(
                image_embeddings, text_embeddings
            )
        return score


if __name__ == "__main__":
    # 测试代码
    colpali = VL_Embedding("vidore/colqwen2-v1.0")
    image_embeddings = colpali.embed_img(
        "./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg"
    )
    text_embeddings = colpali.embed_text("Hello, world!")
    score = colpali.processor.score_multi_vector(
        image_embeddings, text_embeddings
    )
    # 打印形状和分数
    print(image_embeddings.shape)  # torch.Size([1, 779, 128]) 第一个维度表示有几个图片，第二维度表示模型对图片的"token化"表示，类似于文本处理中的token序列长度，相当于有779个视觉单元；第三个维度表示每个视觉单元的特征，表示每一个视觉单元具体的特征
    print(text_embeddings.shape) # torch.Size([1, 26, 128]) 第一个维度表示有几个文本块，第二维度表示模型对文本的"token化"表示，相当于有26个token；第三个维度表示每个token的详细特征
    print(score)  # tensor([[115.5000]])