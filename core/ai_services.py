"""
ai_services.py
===============
AI 服务封装模块
"""

import numpy as np
import requests
import logging
from typing import List, Union
from openai import OpenAI

logger = logging.getLogger(__name__)


class SelfHostedEmbedder:
    """
    自部署 Embedding 服务客户端
    端点：/embedding
    模型：bge-base-zh-v1.5（输出维度 768）
    """

    def __init__(self, base_url: str, model_name: str = "bge-base-zh-v1.5", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        logger.info(f"SelfHostedEmbedder 初始化: base_url={self.base_url}, model={self.model_name}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        if not texts:
            return []
        if isinstance(texts, str):
            texts = [texts]

        url = f"{self.base_url}/embedding"
        payload = {"text": texts}

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"Embedding API 返回 HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()

        # 兼容多种返回格式
        if "embedding" in data:
            # 单条返回 [float, ...] → 包成 [[float, ...]]
            if isinstance(data["embedding"], list) and data["embedding"] and isinstance(data["embedding"][0], float):
                return [data["embedding"]]
            # 批量返回 [[float, ...], ...] → 直接用
            return data["embedding"]
        if "embeddings" in data:
            return data["embeddings"]
        if "data" in data and isinstance(data["data"], list):
            if data["data"] and "embedding" in data["data"][0]:
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_data]
            if data["data"] and isinstance(data["data"][0], list):
                return data["data"]
        if "vectors" in data:
            return data["vectors"]
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data

        raise RuntimeError(f"无法解析 Embedding 响应，返回字段: {list(data.keys())}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """兼容旧版接口，返回 numpy 数组"""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([])
        embeddings = self.embed_texts(texts)
        return np.array(embeddings, dtype=np.float32)


def llm_normalize_user_input(user_input: str, client: OpenAI) -> str:
    """使用 LLM 用户输入去噪"""
    from config.constants import FILTER_SYSTEM_PROPMT
    f_sys_p = FILTER_SYSTEM_PROPMT
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f_sys_p},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=5000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM 文本标准化失败: {e}")
        return user_input
