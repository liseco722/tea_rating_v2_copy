"""
resource_manager.py
====================
资源与数据管理模块
"""

import json
import pickle
import time
import logging
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional


import numpy as np
import streamlit as st
import faiss

from config.settings import PATHS

logger = logging.getLogger(__name__)

# bge-base-zh-v1.5 输出维度
DEFAULT_EMBEDDING_DIM = 768


class ResourceManager:
    """负责外部文件加载、数据持久化、微调数据管理。"""

    # ---------- 通用文件读写 ----------
    @staticmethod
    def _ensure_parent_dir(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_external_text(path: Path, fallback: str = "") -> str:
        """读取外部文本文件。"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback

    @staticmethod
    def load_external_json(path: Path, fallback: Any = None) -> Any:
        """读取外部 JSON 文件。"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"加载文件 {path} 失败: {e}")
        return fallback if fallback is not None else []

    @staticmethod
    def save_json(data: Any, path: Path):
        """保存数据为 JSON 文件。"""
        ResourceManager._ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_pickle(data: Any, path: Path):
        """保存 pickle 文件。"""
        ResourceManager._ensure_parent_dir(path)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_pickle(path: Path, fallback: Any = None) -> Any:
        """读取 pickle 文件。"""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"加载 pickle 失败 ({path}): {e}")
        return [] if fallback is None else fallback

    @staticmethod
    def empty_faiss_index(dim: int = DEFAULT_EMBEDDING_DIM):
        """创建空 FAISS 索引。"""
        return faiss.IndexFlatL2(dim)

    @staticmethod
    def load_index(idx_path: Path, default_dim: int = DEFAULT_EMBEDDING_DIM):
        """仅加载 FAISS 索引文件，不依赖数据文件是否存在。"""
        if idx_path.exists():
            try:
                return faiss.read_index(str(idx_path))
            except Exception as e:
                logger.warning(f"加载索引失败 ({idx_path}): {e}")
        return ResourceManager.empty_faiss_index(default_dim)

    # ---------- FAISS 索引 + 数据 ----------
    @staticmethod
    def save(index: Any, data: Any, idx_path: Path, data_path: Path, is_json: bool = False):
        """保存 FAISS 索引和对应的数据文件。"""
        ResourceManager._ensure_parent_dir(idx_path)
        ResourceManager._ensure_parent_dir(data_path)
        if index is not None:
            faiss.write_index(index, str(idx_path))
        with open(data_path, "w" if is_json else "wb", encoding="utf-8" if is_json else None) as f:
            if is_json:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                pickle.dump(data, f)

    @staticmethod
    def load(idx_path: Path, data_path: Path, is_json: bool = False) -> Tuple[Any, List]:
        """加载 FAISS 索引和对应的数据文件。"""
        if idx_path.exists() and data_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                with open(data_path, "r" if is_json else "rb", encoding="utf-8" if is_json else None) as f:
                    data = json.load(f) if is_json else pickle.load(f)
                return index, data
            except Exception as e:
                logger.warning(f"加载索引/数据失败 ({idx_path}): {e}")
        return ResourceManager.empty_faiss_index(DEFAULT_EMBEDDING_DIM), []

    @staticmethod
    def _normalize_vector(vector: Any, expected_dim: Optional[int] = None) -> Optional[List[float]]:
        """将向量统一转为 list[float]，非法时返回 None。"""
        if vector is None:
            return None

        if isinstance(vector, np.ndarray):
            vector = vector.astype(np.float32).flatten().tolist()
        elif isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]

        if not isinstance(vector, list):
            return None

        try:
            normalized = [float(x) for x in vector]
        except Exception:
            return None

        if expected_dim is not None and len(normalized) != expected_dim:
            return None
        return normalized

    @staticmethod
    def build_index_from_vectors(vectors: List[List[float]], dim: int = DEFAULT_EMBEDDING_DIM):
        """从缓存向量快速构建 FAISS 索引。"""
        arr = np.array(vectors, dtype=np.float32)
        if arr.size == 0:
            return ResourceManager.empty_faiss_index(dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        index = faiss.IndexFlatL2(arr.shape[1])
        index.add(arr.astype('float32'))
        return index

    @staticmethod
    def strip_case_vector(case: Dict) -> Dict:
        """清理基础判例中不需要落盘的隐藏字段。"""
        cleaned = dict(case or {})
        cleaned.pop("_embedding", None)
        return cleaned

    @staticmethod
    def ensure_case_embedding(case: Dict, embedder, expected_dim: int = DEFAULT_EMBEDDING_DIM) -> List[float]:
        """确保单个进阶判例带有可复用的缓存 embedding。"""
        cached = ResourceManager._normalize_vector(case.get("_embedding"), expected_dim)
        if cached is not None:
            case["_embedding"] = cached
            return cached

        text = str(case.get("text", "")).strip()
        if not text:
            raise ValueError("判例缺少 text 字段，无法生成 embedding")

        raw = embedder.encode([text])
        if isinstance(raw, np.ndarray):
            if raw.size == 0:
                raise ValueError("Embedding 服务返回空向量")
            vector = raw[0].tolist() if raw.ndim > 1 else raw.tolist()
        else:
            if not raw:
                raise ValueError("Embedding 服务返回空向量")
            vector = raw[0] if isinstance(raw, list) else raw

        normalized = ResourceManager._normalize_vector(vector, expected_dim)
        if normalized is None:
            raise ValueError("Embedding 维度异常，无法缓存")

        case["_embedding"] = normalized
        return normalized

    @staticmethod
    def sync_supp_cases(cases: List[Dict], embedder=None) -> Tuple[Any, List[Dict]]:
        """同步进阶判例 JSON 与索引。

        规则：
        - 新增 / 编辑时，只对缺失或失效的判例重新做 embedding。
        - 删除 / 迁移时，直接复用缓存向量快速重建索引，不重复调用 embedding。
        """
        normalized_cases = cases or []
        vectors: List[List[float]] = []

        for idx, case in enumerate(normalized_cases):
            cached = ResourceManager._normalize_vector(case.get("_embedding"), DEFAULT_EMBEDDING_DIM)
            if cached is None:
                if embedder is None:
                    raise ValueError(
                        f"进阶判例第 {idx + 1} 条缺少可用缓存向量，且当前没有 embedder 可用于重建"
                    )
                cached = ResourceManager.ensure_case_embedding(case, embedder, DEFAULT_EMBEDDING_DIM)
            else:
                case["_embedding"] = cached
            vectors.append(cached)

        index = ResourceManager.build_index_from_vectors(vectors, DEFAULT_EMBEDDING_DIM)
        ResourceManager.save(index, normalized_cases, PATHS.supp_case_index, PATHS.supp_case_data, is_json=True)
        return index, normalized_cases

    # ---------- RAG 元数据 / 向量缓存 ----------
    @staticmethod
    def save_kb_metadata(metadata: Dict):
        """保存知识库文件 -> chunks 映射元数据。"""
        ResourceManager.save_json(metadata, PATHS.kb_metadata)

    @staticmethod
    def load_kb_metadata() -> Dict:
        """加载知识库文件 -> chunks 映射元数据。"""
        metadata = ResourceManager.load_external_json(PATHS.kb_metadata, fallback={})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("version", 2)
        metadata.setdefault("files", {})
        return metadata

    @staticmethod
    def save_kb_vectors(vectors: List[List[float]]):
        """保存知识库 chunks 对应的向量缓存。"""
        ResourceManager.save_pickle(vectors, PATHS.kb_vectors)

    @staticmethod
    def load_kb_vectors() -> List[List[float]]:
        """加载知识库 chunks 对应的向量缓存。"""
        vectors = ResourceManager.load_pickle(PATHS.kb_vectors, fallback=[])
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        return vectors if isinstance(vectors, list) else []

    # ---------- 微调数据管理 ----------
    @staticmethod
    def _read_existing_finetune_texts() -> set:
        """读取已有微调数据中的判例文本集合，用于去重。"""
        existing = set()
        if PATHS.training_file.exists():
            try:
                with open(PATHS.training_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            user_msg = entry.get("messages", [{}])[1].get("content", "")
                            start = user_msg.find("【待评分产品】")
                            end = user_msg.find("【参考标准")
                            if start >= 0 and end > start:
                                text = user_msg[start + len("【待评分产品】"):end].strip()
                                if text:
                                    existing.add(text)
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"读取微调数据失败: {e}")
        return existing


    
    @staticmethod
    def append_cases_to_finetune(cases: List[Dict], sys_prompt: str, user_tpl: str) -> Tuple[int, int]:
        """
        将判例追加到微调数据文件（Alpaca 格式，自动去重）。
    
        Args:
            cases: 判例列表
            sys_prompt: 系统提示词
            user_tpl: 用户提示词模板
    
        Returns:
            Tuple[int, int]: (新增条数, 跳过的重复条数)
        """
        existing_texts = ResourceManager._read_existing_finetune_texts()
        added, skipped = 0, 0
    
        def build_output(c: Dict) -> str:
            scores = c.get("scores", {})
            outputs = []
    
            for k, v in scores.items():
                score = v.get("score")
                comment = v.get("comment", "")
    
                if score is None:
                    line = f"关于{k}相关的内容有：{comment}"
                    outputs.append(line)
                else:
                    line = f"{k}{score}分，依据是{comment}"
                    outputs.append(line)
                    
            return "。".join(outputs)
    
        try:
            ResourceManager._ensure_parent_dir(PATHS.training_file)
            with open(PATHS.training_file, "a", encoding="utf-8") as f:
                for c in cases:
                    case_text = c.get("text", "").strip()
                    if case_text in existing_texts:
                        skipped += 1
                        continue
    
                    user_content = (
                        user_tpl
                        .replace("{product_desc}", case_text)
                        .replace("{context_text}", "")
                        .replace("{basic_case_text}", "")
                        .replace("{case_text}", "")
                    )
    
                    entry = {
                        "instruction": sys_prompt,
                        "input": user_content,
                        "output": build_output(c)
                    }
    
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    existing_texts.add(case_text)
                    added += 1
    
            return added, skipped
    
        except Exception as e:
            logger.error(f"微调数据追加失败: {e}")
            return 0, 0


    @staticmethod
    def save_ft_status(job_id, status, fine_tuned_model=None):
        """保存微调任务状态。"""
        data = {"job_id": job_id, "status": status, "timestamp": time.time()}
        if fine_tuned_model:
            data["fine_tuned_model"] = fine_tuned_model
        with open(PATHS.ft_status, 'w') as f:
            json.dump(data, f)

    # ---------- RAG 文件列表管理 ----------
    @staticmethod
    def save_kb_files(file_list: List[str]):
        """保存知识库文件列表。"""
        ResourceManager.save_json(file_list, PATHS.kb_files)

    @staticmethod
    def load_kb_files() -> List[str]:
        """加载知识库文件列表。"""
        if PATHS.kb_files.exists():
            try:
                with open(PATHS.kb_files, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载知识库文件列表失败: {e}")
        return []

    # ---------- 茶评示例管理 ----------
    @staticmethod
    def load_tea_examples():
        """加载茶评示例列表，如果文件不存在或数据无效则返回 None（使用默认值）。"""
        if PATHS.tea_examples_file.exists():
            try:
                with open(PATHS.tea_examples_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and all(
                        isinstance(item, dict)
                        and 'title' in item
                        and 'text' in item
                        for item in data
                    ):
                        return data
                    else:
                        st.warning("⚠️ 茶评示例数据格式异常，将使用默认值")
                        return None
            except Exception as e:
                st.error(f"加载茶评示例失败: {e}")
        return None

    @staticmethod
    def save_tea_examples(examples: List[Dict]):
        """保存茶评示例列表到文件。"""
        try:
            ResourceManager.save_json(examples, PATHS.tea_examples_file)
            return True
        except Exception as e:
            st.error(f"保存茶评示例失败: {e}")
            return False
