"""
scoring.py
===========
核心评分逻辑模块
"""

import json
import time
import logging
import faiss
import numpy as np
from typing import Tuple, List, Dict, Any

import streamlit as st
from openai import OpenAI

from config.constants import FACTORS

logger = logging.getLogger(__name__)


def run_scoring(
    user_input: str,
    kb: Tuple,
    basic_cases: List[Dict],
    supp_cases: Tuple,
    prompt_config: Dict,
    embedder,
    client: OpenAI,
    model_id: str,
    r_num: int = 3,
    c_num: int = 5
) -> Tuple[Dict, str, str, str, str]:
    """
    执行评分逻辑

    Args:
        user_input: 用户输入的茶评描述
        kb: 知识库 (index, data)
        basic_cases: 基础判例列表
        supp_cases: 进阶判例 (index, data)
        prompt_config: 提示词配置
        embedder: 嵌入器
        client: OpenAI 客户端
        model_id: 模型 ID
        r_num: 参考知识库条目数量
        c_num: 参考进阶判例条目数量

    Returns:
        Tuple: (scores, kb_history, case_history, sys_prompt, user_prompt)
    """
    kb_idx, kb_data = kb
    supp_idx, supp_data = supp_cases

    sys_prompt = prompt_config.get('system_template', '')
    user_tpl = prompt_config.get('user_template', '')

    # 1. 从知识库检索相关内容
    kb_context = ""
    kb_history = ""
    if kb_data and len(kb_data) > 0:
        try:
            query_vec = embedder.encode([user_input])
            if len(query_vec) > 0:
                D, I = kb_idx.search(query_vec, min(r_num, len(kb_data)))
                kb_chunks = [kb_data[i] for i in I[0] if i < len(kb_data)]
                kb_context = "\n\n".join(kb_chunks)
                kb_history = f"参考了 {len(kb_chunks)} 条知识库内容"
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")

    # 2. 从进阶判例检索相似案例
    case_context = ""
    case_history = ""
    if supp_data and len(supp_data) > 0:
        try:
            query_vec = embedder.encode([user_input])

            # 确保查询向量是 numpy 数组
            if not isinstance(query_vec, np.ndarray):
                query_vec = np.array(query_vec)

            # 检查维度是否匹配
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)

            if query_vec.shape[1] != supp_idx.d:
                logger.warning(f"判例索引维度不匹配：索引 {supp_idx.d}，查询 {query_vec.shape[1]}")

                # 检查是否有缓存的新索引
                if 'rebuilt_supp_idx' in st.session_state:
                    logger.info("使用缓存的新索引")
                    new_idx = st.session_state.rebuilt_supp_idx
                    D, I = new_idx.search(query_vec, min(c_num, len(supp_data)))
                    similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                    case_context = _format_cases_for_prompt(similar_cases)
                    case_history = f"参考了 {len(similar_cases)} 条相似判例（使用缓存索引）"
                else:
                    logger.info(f"判例数据量：{len(supp_data)} 条，重新编码以匹配维度...")

                # 重新编码所有判例数据
                start_time = time.time()

                batch_size = 16
                all_embeddings = []
                failed_batches = []

                for i in range(0, len(supp_data), batch_size):
                    batch_texts = [item["text"] for item in supp_data[i:i+batch_size]]

                    max_retries = 3
                    batch_embeddings = None
                    for retry in range(max_retries):
                        try:
                            batch_embeddings = embedder.encode(batch_texts)
                            break
                        except Exception as e:
                            logger.warning(f"批次 {i//batch_size + 1} 编码失败（重试 {retry+1}/{max_retries}）: {e}")
                            if retry < max_retries - 1:
                                time.sleep(2 ** retry)  # 指数退避
                            else:
                                failed_batches.append(i//batch_size + 1)
                                batch_embeddings = np.zeros((len(batch_texts), query_vec.shape[1]))

                    if batch_embeddings is not None:
                        all_embeddings.append(batch_embeddings)

                    logger.info(f"已编码 {min(i+batch_size, len(supp_data))}/{len(supp_data)} 条...")

                    # 自部署服务不需要太长的间隔
                    time.sleep(0.05)

                all_embeddings = np.vstack(all_embeddings)

                if not isinstance(all_embeddings, np.ndarray):
                    all_embeddings = np.array(all_embeddings)

                if len(all_embeddings.shape) == 1:
                    all_embeddings = all_embeddings.reshape(1, -1)

                if failed_batches:
                    logger.warning(f"以下批次编码失败，已使用零向量填充: {failed_batches}")

                # 创建新的索引
                new_idx = faiss.IndexFlatL2(all_embeddings.shape[1])
                new_idx.add(all_embeddings.astype('float32'))

                encode_time = time.time() - start_time
                logger.info(f"编码完成，耗时 {encode_time:.2f} 秒")

                # 缓存到 session_state
                st.session_state.supp_cases = (new_idx, supp_data)
                st.session_state.rebuilt_supp_idx = new_idx
                logger.info("✅ 新索引已缓存到 session_state")

                # 持久化到文件
                try:
                    from config.settings import PATHS
                    from core.resource_manager import ResourceManager
                    ResourceManager.save(
                        new_idx,
                        supp_data,
                        PATHS.supp_case_index,
                        PATHS.supp_case_data,
                        is_json=True
                    )
                    logger.info("✅ 新索引已保存到文件")
                except Exception as e:
                    logger.warning(f"索引保存失败: {e}")

                # 使用新索引进行搜索
                D, I = new_idx.search(query_vec, min(c_num, len(supp_data)))
                similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                case_context = _format_cases_for_prompt(similar_cases)
                case_history = f"参考了 {len(similar_cases)} 条相似判例（索引已重建并缓存）"
            elif len(query_vec) > 0:
                D, I = supp_idx.search(query_vec, min(c_num, len(supp_data)))
                similar_cases = [supp_data[i] for i in I[0] if i < len(supp_data)]
                case_context = _format_cases_for_prompt(similar_cases)
                case_history = f"参考了 {len(similar_cases)} 条相似判例"
        except Exception as e:
            logger.error(f"判例检索失败: {e}")
            case_history = "判例检索失败"

    # 3. 格式化基础判例
    basic_context = _format_basic_cases(basic_cases)

    # 4. 构建完整提示词
    user_prompt = user_tpl.format(
        product_desc=user_input,
        context_text=kb_context,
        basic_case_text=basic_context,
        case_text=case_context
    )

    # 保存提示词到 session state 供查看
    try:
        if hasattr(st, 'session_state'):
            st.session_state.last_llm_sys_prompt = sys_prompt
            st.session_state.last_llm_user_prompt = user_prompt
    except Exception:
        pass  # 如果在非 Streamlit 环境中运行，忽略错误

    # 5. 调用 LLM（带超时和重试机制）
    max_retries = 2
    timeout = 60

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=timeout
            )

            content = response.choices[0].message.content
            scores = _parse_llm_response(content)

            return scores, kb_history, case_history, sys_prompt, user_prompt

        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt == max_retries - 1 or "timeout" not in error_msg.lower():
                st.error(f"评分失败: {error_msg}")
                if "timeout" in error_msg.lower():
                    st.warning("💡 模型响应时间过长，建议：\n1. 检查网络连接\n2. 减少输入内容\n3. 联系管理员检查模型服务状态")
                return None, "", "", sys_prompt, user_prompt

            logger.info("由于超时，正在重试...")
            continue


def _format_cases_for_prompt(cases: List[Dict]) -> str:
    """格式化判例用于提示词"""
    if not cases:
        return "暂无相似判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')[:200]
        scores = case.get('scores', {})
        score_str = ", ".join([f"{k}:{v.get('score', 0)}" for k, v in scores.items()])
        parts.append(f"[判例{i}] {text}...\n得分: {score_str}")

    return "\n\n".join(parts)


def _format_basic_cases(cases: List[Dict]) -> str:
    """格式化基础判例"""
    if not cases:
        return "暂无基础判例"

    parts = []
    for i, case in enumerate(cases, 1):
        text = case.get('text', '')[:200]
        parts.append(f"[基础判例{i}] {text}...")

    return "\n\n".join(parts)


def _parse_llm_response(content: str) -> Dict:
    """解析 LLM 响应"""
    try:
        return json.loads(content)
    except Exception:
        # 尝试提取 JSON 块
        try:
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

    # 解析失败
    return None
