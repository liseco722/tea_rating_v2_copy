"""
sidebar.py
===========
侧边栏 UI 组件 - 控制台风格
"""

import logging
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI

from config.settings import PATHS
from core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)



def render_sidebar():
    """
    渲染侧边栏 - 控制台风格

    Returns:
        tuple: (embedder, client, client_d, model_id)
    """
    with st.sidebar:
        # 系统配置区域
        st.markdown("**⚙️ 系统配置**")

        # ========== 从 secrets 读取所有配置 ==========
        embedding_url = st.secrets.get("EMBEDDING_URL", "")
        deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        gpu_server_url = st.secrets.get("GPU_SERVER_URL", "")
        gpu_manager_url = st.secrets.get("GPU_MANAGER_URL", "")

        if not embedding_url or not deepseek_key or not gpu_server_url:
            st.warning("未配置必要的 API Key / 服务地址")
            st.caption("请在 secrets.toml 中配置：EMBEDDING_URL, DEEPSEEK_API_KEY, GPU_SERVER_URL")
            st.stop()
        else:
            st.success("✅ API 就绪")

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 模型配置 - 简化版
        st.markdown("**模型配置**")
        st.markdown("<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>预处理：Deepseek-chat</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; color: #666;'>评分：Qwen3-8B</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.85rem; color: #666;'>向量化：bge-base-zh-v1.5</div>", unsafe_allow_html=True)
        model_id = "Qwen3-8B"

        try:
            resp = requests.get(f"{gpu_manager_url}/status", timeout=2)
            if resp.status_code == 200 and resp.json().get("lora_available"):
                model_id = "default_lora"
                st.success("已启用微调模型")
        except Exception as e:
            logger.warning(f"微调模型状态检查失败: {e}")

        # ========== 缓存服务实例到 session_state（避免每次 rerun 重建） ==========
        if 'embedder' not in st.session_state:
            from core.ai_services import SelfHostedEmbedder
            st.session_state.embedder = SelfHostedEmbedder(embedding_url)
        embedder = st.session_state.embedder

        if 'client' not in st.session_state:
            st.session_state.client = OpenAI(
                api_key="dummy",
                base_url=gpu_server_url,
                timeout=60.0
            )
        client = st.session_state.client

        if 'client_d' not in st.session_state:
            st.session_state.client_d = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com",
                timeout=60.0
            )
        client_d = st.session_state.client_d

        # RAG 缓存与延迟加载（优化版）
        _handle_rag_loading()

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 数据概览
        st.markdown("**数据概览**")

        kb_count = len(st.session_state.get('kb', (None, []))[1])
        basic_count = len(st.session_state.get('basic_cases', []))
        supp_count = len(st.session_state.get('supp_cases', (None, []))[1])

        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>知识库：{kb_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>基础判例：{basic_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>进阶判例：{supp_count}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 茶评示例区域
        st.markdown("**📚 茶评示例**")

        if st.button("茶评示例", width='stretch'):
            st.session_state.show_tea_examples = True

    return embedder, client, client_d, model_id



def _list_local_rag_files():
    if not PATHS.RAG_DIR.exists():
        return []
    return [
        f for f in PATHS.RAG_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {'.txt', '.pdf', '.docx'}
    ]



def _load_kb_from_cache() -> bool:
    """仅从本地缓存加载 KB，不触发 embedding。"""
    try:
        kb_idx, kb_chunks = ResourceManager.load(PATHS.kb_index, PATHS.kb_chunks)
        st.session_state.kb = (kb_idx, kb_chunks)
        st.session_state.kb_files = list(ResourceManager.load_kb_metadata().get("files", {}).keys())
        st.session_state.rag_loading_status = 'complete'
        st.session_state.rag_loading_needed = False
        return True
    except Exception as e:
        logger.warning(f"从缓存加载知识库失败: {e}")
        st.session_state.rag_loading_status = 'failed'
        return False



def _rebuild_kb_via_tab3() -> bool:
    """调用 Tab3 的统一安全重建逻辑。"""
    try:
        from ui.tab3_knowledge import rebuild_rag_cache
        ok, _ = rebuild_rag_cache()
        return ok
    except Exception as e:
        logger.error(f"安全重建知识库失败: {e}", exc_info=True)
        return False



def _handle_rag_loading():
    """处理 RAG 延迟加载逻辑 - 优化版。

    新逻辑：
    1. 优先只加载本地缓存 index/chunks，不自动触发 embedding。
    2. 检测 tea_data/RAG 与缓存元数据是否一致。
    3. 只有用户主动点击时，才做安全重建。
    """
    kb_state = st.session_state.get('kb', (None, []))
    kb_data = kb_state[1] if isinstance(kb_state, tuple) and len(kb_state) > 1 else []
    local_files = _list_local_rag_files()
    local_names = {f.name for f in local_files}

    metadata = ResourceManager.load_kb_metadata()
    cached_names = set(metadata.get("files", {}).keys())
    cache_complete = all([
        PATHS.kb_index.exists(),
        PATHS.kb_chunks.exists(),
        hasattr(PATHS, 'kb_vectors') and PATHS.kb_vectors.exists(),
        PATHS.kb_metadata.exists(),
    ])
    cache_mismatch = local_names != cached_names if local_names or cached_names else False

    # 情况 1：Session 内已经有知识库
    if kb_data:
        st.success("✅ 知识库已就绪")
        st.caption(f"📊 已加载 {len(kb_data)} 个知识片段")
        st.caption(f"📁 本地 {len(local_files)} 个文件")

        if cache_mismatch:
            st.warning("⚠️ 检测到 tea_data/RAG 与缓存索引不一致")
            if st.button("🛡️ 安全重建知识库", type="secondary", width='stretch'):
                with st.spinner("🔄 正在重建知识库索引..."):
                    if _rebuild_kb_via_tab3():
                        st.rerun()
                    st.error("❌ 重建失败，请检查日志")
        return

    # 情况 2：没有任何本地文件
    if not local_files:
        st.info("💡 请在「知识库设计」上传文件")
        st.session_state.rag_loading_status = 'empty'
        st.session_state.rag_loading_needed = False
        return

    # 情况 3：有完整缓存且文件集一致 -> 直接从缓存加载
    if cache_complete and not cache_mismatch:
        if _load_kb_from_cache():
            kb_count = len(st.session_state.get('kb', (None, []))[1])
            st.success("✅ 已从本地缓存载入知识库")
            st.caption(f"📊 已载入 {kb_count} 个知识片段 / {len(local_files)} 个文件")
        else:
            st.warning("⚠️ 本地缓存存在，但加载失败")
        return

    # 情况 4：有本地文件，但缓存缺失或不一致 -> 等用户主动修复
    if cache_mismatch:
        st.warning("⚠️ 检测到 tea_data/RAG 与网页缓存不一致")
        st.caption("建议点击下方按钮重新解析文件、更新 chunks 和 index。")
    else:
        st.info(f"📂 发现本地有 {len(local_files)} 个文件，但缓存索引尚未建立")

    if st.button("🛡️ 构建 / 修复知识库索引", type="primary", width='stretch'):
        with st.spinner("🔄 正在构建知识库索引..."):
            if _rebuild_kb_via_tab3():
                st.rerun()
            st.error("❌ 构建失败，请检查日志")
