"""
sidebar.py
===========
侧边栏 UI 组件 - 控制台风格
"""

import os
import time
import logging
import streamlit as st
import requests
from openai import OpenAI

from config.settings import PATHS
from config.constants import TEA_EXAMPLES
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

        # RAG 延迟加载
        _handle_rag_loading()

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 数据概览
        st.markdown("**数据概览**")

        kb_count = len(st.session_state.kb[1])
        basic_count = len(st.session_state.basic_cases)
        supp_count = len(st.session_state.supp_cases[1])

        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.5rem;'>知识库：{kb_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>基础判例：{basic_count}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; color: #666; margin-top: 0.25rem;'>进阶判例：{supp_count}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1px; background: #E8E8E8; margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # 茶评示例区域
        st.markdown("**📚 茶评示例**")

        if st.button("茶评示例", width='stretch'):
            st.session_state.show_tea_examples = True

    return embedder, client, client_d, model_id


def _handle_rag_loading():
    """处理 RAG 延迟加载逻辑 - 查询缓存"""
    from config.settings import PATHS

    kb_has_data = st.session_state.kb[1] is not None and len(st.session_state.kb[1]) > 0
    loading_status = st.session_state.get('rag_loading_status', 'pending')
    rag_loading_needed = st.session_state.get('rag_loading_needed', False)

    try:
        local_files = list(PATHS.RAG_DIR.glob("*.txt")) + \
                     list(PATHS.RAG_DIR.glob("*.pdf")) + \
                     list(PATHS.RAG_DIR.glob("*.docx"))
        local_file_count = len(local_files) if PATHS.RAG_DIR.exists() else 0
    except Exception as e:
        logger.warning(f"获取本地 RAG 文件列表失败: {e}")
        local_file_count = 0

    local_index_exists = PATHS.kb_index.exists()

    # 情况1：加载失败
    if loading_status == 'failed':
        st.warning("⚠️ 知识库加载失败")
        if st.button("🔄 重试加载", type="secondary", width='stretch'):
            st.session_state.rag_loading_status = 'pending'
            st.session_state.rag_loading_needed = True
            st.rerun()
        return

    # 情况2：正在加载中
    if loading_status == 'loading':
        st.info("🔄 正在加载知识库，请稍候...")
        return

    # 情况3：知识库有数据
    if kb_has_data:
        st.success("✅ 知识库加载成功")
        kb_count = len(st.session_state.kb[1])
        st.caption(f"📊 已加载 {kb_count} 个知识片段")
        st.caption(f"📁 本地 {local_file_count} 个文件")

    # 情况4：知识库为空，需要加载
    if rag_loading_needed and loading_status == 'pending':
        with st.status("🔄 正在加载知识库...", expanded=True) as status:
            st.write("📂 读取本地 RAG 文件...")
            st.session_state.rag_loading_status = 'loading'

            try:
                from core import load_rag_from_local
                embedder = st.session_state.get('embedder')
                success, msg = load_rag_from_local(embedder)

                if success:
                    status.update(label="✅ 知识库加载完成", state="complete", expanded=False)
                    st.session_state.rag_loading_status = 'complete'
                    # 性能优化：移除了 time.sleep(1)
                    st.rerun()
                else:
                    status.update(label="❌ 知识库加载失败", state="error", expanded=True)
                    st.error(msg)
                    st.info("💡 您可以在「知识库设计」手动上传 RAG 文件")
                    st.session_state.rag_loading_status = 'failed'

                    if st.button("🔄 重试加载", type="secondary"):
                        st.session_state.rag_loading_status = 'pending'
                        st.rerun()

            except Exception as e:
                status.update(label="❌ 加载出错", state="error", expanded=True)
                st.error(f"加载失败: {str(e)}")
                logger.error(f"RAG 加载失败: {e}", exc_info=True)
                st.session_state.rag_loading_status = 'failed'

                if st.button("🔄 重试加载", type="secondary"):
                    st.session_state.rag_loading_status = 'pending'
                    st.rerun()
        return

    # 情况5：知识库为空且不需要加载
    if not kb_has_data and not rag_loading_needed:
        if local_file_count > 0 and not local_index_exists:
            st.info(f"📂 发现本地有 {local_file_count} 个文件")
            if st.button("📥 加载知识库", type="primary", width='stretch'):
                st.session_state.rag_loading_needed = True
                st.session_state.rag_loading_status = 'pending'
                st.rerun()
            return

        if local_file_count == 0:
            st.info("💡 请在「知识库设计」上传文件")
            return

        st.info("💡 请在「知识库设计」添加文件或点击加载")

