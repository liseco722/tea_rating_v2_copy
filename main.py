"""
main.py
========
茶饮六因子AI评分器 Pro - 主入口

模块化重构后的主程序
"""

import sys
import os
import logging

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 配置日志（在所有导入之前）
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

import streamlit as st
import json
import time
import pickle

# ==========================================
# 导入模块
# ==========================================

# 配置模块
from config.settings import apply_page_config, apply_css_styles, PATHS
from config.constants import DEFAULT_USER_TEMPLATE

# 核心模块
from core.resource_manager import ResourceManager
from core import bootstrap_cases

# UI 模块
from ui.sidebar import render_sidebar
from ui.dialogs import (
    show_prompt_dialog,
    show_tea_examples_dialog,
    manage_tea_examples_dialog,
    edit_tea_example_dialog
)
from ui.tab1_interactive import render_tab1
from ui.tab2_batch import render_tab2
from ui.tab3_knowledge import render_tab3
from ui.tab4_cases import render_tab4
from ui.tab5_finetune import render_tab5
from ui.tab6_prompts import render_tab6


# ==========================================
# 页面配置
# ==========================================

apply_page_config()
apply_css_styles()


# ==========================================
# Session 初始化
# ==========================================

if 'loaded' not in st.session_state:
    logger.info("=" * 60)
    logger.info("========== 茶饮六因子AI评分器 - 系统初始化 ==========")
    logger.info("=" * 60)

    # 1. 加载知识库缓存
    logger.info("步骤 1/5: 加载知识库缓存...")
    try:
        kb_idx, kb_data = ResourceManager.load(PATHS.kb_index, PATHS.kb_chunks)
        st.session_state.kb = (kb_idx, kb_data)
        st.session_state.kb_files = ResourceManager.load_kb_files()
    except:
        st.session_state.kb = (None, [])
        st.session_state.kb_files = []
        kb_data = []
    logger.info(f"  → 知识库: {len(st.session_state.kb[1])} 个片段")

    # 2. 加载判例库（基础 + 进阶）
    logger.info("步骤 2/5: 加载判例库...")
    st.session_state.basic_cases = ResourceManager.load_external_json(PATHS.basic_case_data, fallback=[])

    # ===== 修复：从正确的 supp_case_data 路径加载进阶判例 =====
    supp_data = ResourceManager.load_external_json(PATHS.supp_case_data, fallback=[])

    # 尝试加载进阶判例的 FAISS 索引
    if PATHS.supp_case_index.exists():
        try:
            import faiss
            supp_idx = faiss.read_index(str(PATHS.supp_case_index))
            logger.info(f"  → 进阶判例索引已加载，维度={supp_idx.d}，向量数={supp_idx.ntotal}")
        except Exception as e:
            logger.warning(f"  ⚠️ 进阶判例索引加载失败: {e}")
            import faiss
            supp_idx = faiss.IndexFlatL2(768)
    else:
        import faiss
        supp_idx = faiss.IndexFlatL2(768)
        logger.info("  → 进阶判例索引不存在，使用空索引")

    # ===== 标记：如果有 supp_data 但索引为空或不存在，稍后需要重建 =====
    if len(supp_data) > 0 and (supp_idx.ntotal == 0 or supp_idx.ntotal != len(supp_data)):
        st.session_state.supp_index_needs_rebuild = True
        logger.info(f"  ⚠️ 进阶判例数据 {len(supp_data)} 条，索引向量 {supp_idx.ntotal} 条，需要重建索引")
    else:
        st.session_state.supp_index_needs_rebuild = False

    st.session_state.supp_cases = (supp_idx, supp_data)
    logger.info(f"  → 基础判例: {len(st.session_state.basic_cases)} 条")
    logger.info(f"  → 进阶判例: {len(supp_data)} 条")

    # 3. RAG 延迟加载标记
    logger.info("步骤 3/5: 检查 RAG 状态...")
    kb_data = st.session_state.kb[1]
    if not kb_data or len(kb_data) == 0:
        st.session_state.rag_loading_needed = True
        st.session_state.rag_loading_status = "pending"
        logger.info("  ⚠️ 本地知识库为空，将在侧边栏加载")
    else:
        st.session_state.rag_loading_needed = False
        st.session_state.rag_loading_status = "complete"
        logger.info(f"  ✅ 已加载 {len(kb_data)} 个知识片段")

    # 4. 加载 Prompt 配置
    logger.info("步骤 4/5: 加载 Prompt 配置...")
    if PATHS.prompt_config_file.exists():
        try:
            with open(PATHS.prompt_config_file, 'r', encoding='utf-8') as f:
                st.session_state.prompt_config = json.load(f)
        except Exception as e:
            logger.warning(f"Prompt 配置文件加载失败，使用备用文件: {e}")
            st.session_state.prompt_config = {
                "system_template": ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, ""),
                "user_template": DEFAULT_USER_TEMPLATE
            }
    else:
        st.session_state.prompt_config = {
            "system_template": ResourceManager.load_external_text(PATHS.SRC_SYS_PROMPT, ""),
            "user_template": DEFAULT_USER_TEMPLATE
        }
    logger.info("  ✅ Prompt 配置加载完成")

    # 5. 加载茶评示例
    logger.info("步骤 5/5: 加载茶评示例...")
    tea_examples = ResourceManager.load_tea_examples()
    if tea_examples is None or len(tea_examples) == 0:
        from config.constants import TEA_EXAMPLES
        st.session_state.tea_examples = TEA_EXAMPLES[:]
        logger.info("  ✅ 使用默认茶评示例")
    else:
        st.session_state.tea_examples = tea_examples
        logger.info(f"  ✅ 已加载 {len(tea_examples)} 个茶评示例")

    st.session_state.loaded = True
    logger.info("========== 系统初始化完成 ==========")


# ==========================================
# 渲染侧边栏
# ==========================================

embedder, client, client_d, model_id = render_sidebar()

# 初始化判例（加守卫，避免每次 rerun 重复执行）
if 'cases_bootstrapped' not in st.session_state:
    bootstrap_cases(embedder)
    st.session_state.cases_bootstrapped = True

# ===== 自动重建进阶判例索引（在 embedder 可用后执行） =====
if st.session_state.get('supp_index_needs_rebuild', False) and embedder:
    import faiss
    import numpy as np

    supp_idx, supp_data = st.session_state.supp_cases
    if len(supp_data) > 0:
        try:
            with st.spinner("🔄 正在重建进阶判例索引..."):
                all_texts = [item.get("text", "") for item in supp_data]
                all_embeddings = embedder.encode(all_texts)

                if not isinstance(all_embeddings, np.ndarray):
                    all_embeddings = np.array(all_embeddings, dtype=np.float32)

                new_idx = faiss.IndexFlatL2(all_embeddings.shape[1])
                new_idx.add(all_embeddings.astype('float32'))

                st.session_state.supp_cases = (new_idx, supp_data)

                # 持久化
                ResourceManager.save(
                    new_idx, supp_data,
                    PATHS.supp_case_index, PATHS.supp_case_data,
                    is_json=True
                )
                logger.info(f"✅ 进阶判例索引已重建: {new_idx.ntotal} 条向量，维度={new_idx.d}")
        except Exception as e:
            logger.error(f"❌ 进阶判例索引重建失败: {e}")

    st.session_state.supp_index_needs_rebuild = False


# ==========================================
# Hero 区域
# ==========================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🍵 茶品六因子 AI 评分器 Pro</div>
    <div class="slogan">"一片叶子落入水中，改变了水的味道..."</div>
</div>
""", unsafe_allow_html=True)

# Tab 定义 (更简洁的标签)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["交互评分", "批量评分", "知识库","判例库", "模型微调", "Prompt配置"])

# ==========================================
# 渲染各 Tab
# ==========================================

with tab1:
    render_tab1(embedder, client, client_d, model_id)

with tab2:
    render_tab2(embedder, client, client_d, model_id)

with tab3:
    render_tab3()

with tab4:
    render_tab4(embedder)

with tab5:
    render_tab5()

with tab6:
    render_tab6()


# ==========================================
# 弹窗处理
# ==========================================
# 使用 if-elif 链确保同一脚本运行中最多打开一个弹窗

if st.session_state.get('show_prompt_dialog'):
    show_prompt_dialog()
    st.session_state.show_prompt_dialog = False

elif st.session_state.get('show_tea_examples'):
    show_tea_examples_dialog()
    if st.session_state.get('editing_tea_example_idx') is None:
        st.session_state.show_tea_examples = False

elif st.session_state.get('manage_tea_examples'):
    manage_tea_examples_dialog()
    st.session_state.manage_tea_examples = False

elif st.session_state.get('editing_tea_example_idx') is not None:
    idx = st.session_state.editing_tea_example_idx
    edit_tea_example_dialog(idx)
    st.session_state.editing_tea_example_idx = None

elif st.session_state.get('show_basic_cases'):
    from ui.dialogs import show_basic_cases_dialog
    show_basic_cases_dialog(embedder)
    st.session_state.show_basic_cases = False

elif st.session_state.get('show_supp_cases'):
    from ui.dialogs import show_supp_cases_dialog
    show_supp_cases_dialog(embedder)
    st.session_state.show_supp_cases = False

elif st.session_state.get('editing_basic_idx') is not None:
    from ui.dialogs import edit_basic_case_dialog
    edit_basic_case_dialog(st.session_state.editing_basic_idx)

elif st.session_state.get('editing_supp_idx') is not None:
    from ui.dialogs import edit_supp_case_dialog
    edit_supp_case_dialog(st.session_state.editing_supp_idx, embedder)
