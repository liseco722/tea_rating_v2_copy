"""
tab4_cases.py
==============
判例库设计 Tab - 升级版
"""

import streamlit as st
import time
import os
import logging
import numpy as np
import faiss
from pathlib import Path

from config.constants import FACTORS

logger = logging.getLogger(__name__)


def render_tab4(embedder):
    """渲染判例库设计 Tab - 升级版"""
    with st.container():
        # 两列布局
        col_basic, col_supp = st.columns([1, 1])

        with col_basic:
            _render_basic_cases_panel(embedder)

        with col_supp:
            _render_supp_cases_panel(embedder)


def _render_basic_cases_panel(embedder):
    """渲染基础判例面板 - 升级版"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #EDF5EB 0%, #E8F5E9 100%);
                border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6BAA4A;">
        <div style="color: #2D4A1C; font-size: 1.1em; font-weight: 600;">📗 基础判例</div>
        <div style="color: #666; font-size: 0.85rem; margin-top: 4px;">最基础的判例将作为基本信息，全部提供给评分模型学习。</div>
    </div>
    """, unsafe_allow_html=True)

    # 统计信息
    basic_count = len(st.session_state.basic_cases)
    st.markdown(f"""
    <div style="padding: 8px 12px; background: white; border: 1px solid #E8E8E8; border-radius: 6px; margin-bottom: 15px;">
        <span style="color: #4A5D53; font-weight: 600;">📊 当前数量：</span>
        <span style="color: #6BAA4A; font-weight: 600;">{basic_count} 条</span>
    </div>
    """, unsafe_allow_html=True)

    # 查看全部按钮
    if st.button("📋 查看全部", width='stretch', key="show_basic"):
        st.session_state.show_basic_cases = True

    # 手动添加区域
    with st.expander("➕ 手动添加基础判例", expanded=st.session_state.get('expand_add_basic', False)):
        _render_manual_add_form("basic", embedder)

    # 批量添加区域
    with st.expander("📦 批量导入（Excel）"):
        _render_batch_add_section("basic")


def _render_supp_cases_panel(embedder):
    """渲染进阶判例面板 - 升级版"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #FDF6ED 0%, #FAF0D8 100%);
                border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #C9B037;">
        <div style="color: #8B5A2B; font-size: 1.1em; font-weight: 600;">📘 进阶判例</div>
        <div style="color: #666; font-size: 0.85rem; margin-top: 4px;">作为额外信息，经相似度比较后，筛选出最相似的部分提供给评分模型学习。</div>
    </div>
    """, unsafe_allow_html=True)

    # 统计信息
    _, supp_data = st.session_state.supp_cases
    supp_count = len(supp_data)

    st.markdown(f"""
    <div style="padding: 8px 12px; background: white; border: 1px solid #E8E8E8; border-radius: 6px; margin-bottom: 15px;">
        <span style="color: #4A5D53; font-weight: 600;">📊 当前数量：</span>
        <span style="color: #C9B037; font-weight: 600;">{supp_count} 条</span>
    </div>
    """, unsafe_allow_html=True)

    # 查看全部按钮
    if st.button("📋 查看全部", width='stretch', key="show_supp"):
        st.session_state.show_supp_cases = True

    # 手动添加区域
    with st.expander("➕ 手动添加进阶判例", expanded=st.session_state.get('expand_add_supp', False)):
        _render_manual_add_form("supp", embedder)

    # 批量添加区域
    with st.expander("📦 批量导入（Excel）"):
        _render_batch_add_section("supp")


def _render_manual_add_form(case_type: str, embedder):
    """渲染手动添加表单 - 升级版"""
    with st.form(f"{case_type}_case_form"):
        # 茶评描述
        f_txt = st.text_area(
            "📝 判例描述",
            height=80,
            key=f"{case_type}_txt",
            placeholder="请输入详细的茶评描述..."
        )

        st.markdown("##### 🏷️ 因子评分详情")

        fc1, fc2 = st.columns(2)
        input_scores = {}

        for i, f in enumerate(FACTORS):
            with (fc1 if i % 2 == 0 else fc2):
                st.markdown(f"**{f}**")

                col_score, col_comment = st.columns([1, 2])

                with col_score:
                    val = st.number_input(
                        "分数",
                        0, 9, 7,
                        key=f"{case_type}_s_{i}",
                        label_visibility="collapsed"
                    )

                with col_comment:
                    cmt = st.text_input(
                        "评语",
                        key=f"{case_type}_c_{i}",
                        label_visibility="collapsed"
                    )

                sug = st.text_input(
                    "建议",
                    key=f"{case_type}_a_{i}",
                    placeholder="改进建议..."
                )

                input_scores[f] = {
                    "score": val,
                    "comment": cmt,
                    "suggestion": sug
                }

        # 宗师总评
        f_master = st.text_area(
            "🍵 宗师总评",
            key=f"{case_type}_master",
            height=60,
            placeholder="请输入总评..."
        )

        # 自定义按钮颜色
        button_color = "#6BAA4A" if case_type == "basic" else "#C9B037"
        st.markdown(f"""
        <style>
        .stForm button[kind="primary"] {{
            background-color: {button_color} !important;
            color: white !important;
            border: none !important;
        }}
        .stForm button[kind="primary"]:hover {{
            background-color: {button_color}DD !important;
        }}
        button[kind="primary"] {{
            background-color: {button_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # 提交按钮
        col1, col2 = st.columns([1, 1])

        with col1:
            submitted = st.form_submit_button("保存判例", type="primary", width='stretch')

        with col2:
            if st.form_submit_button("重置", type="secondary", width='stretch'):
                st.rerun()

        if submitted:
            _save_case(f_txt, input_scores, f_master, case_type, embedder)


def _render_batch_add_section(case_type: str):
    """渲染批量添加区域 - 升级版"""
    st.caption("📋 请按照模板格式填写判例数据")

    # 文件上传
    bc_file = st.file_uploader(
        "上传已填写的判例文件",
        type=['xlsx', 'xls'],
        key=f"{case_type}_batch_upload",
        help="支持 Excel 格式"
    )

    # 下载模板按钮
    template_path = PATHS.template_file
    if template_path:
        with open(template_path, 'rb') as f:
            template_data = f.read()

        st.download_button(
            "⬇️ 下载判例模板",
            template_data,
            "判例库导入模板.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_template_{case_type}",
            width='stretch'
        )
    else:
        st.caption("📄 模板文件未找到")

    if bc_file:
        st.success(f"✅ 已选择：{bc_file.name}")

        if st.button("📤 导入判例", key=f"{case_type}_batch_import", type="primary", width='stretch'):
            st.info("⚠️ 批量导入功能待完整实现")


def _save_case(text, scores, master_comment, case_type, embedder):
    """保存判例 - 升级版（进阶判例包含 embedding）"""
    from config.settings import PATHS
    from core.resource_manager import ResourceManager

    if not text or not scores:
        st.warning("⚠️ 请填写完整的判例信息")
        return

    new_c = {
        "text": text,
        "scores": scores,
        "master_comment": master_comment,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if case_type == "basic":
        # ===== 基础判例：只需要保存到 JSON =====
        st.session_state.basic_cases.append(new_c)
        ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)
        st.success("✅ 已保存基础判例！")
        st.balloons()
    else:
        # ===== 进阶判例：需要做 embedding 并添加到 FAISS 索引 =====
        supp_idx, supp_data = st.session_state.supp_cases

        try:
            # 1. 编码文本为嵌入向量
            embedding = embedder.encode([text])
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            # 2. 检查维度是否匹配
            if supp_idx.ntotal > 0 and embedding.shape[1] != supp_idx.d:
                # 维度不匹配，需要重建索引
                st.warning(f"⚠️ 向量维度不匹配（索引: {supp_idx.d}, 当前: {embedding.shape[1]}），正在重建索引...")

                all_texts = [item["text"] for item in supp_data] + [text]
                all_embeddings = embedder.encode(all_texts)
                if not isinstance(all_embeddings, np.ndarray):
                    all_embeddings = np.array(all_embeddings, dtype=np.float32)

                new_idx = faiss.IndexFlatL2(all_embeddings.shape[1])
                new_idx.add(all_embeddings.astype('float32'))

                supp_data.append(new_c)
                st.session_state.supp_cases = (new_idx, supp_data)

                ResourceManager.save(
                    new_idx, supp_data,
                    PATHS.supp_case_index, PATHS.supp_case_data,
                    is_json=True
                )
            else:
                # 维度匹配或索引为空
                if supp_idx.ntotal == 0:
                    # 空索引，创建新的
                    new_idx = faiss.IndexFlatL2(embedding.shape[1])
                    new_idx.add(embedding.astype('float32'))
                    supp_data.append(new_c)
                    st.session_state.supp_cases = (new_idx, supp_data)

                    ResourceManager.save(
                        new_idx, supp_data,
                        PATHS.supp_case_index, PATHS.supp_case_data,
                        is_json=True
                    )
                else:
                    # 正常追加
                    supp_idx.add(embedding.astype('float32'))
                    supp_data.append(new_c)
                    st.session_state.supp_cases = (supp_idx, supp_data)

                    ResourceManager.save(
                        supp_idx, supp_data,
                        PATHS.supp_case_index, PATHS.supp_case_data,
                        is_json=True
                    )

            st.success("✅ 已保存进阶判例！")
            st.balloons()

        except Exception as e:
            # 即使 embedding 失败，也要保存数据（下次启动时可重建索引）
            logger.error(f"进阶判例 embedding 失败: {e}")
            supp_data.append(new_c)
            st.session_state.supp_cases = (supp_idx, supp_data)
            ResourceManager.save_json(supp_data, PATHS.supp_case_data)
            st.warning(f"⚠️ 判例已保存但向量索引更新失败: {e}")
            st.info("💡 下次启动应用时会自动重建索引")

    time.sleep(0.3)
    st.rerun()
