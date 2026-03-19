"""
tab4_cases.py
==============
判例库设计 Tab - 升级版
"""

import streamlit as st
import time
import os
import logging
from pathlib import Path
import json
from config.constants import FACTORS
from config.settings import PATHS
from core.resource_manager import ResourceManager
from core.github_sync import GithubSync

logger = logging.getLogger(__name__)


def _sync_basic_to_github():
    """同步基础判例 JSON 到 GitHub（静默失败）。"""
    try:
        GithubSync.sync_basic_cases(st.session_state.basic_cases)
    except Exception as e:
        logger.warning(f"同步基础判例到 GitHub 失败: {e}")


def _sync_supp_to_github(supp_data):
    """同步进阶判例 JSON + FAISS index 到 GitHub（静默失败）。"""
    try:
        clean_data = [ResourceManager.strip_case_vector(c) for c in supp_data]
        GithubSync.sync_supp_cases(clean_data)
    except Exception as e:
        logger.warning(f"同步进阶判例 JSON 到 GitHub 失败: {e}")
    try:
        tea_data_dir = PATHS.basic_case_data.parent
        index_path = tea_data_dir / "supp_cases.index"
        if index_path.exists():
            with open(index_path, 'rb') as f:
                content = f.read()
            GithubSync.push_binary_file(
                "tea_data/supp_cases.index", content,
                "Update supp_cases.index"
            )
    except Exception as e:
        logger.warning(f"同步进阶判例 index 到 GitHub 失败: {e}")


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
    from config.settings import PATHS
    st.caption("📋 请按照模板格式填写判例数据")

    # 文件上传
    bc_file = st.file_uploader(
        "上传已填写的判例文件",
        type=['xlsx', 'xls'],
        key=f"{case_type}_batch_upload",
        help="支持 Excel 格式"
    )

    # 下载模板按钮 - 使用相对路径
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
            with st.spinner("正在处理文件..."):
            new_entries = basic_case_process(bc_file)
    
        if new_entries:
            try:
                if case_type == "basic":
                    clean_entries = [ResourceManager.strip_case_vector(x) for x in new_entries]
                    st.session_state.basic_cases.extend(clean_entries)
                    ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)
                    _sync_basic_to_github()
                    st.success(f"✅ 成功导入 {len(new_entries)} 条基础判例！")
                else:
                    _, supp_data = st.session_state.supp_cases
                    for entry in new_entries:
                        ResourceManager.ensure_case_embedding(entry, embedder)
                        supp_data.append(entry)
                    new_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
                    st.session_state.supp_cases = (new_idx, supp_data)
                    _sync_supp_to_github(supp_data)
                    st.success(f"✅ 成功导入 {len(new_entries)} 条进阶判例！")
    
                time.sleep(1)
                st.rerun()
    
            except Exception as e:
                st.error(f"写入失败: {e}")

def _save_case(text, scores, master_comment, case_type, embedder):
    """保存判例 - 增量向量版。"""
    if not text or not scores:
        st.warning("⚠️ 请填写完整的判例信息")
        return

    new_c = {
        "text": text,
        "scores": scores,
        "master_comment": master_comment,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        if case_type == "basic":
            clean_case = ResourceManager.strip_case_vector(new_c)
            st.session_state.basic_cases.append(clean_case)
            ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)
            _sync_basic_to_github()
            st.success("✅ 已保存基础判例！")
        else:
            ResourceManager.ensure_case_embedding(new_c, embedder)
            _, supp_data = st.session_state.supp_cases
            supp_data.append(new_c)
            new_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
            st.session_state.supp_cases = (new_idx, supp_data)
            _sync_supp_to_github(supp_data)
            st.success("✅ 已保存进阶判例，并更新向量索引！")

        st.balloons()
        time.sleep(0.3)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 保存判例失败: {e}")
