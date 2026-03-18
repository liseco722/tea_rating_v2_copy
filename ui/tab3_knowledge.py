"""
tab3_knowledge.py
==================
知识库设计 Tab - 本地模式
"""

import streamlit as st
import os
import io
import shutil
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ==========================================
# 公共入口
# ==========================================

def render_tab3():
    """渲染知识库设计 Tab - 本地模式"""
    with st.container():
        # 1. 本地知识库文件列表
        st.markdown("##### 📁 本地知识库文件")
        _render_local_file_list()

        st.markdown("---")

        # 2. 添加新文件
        st.markdown("##### ➕ 添加新文件")
        _render_upload_section()

        st.markdown("---")

        # 3. 从备份重置
        st.markdown("##### 🔄 从备份重置")
        _render_reset_section()

        st.markdown("---")

        # 4. 手动维护
        st.markdown("##### 🔧 手动维护")
        _render_maintenance_section()


# ==========================================
# 工具函数
# ==========================================

def _get_backup_dir():
    """获取 tea_backup 目录路径（与 tea_data 同级）"""
    from config.settings import PATHS
    # PATHS.RAG_DIR 通常是 tea_data/RAG，所以 tea_backup 在 tea_data 的同级
    backup_dir = PATHS.RAG_DIR.parent.parent / "tea_backup"
    return backup_dir


def _get_local_files():
    """获取本地 RAG 目录中的文件列表"""
    from config.settings import PATHS

    if not PATHS.RAG_DIR.exists():
        return []

    files = []
    for file_path in PATHS.RAG_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in ['.pdf', '.txt', '.docx']:
            files.append({
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'path': file_path
            })

    return sorted(files, key=lambda x: x['name'])


def _get_backup_files():
    """获取 tea_backup 目录中的文件列表"""
    backup_dir = _get_backup_dir()
    if not backup_dir.exists():
        return []

    files = []
    for file_path in backup_dir.iterdir():
        if file_path.is_file() and file_path.suffix in ['.pdf', '.txt', '.docx']:
            files.append({
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'path': file_path
            })

    return sorted(files, key=lambda x: x['name'])


def _backup_files(file_paths):
    """
    将文件永久备份到 tea_backup 目录（仅新增，不覆盖已有同名文件除非内容不同）

    Args:
        file_paths: list of Path 对象，指向 RAG_DIR 中已保存的文件
    """
    backup_dir = _get_backup_dir()
    os.makedirs(backup_dir, exist_ok=True)

    backed_up = []
    for src_path in file_paths:
        dst_path = backup_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
            backed_up.append(src_path.name)
        except Exception as e:
            logger.warning(f"备份文件 {src_path.name} 失败: {e}")

    if backed_up:
        logger.info(f"已备份 {len(backed_up)} 个文件到 tea_backup")
    return backed_up


def _rebuild_all_embeddings():
    """
    重新读取 RAG_DIR 中所有文件，全量重建 embedding 和 FAISS 索引。
    成功返回 (True, chunk_count)，失败返回 (False, error_msg)。
    """
    from config.settings import PATHS
    from core.resource_manager import ResourceManager
    import numpy as np
    import faiss

    embedder = st.session_state.get('embedder')
    if not embedder:
        return False, "Embedding 服务未初始化"

    if not PATHS.RAG_DIR.exists():
        return False, "RAG 目录不存在"

    # ===== 收集所有文件 =====
    rag_files = [
        f for f in PATHS.RAG_DIR.iterdir()
        if f.is_file() and f.suffix in ['.pdf', '.txt', '.docx']
    ]

    if not rag_files:
        # 没有文件时，清空索引
        if PATHS.kb_index.exists():
            os.remove(PATHS.kb_index)
        if PATHS.kb_chunks.exists():
            os.remove(PATHS.kb_chunks)
        st.session_state.kb = (None, [])
        st.session_state.rag_loading_status = 'complete'
        st.session_state.rag_loading_needed = False

        kb_files_list = []
        ResourceManager.save_kb_files(kb_files_list)
        st.session_state.kb_files = kb_files_list

        return True, 0

    # ===== 解析所有文件 =====
    all_chunks = []
    saved_file_names = []

    for file_path in rag_files:
        try:
            text = ""
            if file_path.suffix == '.txt':
                text = file_path.read_text(encoding='utf-8', errors='ignore')
            elif file_path.suffix == '.pdf':
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            elif file_path.suffix == '.docx':
                import docx
                doc = docx.Document(str(file_path))
                text = "\n".join([para.text for para in doc.paragraphs])

            if not text.strip():
                continue

            # 文本分块
            chunk_size = 500
            overlap = 50
            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + chunk_size
                chunk = text[start:end].strip()
                if chunk:
                    all_chunks.append(chunk)
                start = end - overlap

            saved_file_names.append(file_path.name)

        except Exception as e:
            logger.warning(f"解析文件 {file_path.name} 失败: {e}")
            continue

    if not all_chunks:
        return False, "未能从文件中提取到任何文本内容"

    # ===== 向量化 =====
    embeddings = []
    batch_size = 25

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        try:
            batch_embeddings = embedder.embed_texts(batch)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            return False, f"向量化失败: {str(e)}"

    # ===== 保存索引 =====
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    os.makedirs(PATHS.kb_index.parent, exist_ok=True)
    faiss.write_index(index, str(PATHS.kb_index))
    with open(PATHS.kb_chunks, 'wb') as f:
        pickle.dump(all_chunks, f)

    # 更新 session_state
    st.session_state.kb = (index, all_chunks)
    st.session_state.rag_loading_status = 'complete'
    st.session_state.rag_loading_needed = False

    # 更新文件名列表
    ResourceManager.save_kb_files(saved_file_names)
    st.session_state.kb_files = saved_file_names

    return True, len(all_chunks)


def _sync_to_github():
    """触发 GitHub 同步 tea_data（静默执行，失败仅记日志）"""
    try:
        from core.github_sync import GithubSync
        configured, _ = GithubSync.check_config()
        if not configured:
            logger.info("GitHub 未配置，跳过同步")
            return

        success, msg, synced_files = GithubSync.sync_all_data(st.session_state)
        if success:
            logger.info(f"GitHub 同步成功，已同步 {len(synced_files)} 个文件")
            # 刷新侧边栏缓存
            if 'github_status_cache' in st.session_state:
                st.session_state.github_status_cache['needs_refresh'] = True
        else:
            logger.warning(f"GitHub 同步失败: {msg}")
    except Exception as e:
        logger.warning(f"GitHub 同步出错: {e}")


# ==========================================
# UI 渲染
# ==========================================

def _render_local_file_list():
    """渲染本地文件列表"""
    from config.settings import PATHS

    # 刷新文件列表
    if 'refresh_local_files' in st.session_state and st.session_state.refresh_local_files:
        st.session_state.local_rag_files = _get_local_files()
        st.session_state.refresh_local_files = False

    # 初始化文件列表
    if 'local_rag_files' not in st.session_state:
        st.session_state.local_rag_files = _get_local_files()

    local_files = st.session_state.local_rag_files

    if not local_files:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <div class="empty-state-text">暂无本地文件</div>
            <div style="font-size: 0.85rem; color: #999; margin-top: 0.5rem;">
                请上传文件到知识库
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 文件统计
    st.markdown(f"""
    <div style="padding: 10px; background: #EDF5EB; border-radius: 6px; margin-bottom: 15px;">
        <span style="color: #4A5D53; font-weight: 600;">📊 共 {len(local_files)} 个文件</span>
    </div>
    """, unsafe_allow_html=True)

    # 文件列表
    for idx, file_info in enumerate(local_files):
        with st.container(border=True):
            col1, col2, col3 = st.columns([4, 2, 1])

            with col1:
                st.markdown(f"**{file_info['name']}**")
                st.caption(f"{file_info['size'] / 1024:.1f} KB")

            with col2:
                st.caption("本地文件")

            with col3:
                if st.button("🗑️", key=f"del_local_{idx}", help="删除文件"):
                    try:
                        # 1. 删除 RAG_DIR 中的文件（不动 tea_backup）
                        os.remove(file_info['path'])
                        st.success(f"✅ 已删除 {file_info['name']}")

                        # 2. 全量重建 embedding
                        with st.spinner("🔄 正在重建知识库索引..."):
                            ok, result = _rebuild_all_embeddings()
                            if ok:
                                st.success(f"📊 知识库已重建（{result} 个片段）")
                            else:
                                st.warning(f"⚠️ 重建索引时出错: {result}")

                        # 3. 同步到 GitHub
                        _sync_to_github()

                        # 4. 刷新文件列表
                        st.session_state.refresh_local_files = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败: {str(e)}")


def _render_upload_section():
    """渲染上传区域"""
    st.caption("支持格式：PDF、TXT、DOCX")

    # 上传区域
    up = st.file_uploader(
        "选择文件",
        accept_multiple_files=True,
        key="kb_uploader",
        type=['pdf', 'txt', 'docx'],
        help="可一次选择多个文件"
    )

    # 显示已选文件信息
    if up:
        st.markdown(f"""
        <div style="padding: 10px; background: #FDF6ED; border-radius: 6px; margin: 10px 0;">
            <span style="color: #8B5A2B; font-weight: 600;">📋 已选择 {len(up)} 个文件</span>
        </div>
        """, unsafe_allow_html=True)

        for file in up:
            st.markdown(f"- {file.name} ({file.size / 1024:.1f} KB)")

    # 上传按钮
    if st.button("📤 添加到知识库", type="primary", width='stretch'):
        if not up or len(up) == 0:
            st.warning("⚠️ 请先选择要上传的文件")
        else:
            _handle_upload(up)


def _render_reset_section():
    """渲染「从备份重置」区域 —— 从 tea_backup 读取文件，用户勾选后添加到知识库"""
    from config.settings import PATHS

    backup_dir = _get_backup_dir()
    backup_files = _get_backup_files()

    if not backup_files:
        st.markdown(f"""
        <div style="padding: 10px; background: #F5F5F5; border-radius: 6px;">
            <span style="color: #999;">📭 备份目录暂无文件</span>
            <div style="font-size: 0.8rem; color: #BBB; margin-top: 4px;">
                路径：{backup_dir}
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.caption(f"备份目录共 {len(backup_files)} 个文件（上传过的文件会永久保留在此处）")

    # 获取当前 RAG_DIR 中已有的文件名
    current_files = {f['name'] for f in _get_local_files()}

    # 用 checkbox 让用户勾选
    selected = []
    for idx, bf in enumerate(backup_files):
        already_exists = bf['name'] in current_files
        label = f"{bf['name']}（{bf['size'] / 1024:.1f} KB）"
        if already_exists:
            label += "  ✅ 已在知识库中"

        checked = st.checkbox(
            label,
            value=False,
            key=f"reset_sel_{idx}",
            disabled=already_exists
        )
        if checked and not already_exists:
            selected.append(bf)

    # 添加按钮
    if st.button("📥 将选中文件添加到知识库", type="primary", width='stretch'):
        if not selected:
            st.warning("⚠️ 请先勾选要恢复的文件")
        else:
            _handle_reset_restore(selected)


def _render_maintenance_section():
    """渲染手动维护区域"""
    from config.settings import PATHS

    st.caption("管理本地知识库文件")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 刷新文件列表", width='stretch'):
            st.session_state.local_rag_files = _get_local_files()
            st.success(f"✅ 已刷新文件列表，共 {len(st.session_state.local_rag_files)} 个文件")
            st.rerun()

    with col2:
        if st.button("🗑️ 清空知识库", width='stretch'):
            try:
                # 1. 清空 RAG_DIR 中的文件
                if PATHS.RAG_DIR.exists():
                    for f in PATHS.RAG_DIR.iterdir():
                        if f.is_file() and f.suffix in ['.pdf', '.txt', '.docx']:
                            os.remove(f)

                # 2. 清空索引文件
                if PATHS.kb_index.exists():
                    os.remove(PATHS.kb_index)
                if PATHS.kb_chunks.exists():
                    os.remove(PATHS.kb_chunks)

                # 3. 清空 session_state 中的知识库数据
                st.session_state.kb = (None, [])

                # 4. 重置加载状态
                st.session_state.rag_loading_needed = True
                st.session_state.rag_loading_status = "pending"

                # 5. 同步到 GitHub（RAG 目录已清空）
                _sync_to_github()

                st.success("✅ 知识库已清空（备份文件仍保留在 tea_backup 中）")
                st.session_state.refresh_local_files = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ 清空失败: {str(e)}")


# ==========================================
# 核心处理逻辑
# ==========================================

def _handle_upload(files):
    """处理文件上传 —— 保存 + 备份 + 全量重建 embedding + GitHub 同步"""
    from config.settings import PATHS

    with st.spinner("🔄 正在处理文件，请稍候..."):
        try:
            # ===== 步骤 1：保存文件到 RAG_DIR =====
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            saved_paths = []
            for uploaded_file in files:
                file_path = PATHS.RAG_DIR / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

            # ===== 步骤 2：永久备份到 tea_backup =====
            backed_up = _backup_files(saved_paths)
            if backed_up:
                logger.info(f"已备份到 tea_backup: {backed_up}")

            # ===== 步骤 3：全量重建 embedding（所有 RAG_DIR 文件） =====
            ok, result = _rebuild_all_embeddings()
            if not ok:
                st.error(f"❌ 重建索引失败: {result}")
                return

            # ===== 步骤 4：同步到 GitHub =====
            _sync_to_github()

            # 刷新本地文件列表
            st.session_state.refresh_local_files = True

        except Exception as e:
            st.error("❌ 处理失败")
            st.error(f"错误信息: {str(e)}")
            logger.error(f"知识库上传处理失败: {e}", exc_info=True)
            return

    # ===== 显示最终结果 =====
    st.success(f"✅ 成功添加 {len(saved_paths)} 个文件到知识库")
    st.success(f"📊 全量重建完成，共 {result} 个知识片段")
    if backed_up:
        st.info(f"💾 已备份 {len(backed_up)} 个文件到 tea_backup（永久保留）")
    st.info("💡 知识库已更新，现在可以使用这些文件进行检索")


def _handle_reset_restore(selected_files):
    """处理从 tea_backup 恢复选中的文件到知识库"""
    from config.settings import PATHS

    with st.spinner("🔄 正在从备份恢复文件..."):
        try:
            # 1. 将选中的备份文件复制到 RAG_DIR
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            restored = []
            for bf in selected_files:
                dst = PATHS.RAG_DIR / bf['name']
                shutil.copy2(bf['path'], dst)
                restored.append(bf['name'])

            # 2. 全量重建 embedding
            ok, result = _rebuild_all_embeddings()
            if not ok:
                st.error(f"❌ 重建索引失败: {result}")
                return

            # 3. 同步到 GitHub
            _sync_to_github()

            # 4. 刷新文件列表
            st.session_state.refresh_local_files = True

        except Exception as e:
            st.error(f"❌ 恢复失败: {str(e)}")
            logger.error(f"从备份恢复失败: {e}", exc_info=True)
            return

    st.success(f"✅ 已恢复 {len(restored)} 个文件：{', '.join(restored)}")
    st.success(f"📊 全量重建完成，共 {result} 个知识片段")
    st.rerun()
