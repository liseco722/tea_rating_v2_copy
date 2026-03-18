"""
tab3_knowledge.py
==================
知识库设计 Tab - 增量 Embedding 优化版
"""

import streamlit as st
import os
import io
import shutil
import pickle
import json
import logging
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)


# ==========================================
# 公共入口
# ==========================================

def render_tab3():
    """渲染知识库设计 Tab - 增量 Embedding 版"""
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
        st.markdown("##### 🔄 从备份恢复")
        _render_reset_section()

        st.markdown("---")

        # 4. 安全冗余：全量重建按钮
        st.markdown("##### 🛡️ 安全冗余")
        st.caption("如果知识库索引与文件不匹配，可点击下方按钮全量重建。")
        if st.button("🔄 全量重建知识库索引", type="secondary", width='stretch'):
            with st.spinner("🔄 正在全量重建知识库索引..."):
                ok, result = _rebuild_all_embeddings()
                if ok:
                    st.success(f"✅ 全量重建完成，共 {result} 个知识片段")
                    _sync_to_github()
                    st.session_state.refresh_local_files = True
                    st.rerun()
                else:
                    st.error(f"❌ 重建失败: {result}")


# ==========================================
# 工具函数
# ==========================================

def _get_backup_dir():
    """获取 RAG_backup 目录路径"""
    from config.settings import PATHS
    return PATHS.BACKUP_DIR


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
    """将文件永久备份到 tea_backup 目录"""
    from config.settings import PATHS
    backup_dir = PATHS.BACKUP_DIR
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
        logger.info(f"已备份 {len(backed_up)} 个文件到 RAG_backup")
    return backed_up


# ==========================================
# 元数据管理（文件 → chunk 映射）
# ==========================================

def _load_kb_metadata():
    """
    加载 kb_metadata.json
    格式: { "filename.pdf": {"chunk_start": 0, "chunk_count": 15}, ... }
    """
    from config.settings import PATHS
    meta_path = PATHS.kb_metadata
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载 kb_metadata 失败: {e}")
    return {}


def _save_kb_metadata(metadata):
    """保存 kb_metadata.json"""
    from config.settings import PATHS
    meta_path = PATHS.kb_metadata
    os.makedirs(meta_path.parent, exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _parse_file_to_chunks(file_path):
    """
    解析单个文件为 chunks 列表
    返回: list of str (chunk 文本)
    """
    text = ""
    try:
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
    except Exception as e:
        logger.warning(f"解析文件 {file_path.name} 失败: {e}")
        return []

    if not text.strip():
        return []

    # 文本分块
    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ==========================================
# 增量 Embedding 操作
# ==========================================

def _add_file_embeddings(file_path):
    """
    为单个文件生成 embedding 并追加到索引
    返回: (success: bool, chunk_count: int or error_msg: str)
    """
    from config.settings import PATHS
    from core.resource_manager import ResourceManager

    embedder = st.session_state.get('embedder')
    if not embedder:
        return False, "Embedding 服务未初始化"

    # 1. 解析文件
    new_chunks = _parse_file_to_chunks(file_path)
    if not new_chunks:
        return False, f"文件 {file_path.name} 未能提取到文本内容"

    # 2. 向量化
    try:
        batch_size = 25
        new_embeddings = []
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            batch_emb = embedder.embed_texts(batch)
            new_embeddings.extend(batch_emb)
    except Exception as e:
        return False, f"向量化失败: {str(e)}"

    new_embeddings_array = np.array(new_embeddings, dtype=np.float32)

    # 3. 加载或创建索引
    kb_idx, kb_chunks = st.session_state.kb
    if kb_idx is None or not isinstance(kb_chunks, list):
        kb_chunks = []

    # 如果索引为空，创建新的
    if kb_idx is None or kb_idx.ntotal == 0:
        dimension = new_embeddings_array.shape[1]
        kb_idx = faiss.IndexFlatL2(dimension)

    # 检查维度匹配
    if new_embeddings_array.shape[1] != kb_idx.d:
        # 维度不匹配，需要全量重建
        logger.warning(f"向量维度不匹配: 索引={kb_idx.d}, 新向量={new_embeddings_array.shape[1]}，触发全量重建")
        return False, "向量维度不匹配，请使用全量重建"

    # 4. 记录 chunk 起始位置和数量
    chunk_start = len(kb_chunks)
    chunk_count = len(new_chunks)

    # 5. 追加 chunks 和向量
    kb_chunks.extend(new_chunks)
    kb_idx.add(new_embeddings_array)

    # 6. 更新元数据
    metadata = _load_kb_metadata()
    metadata[file_path.name] = {
        "chunk_start": chunk_start,
        "chunk_count": chunk_count
    }
    _save_kb_metadata(metadata)

    # 7. 保存索引到文件
    os.makedirs(PATHS.kb_index.parent, exist_ok=True)
    faiss.write_index(kb_idx, str(PATHS.kb_index))
    with open(PATHS.kb_chunks, 'wb') as f:
        pickle.dump(kb_chunks, f)

    # 8. 更新 session_state
    st.session_state.kb = (kb_idx, kb_chunks)
    st.session_state.rag_loading_status = 'complete'
    st.session_state.rag_loading_needed = False

    # 9. 更新文件名列表
    saved_file_names = [f['name'] for f in _get_local_files()]
    ResourceManager.save_kb_files(saved_file_names)
    st.session_state.kb_files = saved_file_names

    return True, chunk_count


def _remove_file_embeddings(filename):
    """
    从索引中删除指定文件对应的 chunks 和向量
    返回: (success: bool, msg: str)
    """
    from config.settings import PATHS
    from core.resource_manager import ResourceManager

    metadata = _load_kb_metadata()

    if filename not in metadata:
        # 元数据中没有这个文件，需要全量重建
        logger.warning(f"元数据中找不到 {filename}，执行全量重建")
        ok, result = _rebuild_all_embeddings()
        return ok, f"全量重建完成（{result} 个片段）" if ok else f"全量重建失败: {result}"

    file_meta = metadata[filename]
    chunk_start = file_meta["chunk_start"]
    chunk_count = file_meta["chunk_count"]
    chunk_end = chunk_start + chunk_count

    kb_idx, kb_chunks = st.session_state.kb

    if kb_idx is None or not isinstance(kb_chunks, list):
        # 无索引可操作，直接返回
        del metadata[filename]
        _save_kb_metadata(metadata)
        return True, "索引为空"

    # 1. 删除 chunks
    new_chunks = kb_chunks[:chunk_start] + kb_chunks[chunk_end:]

    # 2. 重建 FAISS 索引（FAISS IndexFlatL2 不支持按位置删除，需要从剩余向量重建）
    if len(new_chunks) > 0 and kb_idx.ntotal > 0:
        # 从现有索引提取所有向量
        total_vectors = kb_idx.ntotal
        dimension = kb_idx.d

        all_vectors = np.zeros((total_vectors, dimension), dtype=np.float32)
        for i in range(total_vectors):
            all_vectors[i] = kb_idx.reconstruct(i)

        # 删除对应范围的向量
        keep_indices = list(range(0, chunk_start)) + list(range(chunk_end, total_vectors))
        if len(keep_indices) > 0:
            remaining_vectors = all_vectors[keep_indices]
            new_idx = faiss.IndexFlatL2(dimension)
            new_idx.add(remaining_vectors)
        else:
            new_idx = faiss.IndexFlatL2(dimension)
    else:
        new_idx = faiss.IndexFlatL2(kb_idx.d if kb_idx else 768)
        new_chunks = []

    # 3. 更新元数据：删除该文件，并调整后续文件的 chunk_start
    del metadata[filename]
    for fname, fmeta in metadata.items():
        if fmeta["chunk_start"] > chunk_start:
            fmeta["chunk_start"] -= chunk_count
    _save_kb_metadata(metadata)

    # 4. 保存索引
    os.makedirs(PATHS.kb_index.parent, exist_ok=True)
    if new_idx.ntotal > 0:
        faiss.write_index(new_idx, str(PATHS.kb_index))
    else:
        # 空索引，删除文件
        if PATHS.kb_index.exists():
            os.remove(PATHS.kb_index)

    with open(PATHS.kb_chunks, 'wb') as f:
        pickle.dump(new_chunks, f)

    # 5. 更新 session_state
    st.session_state.kb = (new_idx, new_chunks)
    st.session_state.rag_loading_status = 'complete'
    st.session_state.rag_loading_needed = False

    # 6. 更新文件名列表
    saved_file_names = [f['name'] for f in _get_local_files()]
    ResourceManager.save_kb_files(saved_file_names)
    st.session_state.kb_files = saved_file_names

    return True, f"已移除 {chunk_count} 个片段"


def _rebuild_all_embeddings():
    """
    全量重建：重新读取 RAG_DIR 中所有文件，全量重建 embedding 和 FAISS 索引。
    同时重建 kb_metadata.json。
    成功返回 (True, chunk_count)，失败返回 (False, error_msg)。
    """
    from config.settings import PATHS
    from core.resource_manager import ResourceManager

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
        _save_kb_metadata({})

        return True, 0

    # ===== 解析所有文件，同时记录元数据 =====
    all_chunks = []
    metadata = {}
    saved_file_names = []

    for file_path in rag_files:
        chunks = _parse_file_to_chunks(file_path)
        if chunks:
            chunk_start = len(all_chunks)
            all_chunks.extend(chunks)
            metadata[file_path.name] = {
                "chunk_start": chunk_start,
                "chunk_count": len(chunks)
            }
            saved_file_names.append(file_path.name)

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

    # 保存元数据
    _save_kb_metadata(metadata)

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
                        filename = file_info['name']

                        # 1. 删除 RAG_DIR 中的文件（不动 tea_backup）
                        os.remove(file_info['path'])

                        # 2. 增量删除：从索引中移除对应的 chunks 和向量
                        with st.spinner("🔄 正在更新知识库索引..."):
                            ok, result_msg = _remove_file_embeddings(filename)
                            if ok:
                                st.success(f"✅ 已删除 {filename}（{result_msg}）")
                            else:
                                st.warning(f"⚠️ 索引更新时出错: {result_msg}")

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
    """渲染「从备份恢复」区域"""
    from config.settings import PATHS

    backup_dir = PATHS.BACKUP_DIR
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


# ==========================================
# 核心处理逻辑
# ==========================================

def _handle_upload(files):
    """处理文件上传 —— 保存 + 备份 + 增量 embedding + GitHub 同步"""
    from config.settings import PATHS

    with st.spinner("🔄 正在处理文件，请稍候..."):
        try:
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            saved_paths = []
            total_chunks = 0

            for uploaded_file in files:
                # ===== 步骤 1：保存文件到 RAG_DIR =====
                file_path = PATHS.RAG_DIR / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)

                # ===== 步骤 2：永久备份到 RAG_backup =====
                _backup_files([file_path])

                # ===== 步骤 3：增量 embedding（仅处理当前文件） =====
                ok, result = _add_file_embeddings(file_path)
                if ok:
                    total_chunks += result
                    logger.info(f"✅ {uploaded_file.name}: {result} 个片段")
                else:
                    st.warning(f"⚠️ {uploaded_file.name} embedding 失败: {result}")

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
    st.success(f"📊 增量更新完成，新增 {total_chunks} 个知识片段")
    st.info("💡 知识库已更新，现在可以使用这些文件进行检索")
    st.rerun()


def _handle_reset_restore(selected_files):
    """处理从 tea_backup 恢复选中的文件到知识库（增量模式）"""
    from config.settings import PATHS

    with st.spinner("🔄 正在从备份恢复文件..."):
        try:
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            restored = []
            total_chunks = 0

            for bf in selected_files:
                # 1. 复制文件到 RAG_DIR
                dst = PATHS.RAG_DIR / bf['name']
                shutil.copy2(bf['path'], dst)
                restored.append(bf['name'])

                # 2. 增量 embedding
                ok, result = _add_file_embeddings(dst)
                if ok:
                    total_chunks += result
                else:
                    st.warning(f"⚠️ {bf['name']} embedding 失败: {result}")

            # 3. 同步到 GitHub
            _sync_to_github()

            # 4. 刷新文件列表
            st.session_state.refresh_local_files = True

        except Exception as e:
            st.error(f"❌ 恢复失败: {str(e)}")
            logger.error(f"从备份恢复失败: {e}", exc_info=True)
            return

    st.success(f"✅ 已恢复 {len(restored)} 个文件：{', '.join(restored)}")
    st.success(f"📊 增量更新完成，新增 {total_chunks} 个知识片段")
    st.rerun()
