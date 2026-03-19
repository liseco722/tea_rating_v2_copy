"""
tab3_knowledge.py
==================
知识库设计 Tab - 本地模式（增量索引版）
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from config.settings import PATHS
from core.resource_manager import ResourceManager, DEFAULT_EMBEDDING_DIM
from core.github_sync import GithubSync

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {'.pdf', '.txt', '.docx'}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ==========================================
# 公共入口
# ==========================================

def render_tab3():
    """渲染知识库设计 Tab - 增量索引模式"""
    with st.container():
        st.markdown("##### 📁 本地知识库文件")
        _render_local_file_list()

        st.markdown("---")

        st.markdown("##### ➕ 添加新文件")
        _render_upload_section()

        st.markdown("---")

        st.markdown("##### 🛡️ 安全冗余重建")
        _render_safety_rebuild_section()

# ==========================================
# 对外暴露：供 sidebar 调用的统一重建入口
# ==========================================

def rebuild_rag_cache() -> Tuple[bool, str]:
    """公开的 RAG 安全重建入口。"""
    ok, result = _rebuild_all_embeddings()
    if ok:
        return True, f"知识库索引已重建，共 {result} 个知识片段"
    return False, str(result)


# ==========================================
# 工具函数
# ==========================================

def _get_backup_dir() -> Path:
    """RAG 备份目录：tea_data/RAG_backup。"""
    return PATHS.BACKUP_DIR



def _empty_kb_metadata() -> Dict:
    return {
        "version": 2,
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "vector_dim": DEFAULT_EMBEDDING_DIM,
        "files": {},
        "updated_at": ""
    }



def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")



def _get_local_files() -> List[Dict]:
    """获取本地 RAG 目录中的文件列表。"""
    if not PATHS.RAG_DIR.exists():
        return []

    files = []
    for file_path in PATHS.RAG_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append({
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'path': file_path
            })

    return sorted(files, key=lambda x: x['name'])



def _backup_files(file_paths: List[Path]) -> List[str]:
    """将文件永久备份到 tea_data/RAG_backup。"""
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
        logger.info(f"已备份 {len(backed_up)} 个文件到 {backup_dir}")
    return backed_up



def _load_kb_state() -> Tuple[List[str], List[List[float]], Dict]:
    """加载 chunks / vectors / metadata。"""
    chunks = ResourceManager.load_pickle(PATHS.kb_chunks, fallback=[])
    vectors = ResourceManager.load_kb_vectors()
    metadata = ResourceManager.load_kb_metadata()

    if not isinstance(chunks, list):
        chunks = []
    if not isinstance(vectors, list):
        vectors = []
    if not isinstance(metadata, dict):
        metadata = _empty_kb_metadata()

    metadata.setdefault("files", {})
    metadata.setdefault("version", 2)
    metadata.setdefault("chunk_size", CHUNK_SIZE)
    metadata.setdefault("overlap", CHUNK_OVERLAP)
    metadata.setdefault("vector_dim", DEFAULT_EMBEDDING_DIM)

    return chunks, vectors, metadata



def _serialize_blocks(file_blocks: List[Dict]) -> Tuple[List[str], List[List[float]], Dict]:
    """把分文件 block 重新序列化为总 chunks/vectors/metadata。"""
    all_chunks: List[str] = []
    all_vectors: List[List[float]] = []
    metadata = _empty_kb_metadata()

    cursor = 0
    for block in file_blocks:
        name = block['name']
        chunks = block.get('chunks', [])
        vectors = block.get('vectors', [])
        size = int(block.get('size', 0))

        if len(chunks) != len(vectors):
            raise ValueError(f"文件 {name} 的 chunks / vectors 数量不一致")

        normalized_vectors = []
        for vector in vectors:
            nv = ResourceManager._normalize_vector(vector, DEFAULT_EMBEDDING_DIM)
            if nv is None:
                raise ValueError(f"文件 {name} 存在非法向量缓存")
            normalized_vectors.append(nv)

        start = cursor
        end = cursor + len(chunks)

        all_chunks.extend(chunks)
        all_vectors.extend(normalized_vectors)
        metadata['files'][name] = {
            'start': start,
            'end': end,
            'chunk_count': len(chunks),
            'size': size,
        }
        cursor = end

    metadata['updated_at'] = _now_str()
    return all_chunks, all_vectors, metadata



def _load_file_blocks_from_cache() -> List[Dict] | None:
    """从缓存中恢复分文件 block；若缓存不可靠则返回 None。"""
    chunks, vectors, metadata = _load_kb_state()
    files_meta = metadata.get('files', {})

    if not files_meta:
        return [] if not chunks and not vectors else None

    if len(chunks) != len(vectors):
        return None

    file_blocks: List[Dict] = []
    max_end = 0
    for filename, info in files_meta.items():
        start = int(info.get('start', 0))
        end = int(info.get('end', 0))
        if start < 0 or end < start or end > len(chunks):
            return None
        max_end = max(max_end, end)
        file_blocks.append({
            'name': filename,
            'size': int(info.get('size', 0)),
            'chunks': chunks[start:end],
            'vectors': vectors[start:end],
        })

    if max_end != len(chunks):
        return None

    return file_blocks



def _state_matches_local_files(metadata: Dict) -> bool:
    local_names = {item['name'] for item in _get_local_files()}
    cached_names = set(metadata.get('files', {}).keys())
    return local_names == cached_names



def _extract_text_from_file(file_path: Path) -> str:
    text = ""
    suffix = file_path.suffix.lower()

    if suffix == '.txt':
        text = file_path.read_text(encoding='utf-8', errors='ignore')
    elif suffix == '.pdf':
        import PyPDF2
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif suffix == '.docx':
        import docx
        doc = docx.Document(str(file_path))
        text = "\n".join([para.text for para in doc.paragraphs])

    return text.strip()



def _chunk_text(text: str) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks



def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    embedder = st.session_state.get('embedder')
    if not embedder:
        raise RuntimeError("Embedding 服务未初始化")

    if not chunks:
        return []

    vectors: List[List[float]] = []
    batch_size = 25
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_vectors = embedder.embed_texts(batch)
        for vector in batch_vectors:
            normalized = ResourceManager._normalize_vector(vector, DEFAULT_EMBEDDING_DIM)
            if normalized is None:
                raise RuntimeError("Embedding 服务返回了异常维度的向量")
            vectors.append(normalized)
    return vectors



def _parse_file_to_block(file_path: Path) -> Dict:
    text = _extract_text_from_file(file_path)
    chunks = _chunk_text(text)
    vectors = _embed_chunks(chunks) if chunks else []

    return {
        'name': file_path.name,
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'chunks': chunks,
        'vectors': vectors,
    }



def _to_github_path(local_path: Path) -> str:
    """将本地 Path 转换为 GitHub 仓库路径，如 'tea_data/xxx'。"""
    s = str(local_path).replace('\\', '/')
    idx = s.find('tea_data/')
    if idx >= 0:
        return s[idx:]
    return f"tea_data/{local_path.name}"


def _push_local_file_to_github(local_path: Path, commit_msg: str = None):
    """将本地文件推送到 GitHub 对应路径（静默失败，仅记录日志）。"""
    if not local_path.exists():
        return
    try:
        github_path = _to_github_path(local_path)
        if commit_msg is None:
            commit_msg = f"Update {github_path}"
        with open(local_path, 'rb') as f:
            content = f.read()
        ok = GithubSync.push_binary_file(github_path, content, commit_msg)
        if ok:
            logger.info(f"已同步 {github_path} 到 GitHub")
        else:
            logger.warning(f"同步 {github_path} 到 GitHub 失败")
    except Exception as e:
        logger.warning(f"推送 {local_path.name} 到 GitHub 异常: {e}")


def _save_kb_state(chunks: List[str], vectors: List[List[float]], metadata: Dict):
    """保存 KB 缓存并刷新 session_state，同时同步 index 到 GitHub。"""
    ResourceManager.save_kb_metadata(metadata)
    ResourceManager.save_kb_vectors(vectors)
    ResourceManager.save_kb_files(list(metadata.get('files', {}).keys()))

    index = ResourceManager.build_index_from_vectors(vectors, DEFAULT_EMBEDDING_DIM)
    ResourceManager.save(index, chunks, PATHS.kb_index, PATHS.kb_chunks)

    st.session_state.kb = (index, chunks)
    st.session_state.kb_files = list(metadata.get('files', {}).keys())
    st.session_state.rag_loading_status = 'complete'
    st.session_state.rag_loading_needed = False

    # 同步关键文件到 GitHub tea_data/
    for path in [PATHS.kb_index, PATHS.kb_chunks, PATHS.kb_metadata, PATHS.kb_files]:
        _push_local_file_to_github(path, f"Update KB: {path.name}")



def _clear_kb_state():
    for path in [PATHS.kb_index, PATHS.kb_chunks, PATHS.kb_metadata, PATHS.kb_vectors, PATHS.kb_files]:
        try:
            # 删本地
            if path.exists():
                path.unlink()
            # 删 GitHub
            github_path = _to_github_path(path)
            GithubSync.delete_file(github_path, f"Delete KB: {path.name}")
        except Exception as e:
            logger.warning(f"删除缓存文件失败 ({path}): {e}")

    st.session_state.kb = (ResourceManager.empty_faiss_index(DEFAULT_EMBEDDING_DIM), [])
    st.session_state.kb_files = []
    st.session_state.rag_loading_status = 'empty'
    st.session_state.rag_loading_needed = False



def _rebuild_all_embeddings() -> Tuple[bool, int | str]:
    """重新读取 RAG_DIR 中所有文件，全量重建 embedding 和 FAISS 索引。"""
    if not PATHS.RAG_DIR.exists():
        return False, "RAG 目录不存在"

    rag_files = [
        f for f in sorted(PATHS.RAG_DIR.iterdir(), key=lambda p: p.name)
        if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    if not rag_files:
        _clear_kb_state()
        return True, 0

    file_blocks: List[Dict] = []
    total_chunks = 0
    for file_path in rag_files:
        try:
            block = _parse_file_to_block(file_path)
            file_blocks.append(block)
            total_chunks += len(block['chunks'])
        except Exception as e:
            logger.warning(f"解析文件 {file_path.name} 失败: {e}")
            return False, f"解析 / 向量化失败：{file_path.name} - {e}"

    chunks, vectors, metadata = _serialize_blocks(file_blocks)
    _save_kb_state(chunks, vectors, metadata)
    return True, total_chunks



def _upsert_files_into_kb(file_paths: List[Path]) -> Tuple[bool, int | str]:
    """增量添加 / 替换文件到知识库。

    若当前缓存不可安全增量更新，则自动退化为全量重建（仍只发生一次）。
    """
    if not file_paths:
        return True, 0

    file_blocks = _load_file_blocks_from_cache()
    _, _, metadata = _load_kb_state()
    cached_names = set(metadata.get('files', {}).keys())
    local_names = {item['name'] for item in _get_local_files()}
    incoming_names = {path.name for path in file_paths}

    # 允许的差异：仅来自当前这批新增 / 替换文件
    safe_incremental = (
        file_blocks is not None
        and (local_names - cached_names).issubset(incoming_names)
        and (cached_names - local_names).issubset(incoming_names)
    )

    if not safe_incremental:
        logger.info("检测到旧缓存不可安全增量更新，退化为全量重建")
        return _rebuild_all_embeddings()

    block_map = {block['name']: block for block in file_blocks}
    block_order = [block['name'] for block in file_blocks]

    added_chunks = 0
    for file_path in file_paths:
        block = _parse_file_to_block(file_path)
        block_map[file_path.name] = block
        if file_path.name not in block_order:
            block_order.append(file_path.name)
        added_chunks += len(block['chunks'])

    new_blocks = [block_map[name] for name in block_order if name in block_map]
    chunks, vectors, metadata = _serialize_blocks(new_blocks)
    _save_kb_state(chunks, vectors, metadata)
    return True, added_chunks



def _remove_file_from_kb(filename: str) -> Tuple[bool, str]:
    """从缓存中删除某个文件对应的 chunks / vectors，并重建 index。"""
    file_blocks = _load_file_blocks_from_cache()
    _, _, metadata = _load_kb_state()
    cached_names = set(metadata.get('files', {}).keys())
    local_names = {item['name'] for item in _get_local_files()}

    # 允许的差异：仅删除了当前这个 filename
    safe_incremental = (
        file_blocks is not None
        and (cached_names - local_names).issubset({filename})
        and not (local_names - cached_names)
    )

    if not safe_incremental:
        logger.info("删除文件时检测到缓存不可安全增量更新，退化为全量重建")
        ok, result = _rebuild_all_embeddings()
        return ok, str(result)

    new_blocks = [block for block in file_blocks if block['name'] != filename]
    chunks, vectors, metadata = _serialize_blocks(new_blocks)
    _save_kb_state(chunks, vectors, metadata)
    return True, f"已删除 {filename} 对应的 chunks 和索引"


# ==========================================
# UI 渲染
# ==========================================

def _render_local_file_list():
    """渲染本地文件列表。"""
    if 'refresh_local_files' in st.session_state and st.session_state.refresh_local_files:
        st.session_state.local_rag_files = _get_local_files()
        st.session_state.refresh_local_files = False

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

    st.markdown(f"""
    <div style="padding: 10px; background: #EDF5EB; border-radius: 6px; margin-bottom: 15px;">
        <span style="color: #4A5D53; font-weight: 600;">📊 共 {len(local_files)} 个文件</span>
    </div>
    """, unsafe_allow_html=True)

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
                        # 1. 删除本地文件
                        if file_info['path'].exists():
                            os.remove(file_info['path'])

                        # 2. 从 GitHub tea_data/RAG 删除
                        gh_deleted = GithubSync.delete_rag_file(file_info['name'])
                        if not gh_deleted:
                            logger.warning(f"GitHub 删除 {file_info['name']} 失败或未配置")

                        # 3. 删除对应 chunks 并更新 index
                        ok, msg = _remove_file_from_kb(file_info['name'])
                        if not ok:
                            st.error(f"❌ 删除后更新索引失败: {msg}")
                            return

                        st.session_state.refresh_local_files = True
                        gh_msg = "，已同步删除 GitHub 端文件" if gh_deleted else ""
                        st.success(f"✅ 已删除 {file_info['name']}，并更新对应 chunks / index{gh_msg}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败: {str(e)}")



def _render_upload_section():
    """渲染上传区域。"""
    st.caption("支持格式：PDF、TXT、DOCX")

    up = st.file_uploader(
        "选择文件",
        accept_multiple_files=True,
        key="kb_uploader",
        type=['pdf', 'txt', 'docx'],
        help="可一次选择多个文件"
    )

    if up:
        st.markdown(f"""
        <div style="padding: 10px; background: #FDF6ED; border-radius: 6px; margin: 10px 0;">
            <span style="color: #8B5A2B; font-weight: 600;">📋 已选择 {len(up)} 个文件</span>
        </div>
        """, unsafe_allow_html=True)

        for file in up:
            st.markdown(f"- {file.name} ({file.size / 1024:.1f} KB)")

    if st.button("📤 添加到知识库", type="primary", width='stretch'):
        if not up:
            st.warning("⚠️ 请先选择要上传的文件")
        else:
            _handle_upload(up)



def _render_safety_rebuild_section():
    """渲染安全重建区域。"""
    metadata = ResourceManager.load_kb_metadata()
    local_files = _get_local_files()
    local_names = {item['name'] for item in local_files}
    cached_names = set(metadata.get('files', {}).keys())
    mismatch = local_names != cached_names

    if mismatch:
        st.warning("⚠️ 检测到 tea_data/RAG 与当前网页缓存不一致。")
        st.caption("点击下方按钮可重新解析全部文件、重新 embedding，并重建 chunks / index。")
    else:
        st.info("当前缓存与 tea_data/RAG 文件列表一致。若怀疑缓存损坏，仍可手动执行安全重建。")

    if st.button("🛡️ 安全重建知识库索引", width='stretch'):
        with st.spinner("🔄 正在重新解析文件并重建索引..."):
            ok, result = _rebuild_all_embeddings()
            if ok:
                st.session_state.refresh_local_files = True
                st.success(f"✅ 安全重建完成，共 {result} 个知识片段")
                st.rerun()
            else:
                st.error(f"❌ 重建失败: {result}")


# ==========================================
# 核心处理逻辑
# ==========================================

def _handle_upload(files):
    """处理文件上传：保存本地 + 推送 GitHub + 备份 + 增量 embedding + 刷新 index。"""
    with st.spinner("🔄 正在处理文件，请稍候..."):
        try:
            os.makedirs(PATHS.RAG_DIR, exist_ok=True)
            os.makedirs(PATHS.BACKUP_DIR, exist_ok=True)

            saved_paths: List[Path] = []
            github_uploaded: List[str] = []
            github_backed_up: List[str] = []

            for uploaded_file in files:
                # 1. 保存到本地 tea_data/RAG
                file_path = PATHS.RAG_DIR / uploaded_file.name
                uploaded_file.seek(0)
                file_content = uploaded_file.read()
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                saved_paths.append(file_path)

                # 2. 推送到 GitHub tea_data/RAG
                if GithubSync.push_binary_file(
                    f"tea_data/RAG/{uploaded_file.name}",
                    file_content,
                    f"Add RAG file: {uploaded_file.name}"
                ):
                    github_uploaded.append(uploaded_file.name)
                else:
                    logger.warning(f"GitHub 上传 {uploaded_file.name} 到 tea_data/RAG 失败")

                # 3. 推送到 GitHub tea_backup（备份）
                if GithubSync.backup_rag_file(file_content, uploaded_file.name):
                    github_backed_up.append(uploaded_file.name)
                else:
                    logger.warning(f"GitHub 备份 {uploaded_file.name} 到 tea_backup 失败")

            # 4. 本地备份
            backed_up = _backup_files(saved_paths)

            # 5. 增量 embedding + 更新 index
            ok, result = _upsert_files_into_kb(saved_paths)
            if not ok:
                st.error(f"❌ 更新知识库索引失败: {result}")
                return
            st.session_state.refresh_local_files = True

        except Exception as e:
            st.error("❌ 处理失败")
            st.error(f"错误信息: {str(e)}")
            logger.error(f"知识库上传处理失败: {e}", exc_info=True)
            return

    st.success(f"✅ 成功添加 {len(saved_paths)} 个文件到知识库")
    st.success(f"📊 索引已更新，本次新增 / 替换共影响 {result} 个知识片段")
    if github_uploaded:
        st.info(f"☁️ 已同步到 GitHub tea_data/RAG：{', '.join(github_uploaded)}")
    if github_backed_up:
        st.info(f"💾 已备份到 GitHub tea_backup：{', '.join(github_backed_up)}")
    if backed_up:
        st.info(f"📂 已本地备份到 RAG_backup：{', '.join(backed_up)}")
    st.rerun()
