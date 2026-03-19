"""
github_sync.py
===============
GitHub 同步工具模块
"""

import json
import time
import base64
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import streamlit as st
import requests
from github import Github, GithubException, Auth

logger = logging.getLogger(__name__)


class GithubSync:
    """负责将数据同步到 GitHub 仓库"""

    @staticmethod
    def _get_github_config():
        """获取 GitHub 配置"""
        token = st.secrets.get("GITHUB_TOKEN")
        repo_name = st.secrets.get("GITHUB_REPO")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        return token, repo_name, branch

    @staticmethod
    def _get_github_client():
        """获取 GitHub 客户端"""
        token, repo_name, branch = GithubSync._get_github_config()
        if not token or not repo_name:
            return None, None, None
        g = Github(auth=Auth.Token(token))
        return g, repo_name, branch

    # ---------- 通用推送 / 删除 ----------
    @staticmethod
    def push_json(file_path_in_repo: str, data_dict, commit_msg: str = "Update via Streamlit") -> bool:
        """
        推送 JSON 文件到 GitHub

        Args:
            file_path_in_repo: 仓库中的文件路径
            data_dict: 数据字典
            commit_msg: 提交消息

        Returns:
            bool: 是否成功
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名 (GITHUB_TOKEN / GITHUB_REPO)")
            return False
        try:
            repo = g.get_repo(repo_name)
            content_str = json.dumps(data_dict, ensure_ascii=False, indent=2)
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.update_file(path=contents.path, message=commit_msg, content=content_str, sha=contents.sha, branch=branch)
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(path=file_path_in_repo, message=f"Create {file_path_in_repo}", content=content_str, branch=branch)
                else:
                    raise e
            return True
        except Exception as e:
            st.error(f"Github 同步失败: {str(e)}")
            return False

    @staticmethod
    def push_binary_file(file_path_in_repo: str, file_content: bytes, commit_msg: str = "Upload file") -> bool:
        """
        推送二进制文件到 GitHub

        Args:
            file_path_in_repo: 仓库中的文件路径
            file_content: 文件内容（字节）
            commit_msg: 提交消息

        Returns:
            bool: 是否成功
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名")
            return False
        try:
            repo = g.get_repo(repo_name)
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.update_file(path=contents.path, message=commit_msg, content=file_content, sha=contents.sha, branch=branch)
            except GithubException as e:
                if e.status == 404:
                    repo.create_file(path=file_path_in_repo, message=f"Create {file_path_in_repo}", content=file_content, branch=branch)
                else:
                    raise e
            return True
        except Exception as e:
            st.error(f"Github 文件上传失败: {str(e)}")
            return False

    @staticmethod
    def delete_file(file_path_in_repo: str, commit_msg: str = "Delete file") -> bool:
        """
        从 GitHub 删除文件

        Args:
            file_path_in_repo: 仓库中的文件路径
            commit_msg: 提交消息

        Returns:
            bool: 是否成功
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            return False
        try:
            repo = g.get_repo(repo_name)
            try:
                contents = repo.get_contents(file_path_in_repo, ref=branch)
                repo.delete_file(path=contents.path, message=commit_msg, sha=contents.sha, branch=branch)
                return True
            except GithubException as e:
                if e.status == 404:
                    return True
                raise e
        except Exception as e:
            st.error(f"Github 删除文件失败: {str(e)}")
            return False

    # ---------- 判例库同步（基础 / 进阶） ----------
    @staticmethod
    def sync_basic_cases(cases: List[Dict]) -> bool:
        """
        同步基础判例到 GitHub

        Args:
            cases: 基础判例列表

        Returns:
            bool: 是否成功
        """
        return GithubSync.push_json("tea_data/basic_case.json", cases, "Update basic_case.json from App")

    @staticmethod
    def sync_supp_cases(cases: List[Dict]) -> bool:
        """
        同步进阶判例到 GitHub

        Args:
            cases: 进阶判例列表

        Returns:
            bool: 是否成功
        """
        return GithubSync.push_json("tea_data/supplementary_case.json", cases, "Update supplementary_case.json from App")

    # ---------- RAG 文件管理 ----------
    @staticmethod
    def backup_rag_file(file_content: bytes, filename: str, backup_folder: str = "tea_backup") -> bool:
        """
        备份 RAG 文件到 GitHub

        Args:
            file_content: 文件内容
            filename: 文件名
            backup_folder: 备份文件夹

        Returns:
            bool: 是否成功
        """
        file_path = f"{backup_folder}/{filename}"
        try:
            result = GithubSync.push_binary_file(file_path, file_content, f"Backup RAG file: {filename}")
            if result:
                logger.info(f" ✅ 已备份到 {file_path}")
            return result
        except Exception as e:
            logger.warning(f" 备份文件 {filename} 到 {backup_folder} 失败: {e}")
            return False

    @staticmethod
    def add_rag_files(uploaded_files: List, rag_folder: str = "tea_data/RAG") -> Tuple[bool, List[str]]:
        """
        添加 RAG 文件到 GitHub

        Args:
            uploaded_files: 上传的文件列表
            rag_folder: RAG 文件夹路径

        Returns:
            Tuple[bool, List[str]]: (是否成功, 上传成功的文件名列表)
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            st.error("❌ 未配置 Github Token 或 仓库名")
            return False, []
        try:
            uploaded_names = []
            for uf in uploaded_files:
                file_path = f"{rag_folder}/{uf.name}"
                uf.seek(0)
                file_content = uf.read()
                if GithubSync.push_binary_file(file_path, file_content, f"Add RAG file: {uf.name}"):
                    uploaded_names.append(uf.name)
                    GithubSync.backup_rag_file(file_content, uf.name, backup_folder="tea_backup")
                else:
                    st.warning(f"⚠️ 上传 {uf.name} 失败")
            return len(uploaded_names) > 0, uploaded_names
        except Exception as e:
            st.error(f"RAG文件添加失败: {str(e)}")
            return False, []

    @staticmethod
    def list_rag_files(rag_folder: str = "tea_data/RAG") -> List[str]:
        """
        列出 GitHub 上的 RAG 文件

        Args:
            rag_folder: RAG 文件夹路径

        Returns:
            List[str]: 文件名列表
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            return []
        try:
            repo = g.get_repo(repo_name)
            contents = repo.get_contents(rag_folder, ref=branch)
            return [c.name for c in contents if c.type == "file"]
        except GithubException as e:
            if e.status == 404:
                return []
            logger.error(f" 获取RAG文件列表失败: {e}")
            return []
        except Exception as e:
            logger.error(f" 获取RAG文件列表失败: {e}")
            return []

    @staticmethod
    def delete_rag_file(filename: str, rag_folder: str = "tea_data/RAG") -> bool:
        """
        删除 GitHub 上的 RAG 文件

        Args:
            filename: 文件名
            rag_folder: RAG 文件夹路径

        Returns:
            bool: 是否成功
        """
        file_path = f"{rag_folder}/{filename}"
        return GithubSync.delete_file(file_path, f"Delete RAG file: {filename}")

    @staticmethod
    def pull_rag_folder(rag_folder: str = "tea_data/RAG") -> List[Tuple[str, bytes]]:
        """
        从 GitHub 拉取 RAG 文件夹中的所有文件

        Args:
            rag_folder: RAG 文件夹路径

        Returns:
            List[Tuple[str, bytes]]: (文件名, 文件内容) 列表
        """
        token, repo_name, branch = GithubSync._get_github_config()
        if not token or not repo_name:
            logger.warning(" GitHub config not found, skip pulling RAG")
            return []

        def download_with_retry(url, headers, max_retries=3):
            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.get(url, headers=headers, timeout=180, stream=True)
                    if response.status_code == 200:
                        content = b''
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                content += chunk
                        return content, True
                    else:
                        logger.warning(f"     尝试 {attempt}/{max_retries}: HTTP {response.status_code}")
                except Exception as e:
                    logger.warning(f"     尝试 {attempt}/{max_retries}: {e}")
                    if attempt < max_retries:
                        time.sleep(2)
            return None, False

        try:
            g = Github(auth=Auth.Token(token))
            repo = g.get_repo(repo_name)
            files = []
            logger.info(f" ========== 开始从 GitHub 拉取 RAG 文件 ==========")

            try:
                contents = repo.get_contents(rag_folder, ref=branch)
                file_list = [c for c in contents if c.type == "file"]
                logger.info(f" 发现 {len(file_list)} 个文件")

                for idx, content in enumerate(file_list, 1):
                    logger.info(f" [{idx}/{len(file_list)}] 正在处理: {content.name}")
                    file_content = None
                    download_method = None

                    # 方法1：Raw URL
                    raw_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{rag_folder}/{content.name}"
                    headers = {"Authorization": f"Bearer {token}"}
                    file_content, success = download_with_retry(raw_url, headers, max_retries=3)
                    if success and file_content:
                        download_method = "Raw URL"

                    # 方法2：Git Blob（小于1MB）
                    if file_content is None and content.size < 1024 * 1024:
                        try:
                            blob = repo.get_git_blob(content.sha)
                            if blob.encoding == "base64":
                                file_content = base64.b64decode(blob.content)
                                download_method = "Git Blob"
                        except Exception as e:
                            logger.warning(f"   Git Blob 失败: {e}")

                    # 方法3：Download URL
                    if file_content is None and content.download_url:
                        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3.raw"}
                        file_content, success = download_with_retry(content.download_url, headers, max_retries=3)
                        if success:
                            download_method = "Download URL"

                    # 验证完整性
                    if file_content:
                        if len(file_content) == content.size:
                            files.append((content.name, file_content))
                            logger.info(f"   ✅ {content.name} 验证通过 ({download_method})")
                        else:
                            logger.error(f"  ❌ {content.name} 大小不匹配")
                    else:
                        logger.error(f"  ❌ {content.name} 所有下载方法均失败")

            except GithubException as e:
                if e.status == 404:
                    return []
                raise e

            logger.info(f" ========== RAG 拉取完成: {len(files)}/{len(file_list)} ==========\n")
            return files

        except Exception as e:
            logger.error(f" 拉取 RAG 文件夹失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def download_github_file(file_path_in_repo: str) -> Optional[bytes]:
        """
        从 GitHub 下载单个文件的内容

        Args:
            file_path_in_repo: 仓库中的文件路径

        Returns:
            Optional[bytes]: 文件内容，失败返回 None
        """
        token, repo_name, branch = GithubSync._get_github_config()
        if not token or not repo_name:
            return None
        try:
            g = Github(auth=Auth.Token(token))
            repo = g.get_repo(repo_name)
            content = repo.get_contents(file_path_in_repo, ref=branch)
            if content.encoding == "base64" and content.content:
                return base64.b64decode(content.content)
            # 大文件走 Raw URL
            raw_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{file_path_in_repo}"
            resp = requests.get(raw_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
            if resp.status_code == 200:
                return resp.content
        except Exception as e:
            logger.warning(f" 下载 {file_path_in_repo} 失败: {e}")
        return None

    # ---------- 配置状态与通用同步 ----------
    @staticmethod
    def check_config() -> Tuple[bool, str]:
        """
        检查 GitHub 配置状态

        Returns:
            Tuple[bool, str]: (是否配置, 状态消息)
        """
        token, repo_name, branch = GithubSync._get_github_config()
        if not token:
            return False, "未配置 GITHUB_TOKEN"
        if not repo_name:
            return False, "未配置 GITHUB_REPO"
        if not branch:
            return False, "未配置 GITHUB_BRANCH"

        try:
            # 添加超时参数（3秒超时，避免长时间等待）
            g = Github(auth=Auth.Token(token), timeout=3)
            repo = g.get_repo(repo_name)
            return True, f"已连接: {repo_name}"
        except Exception as e:
            return False, f"连接失败: {str(e)}"

    @staticmethod
    def get_repo_info() -> Optional[Dict]:
        """
        获取仓库信息

        Returns:
            Optional[Dict]: 仓库信息字典
        """
        g, repo_name, branch = GithubSync._get_github_client()
        if not g or not repo_name:
            return None
        try:
            repo = g.get_repo(repo_name)
            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "private": repo.private,
                "url": repo.html_url,
                "branch": branch
            }
        except Exception as e:
            logger.error(f" 获取仓库信息失败: {e}")
            return None
