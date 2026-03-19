"""
dialogs.py
===========
各种弹窗 UI 组件 - 升级版
"""

import streamlit as st
import time
import numpy as np
import faiss
from config.constants import TEA_EXAMPLES, FACTORS
from config.settings import PATHS
from core.resource_manager import ResourceManager


# ==========================================
# 提示词查看弹窗
# ==========================================

@st.dialog("📝 本次发送给 LLM 的完整 Prompt", width="large")
def show_prompt_dialog():
    """弹窗展示发送给 LLM 的系统提示词和用户提示词"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #F5F9F5 0%, #EDF5EB 100%); border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A5D53;">
        <span style="color: #4A5D53; font-size: 1.1em; font-weight: 600;">🔍 提示词详情</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔧 System Prompt（系统提示词）：**")
        system_prompt = st.session_state.get('last_llm_sys_prompt', '（暂无）')
        st.code(system_prompt, language=None, height=400)

    with col2:
        st.markdown("**💬 User Prompt（用户提示词）：**")
        user_prompt = st.session_state.get('last_llm_user_prompt', '（暂无）')
        st.code(user_prompt, language=None, height=400)




# ==========================================
# 茶评示例弹窗
# ==========================================

@st.dialog("🍵 茶评示例", width="large")
def show_tea_examples_dialog():
    """展示预置茶评示例文本 - 参考 app.py 的简洁实现"""

    # 副标题区域
    st.info("📜 品鉴案例精选")

    # 提示信息（参考 app.py）
    st.caption("💡 以下是五组茶评示例，点击文本框即可选中复制，粘贴到「交互评分」中使用")

    # 从 session_state 加载示例
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES)

    st.divider()

    # 遍历展示每个示例（平铺展示，不用 expander）
    for i, ex in enumerate(examples):
        # 标题
        st.markdown(f"**{ex['title']}**")

        # 内容显示（使用 st.code()，方便选择复制）
        st.code(ex["text"], language=None)

        # 分隔线（最后一个示例后不加）
        if i < len(examples) - 1:
            st.markdown("")


# ==========================================
# 基础判例弹窗
# ==========================================

@st.dialog("📋 基础判例列表", width="large")
def show_basic_cases_dialog(embedder):
    """展示当前基础判例列表 - 升级版（支持勾选转移到进阶判例）"""
    cases = st.session_state.basic_cases

    if not cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">暂无基础判例</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 初始化勾选状态存储
    if 'basic_case_checkboxes' not in st.session_state:
        st.session_state.basic_case_checkboxes = {}

    st.markdown(f"**📊 共 {len(cases)} 条基础判例**")
    st.markdown("---")

    for idx, case in enumerate(cases):
        with st.container():
            # 使用两列布局：左侧勾选框，右侧expander
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                # 勾选框
                current_value = st.session_state.basic_case_checkboxes.get(idx, False)
                checked = st.checkbox("", key=f"basic_check_{idx}",
                                     value=current_value,
                                     label_visibility="collapsed")
                st.session_state.basic_case_checkboxes[idx] = checked
            with col2:
                with st.expander(f"📋 判例 {idx + 1}", expanded=False):
                    # 1. 判例描述
                    st.markdown("**📝 判例描述：**")
                    st.markdown(f"> {case.get('text', '')}")

                    # 2. 六因子评分详情
                    st.markdown("---")
                    st.markdown("**🏷️ 因子评分详情：**")

                    scores = case.get('scores', {})

                    # 分两列展示因子
                    fc1, fc2 = st.columns(2)

                    for i, (factor, data) in enumerate(scores.items()):
                        with (fc1 if i % 2 == 0 else fc2):
                            with st.container(border=True):
                                st.markdown(f"**{factor}**")
                                score_val = data.get('score', '-')
                                comment = data.get('comment', '暂无评语')
                                suggestion = data.get('suggestion', '暂无建议')

                                col_s, col_d = st.columns([1, 2])
                                with col_s:
                                    st.markdown(f"<span style='font-size:12px; color:#666;'>分数: </span><span style='font-size:14px; font-weight:bold; color:#2E7D32;'>{score_val}/9</span>", unsafe_allow_html=True)
                                with col_d:
                                    if comment and comment != '暂无评语':
                                        st.caption(f"💬 {comment}")
                                    if suggestion and suggestion != '暂无建议':
                                        st.caption(f"💡 {suggestion}")

                    # 3. 宗师总评
                    st.markdown("---")
                    st.markdown("**🍵 宗师总评：**")
                    st.info(case.get('master_comment', '暂无'))

                    # 4. 编辑按钮
                    st.markdown("---")
                    if st.button("✏️ 编辑此判例", key=f"edit_basic_{idx}", width='stretch'):
                        st.session_state.editing_basic_idx = idx
                        st.rerun()

    st.markdown("---")

    # 显示已选数量和操作按钮
    selected_count = sum(1 for v in st.session_state.basic_case_checkboxes.values() if v)
    st.markdown(f"已选择: **{selected_count}** 条")

    col_transfer, col_close = st.columns([1, 1])

    with col_transfer:
        if st.button("➡️ 转移到进阶判例", type="primary",
                     disabled=selected_count == 0,
                     width='stretch'):
            _transfer_basic_to_supp(embedder)

    with col_close:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


def _transfer_basic_to_supp(embedder):
    """将选中的基础判例转移到进阶判例。"""
    selected_indices = sorted([
        idx for idx, checked in st.session_state.basic_case_checkboxes.items()
        if checked
    ])

    if not selected_indices:
        return

    selected_cases = []
    for idx in reversed(selected_indices):
        if idx < len(st.session_state.basic_cases):
            case = dict(st.session_state.basic_cases.pop(idx))
            selected_cases.append(case)
    selected_cases.reverse()

    try:
        # 先保存基础判例
        st.session_state.basic_cases = [
            ResourceManager.strip_case_vector(case)
            for case in st.session_state.basic_cases
        ]
        ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)

        # 再追加到进阶判例并更新索引
        _, supp_data = st.session_state.supp_cases
        for case in selected_cases:
            ResourceManager.ensure_case_embedding(case, embedder)
            supp_data.append(case)

        new_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
        st.session_state.supp_cases = (new_idx, supp_data)
        st.session_state.basic_case_checkboxes = {}

        st.success(f"✅ 已成功转移 {len(selected_cases)} 条到进阶判例！")
        time.sleep(0.3)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 转移失败: {e}")


@st.dialog("✏️ 编辑基础判例", width="large")
def edit_basic_case_dialog(idx: int):
    """编辑指定基础判例 - 完整表单编辑"""
    cases = st.session_state.basic_cases

    if idx >= len(cases):
        st.error("❌ 判例索引无效")
        return

    case = cases[idx]

    st.markdown(f"""
    <div style="padding: 12px; background: #F5F9F5; border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #4A5D53; font-weight: 600;">📝 编辑判例 #{idx + 1}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form(f"edit_basic_form_{idx}"):
        # 判例描述
        f_txt = st.text_area(
            "📝 判例描述",
            case.get('text', ''),
            height=80,
            key=f"edit_basic_txt_{idx}"
        )

        st.markdown("##### 🏷️ 因子评分详情")

        fc1, fc2 = st.columns(2)
        input_scores = {}
        current_scores = case.get('scores', {})

        for i, f in enumerate(FACTORS):
            data = current_scores.get(f, {})
            with (fc1 if i % 2 == 0 else fc2):
                st.markdown(f"**{f}**")

                col_score, col_comment = st.columns([1, 2])

                with col_score:
                    val = st.number_input(
                        "分数",
                        0, 9, data.get('score', 7),
                        key=f"edit_basic_s_{idx}_{i}",
                        label_visibility="collapsed"
                    )

                with col_comment:
                    cmt = st.text_input(
                        "评语",
                        data.get('comment', ''),
                        key=f"edit_basic_c_{idx}_{i}",
                        label_visibility="collapsed"
                    )

                sug = st.text_input(
                    "建议",
                    data.get('suggestion', ''),
                    key=f"edit_basic_a_{idx}_{i}",
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
            case.get('master_comment', ''),
            height=60,
            key=f"edit_basic_master_{idx}"
        )

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.form_submit_button("💾 保存修改", type="primary", width='stretch'):
                # 更新判例数据
                cases[idx] = {
                    "text": f_txt,
                    "scores": input_scores,
                    "master_comment": f_master,
                    "created_at": case.get('created_at', time.strftime("%Y-%m-%d %H:%M:%S"))
                }
                cases[idx] = ResourceManager.strip_case_vector(cases[idx])
                st.session_state.basic_cases = cases
                ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)
                st.session_state.editing_basic_idx = None
                st.success("✅ 已保存修改！")
                time.sleep(0.3)
                st.rerun()

        with col2:
            if st.form_submit_button("❌ 取消", width='stretch'):
                st.session_state.editing_basic_idx = None
                st.rerun()


# ==========================================
# 进阶判例弹窗
# ==========================================

@st.dialog("📋 进阶判例列表", width="large")
def show_supp_cases_dialog(embedder):
    """展示当前进阶判例列表 - 升级版（支持勾选转移到基础判例）"""
    _, cases = st.session_state.supp_cases

    if not cases:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-text">暂无进阶判例</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # 初始化勾选状态存储
    if 'supp_case_checkboxes' not in st.session_state:
        st.session_state.supp_case_checkboxes = {}

    st.markdown(f"**📊 共 {len(cases)} 条进阶判例**")
    st.markdown("---")

    for idx, case in enumerate(cases):
        with st.container():
            # 使用两列布局：左侧勾选框，右侧expander
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                # 勾选框
                current_value = st.session_state.supp_case_checkboxes.get(idx, False)
                checked = st.checkbox("", key=f"supp_check_{idx}",
                                     value=current_value,
                                     label_visibility="collapsed")
                st.session_state.supp_case_checkboxes[idx] = checked
            with col2:
                with st.expander(f"📋 判例 {idx + 1}", expanded=False):
                    # 1. 判例描述
                    st.markdown("**📝 判例描述：**")
                    st.markdown(f"> {case.get('text', '')}")

                    # 2. 六因子评分详情
                    st.markdown("---")
                    st.markdown("**🏷️ 因子评分详情：**")

                    scores = case.get('scores', {})

                    # 分两列展示因子
                    fc1, fc2 = st.columns(2)

                    for i, (factor, data) in enumerate(scores.items()):
                        with (fc1 if i % 2 == 0 else fc2):
                            with st.container(border=True):
                                st.markdown(f"**{factor}**")
                                score_val = data.get('score', '-')
                                comment = data.get('comment', '暂无评语')
                                suggestion = data.get('suggestion', '暂无建议')

                                col_s, col_d = st.columns([1, 2])
                                with col_s:
                                    st.markdown(f"<span style='font-size:12px; color:#666;'>分数: </span><span style='font-size:14px; font-weight:bold; color:#2E7D32;'>{score_val}/9</span>", unsafe_allow_html=True)
                                with col_d:
                                    if comment and comment != '暂无评语':
                                        st.caption(f"💬 {comment}")
                                    if suggestion and suggestion != '暂无建议':
                                        st.caption(f"💡 {suggestion}")

                    # 3. 宗师总评
                    st.markdown("---")
                    st.markdown("**🍵 宗师总评：**")
                    st.info(case.get('master_comment', '暂无'))

                    # 4. 编辑按钮
                    st.markdown("---")
                    if st.button("✏️ 编辑此判例", key=f"edit_supp_{idx}", width='stretch'):
                        st.session_state.editing_supp_idx = idx
                        st.rerun()

    st.markdown("---")

    # 显示已选数量和操作按钮
    selected_count = sum(1 for v in st.session_state.supp_case_checkboxes.values() if v)
    st.markdown(f"已选择: **{selected_count}** 条")

    col_transfer, col_close = st.columns([1, 1])

    with col_transfer:
        if st.button("⬅️ 转移到基础判例", type="primary",
                     disabled=selected_count == 0,
                     width='stretch'):
            _transfer_supp_to_basic(embedder)

    with col_close:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


def _transfer_supp_to_basic(embedder):
    """将选中的进阶判例转移到基础判例。"""
    selected_indices = sorted([
        idx for idx, checked in st.session_state.supp_case_checkboxes.items()
        if checked
    ])

    if not selected_indices:
        return

    _, supp_data = st.session_state.supp_cases
    selected_cases = []
    for idx in reversed(selected_indices):
        if idx < len(supp_data):
            case = dict(supp_data.pop(idx))
            selected_cases.append(ResourceManager.strip_case_vector(case))
    selected_cases.reverse()

    try:
        st.session_state.basic_cases.extend(selected_cases)
        ResourceManager.save_json(st.session_state.basic_cases, PATHS.basic_case_data)

        new_idx, supp_data = ResourceManager.sync_supp_cases(supp_data, embedder=embedder)
        st.session_state.supp_cases = (new_idx, supp_data)
        st.session_state.supp_case_checkboxes = {}

        st.success(f"✅ 已成功转移 {len(selected_cases)} 条到基础判例！")
        time.sleep(0.3)
        st.rerun()
    except Exception as e:
        st.error(f"❌ 转移失败: {e}")


@st.dialog("✏️ 编辑进阶判例", width="large")
def edit_supp_case_dialog(idx: int, embedder):
    """编辑指定进阶判例 - 完整表单编辑。"""
    _, cases = st.session_state.supp_cases

    if idx >= len(cases):
        st.error("❌ 判例索引无效")
        return

    case = cases[idx]

    st.markdown(f"""
    <div style="padding: 12px; background: #F5F9F5; border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #4A5D53; font-weight: 600;">📝 编辑判例 #{idx + 1}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.form(f"edit_supp_form_{idx}"):
        f_txt = st.text_area(
            "📝 判例描述",
            case.get('text', ''),
            height=80,
            key=f"edit_supp_txt_{idx}"
        )

        st.markdown("##### 🏷️ 因子评分详情")

        fc1, fc2 = st.columns(2)
        input_scores = {}
        current_scores = case.get('scores', {})

        for i, f in enumerate(FACTORS):
            data = current_scores.get(f, {})
            with (fc1 if i % 2 == 0 else fc2):
                st.markdown(f"**{f}**")

                col_score, col_comment = st.columns([1, 2])

                with col_score:
                    val = st.number_input(
                        "分数",
                        0, 9, data.get('score', 7),
                        key=f"edit_supp_s_{idx}_{i}",
                        label_visibility="collapsed"
                    )

                with col_comment:
                    cmt = st.text_input(
                        "评语",
                        data.get('comment', ''),
                        key=f"edit_supp_c_{idx}_{i}",
                        label_visibility="collapsed"
                    )

                sug = st.text_input(
                    "建议",
                    data.get('suggestion', ''),
                    key=f"edit_supp_a_{idx}_{i}",
                    placeholder="改进建议..."
                )

                input_scores[f] = {
                    "score": val,
                    "comment": cmt,
                    "suggestion": sug
                }

        f_master = st.text_area(
            "🍵 宗师总评",
            case.get('master_comment', ''),
            height=60,
            key=f"edit_supp_master_{idx}"
        )

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.form_submit_button("💾 保存修改", type="primary", width='stretch'):
                try:
                    updated_case = {
                        "text": f_txt,
                        "scores": input_scores,
                        "master_comment": f_master,
                        "created_at": case.get('created_at', time.strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    if f_txt == case.get('text') and ResourceManager._normalize_vector(case.get("_embedding")) is not None:
                        updated_case["_embedding"] = case.get("_embedding")
                    ResourceManager.ensure_case_embedding(updated_case, embedder)
                    cases[idx] = updated_case
                    new_idx, cases = ResourceManager.sync_supp_cases(cases, embedder=embedder)
                    st.session_state.supp_cases = (new_idx, cases)
                    st.session_state.editing_supp_idx = None
                    st.success("✅ 已保存修改，并刷新进阶判例索引！")
                    time.sleep(0.3)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 保存失败: {e}")

        with col2:
            if st.form_submit_button("❌ 取消", width='stretch'):
                st.session_state.editing_supp_idx = None
                st.rerun()



# ==========================================
# 茶评示例管理弹窗
# ==========================================

@st.dialog("⚙️ 茶评示例管理", width="large")
def manage_tea_examples_dialog():
    """管理茶评示例列表"""
    st.markdown("""
    <div style="padding: 12px; background: linear-gradient(135deg, #F5F9F5 0%, #EDF5EB 100%); border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4A5D53;">
        <span style="color: #4A5D53; font-size: 1.1em; font-weight: 600;">📚 茶评示例管理</span>
    </div>
    """, unsafe_allow_html=True)

    # 加载示例列表
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES[:])  # 复制一份

    st.markdown(f"**📊 共 {len(examples)} 个示例**")
    st.markdown("---")

    # 显示所有示例
    for idx, ex in enumerate(examples):
        with st.container(border=True):
            col1, col2, col3 = st.columns([5, 1, 1])

            with col1:
                st.markdown(f"**{ex['title']}**")
                st.caption(ex.get('text', '')[:80] + "...")

            with col2:
                if st.button("✏️", key=f"edit_tea_{idx}", width='stretch', help="编辑"):
                    st.session_state.editing_tea_example_idx = idx
                    st.rerun()

            with col3:
                if st.button("🗑️", key=f"del_tea_{idx}", width='stretch', help="删除"):
                    examples.pop(idx)
                    ResourceManager.save_tea_examples(examples)
                    st.session_state.tea_examples = examples
                    st.success(f"✅ 已删除示例")
                    st.rerun()

    st.markdown("---")

    # 底部按钮
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ 新增示例", type="primary", width='stretch'):
            st.session_state.editing_tea_example_idx = -1  # -1 表示新增
            st.rerun()

    with col2:
        if st.button("🔄 恢复默认", width='stretch'):
            ResourceManager.save_tea_examples(TEA_EXAMPLES)
            st.session_state.tea_examples = TEA_EXAMPLES[:]
            st.success("✅ 已恢复为默认示例")
            st.rerun()

    with col4:
        if st.button("✅ 关闭", type="secondary", width='stretch'):
            st.rerun()


@st.dialog("✏️ 编辑茶评示例", width="large")
def edit_tea_example_dialog(idx: int):
    """编辑指定茶评示例"""
    # 加载当前示例列表
    examples = st.session_state.get('tea_examples', TEA_EXAMPLES[:])

    if idx == -1:
        # 新增模式
        st.markdown("""
        <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #4A5D53; font-weight: 600;">➕ 新增茶评示例</span>
        </div>
        """, unsafe_allow_html=True)
        current_title = ""
        current_text = ""
    else:
        # 编辑模式
        if idx >= len(examples):
            st.error("❌ 示例索引无效")
            return

        st.markdown(f"""
        <div style="padding: 12px; background: #EDF5EB; border-radius: 8px; margin-bottom: 20px;">
            <span style="color: #4A5D53; font-weight: 600;">✏️ 编辑茶评示例 #{idx + 1}</span>
        </div>
        """, unsafe_allow_html=True)
        current_title = examples[idx]['title']
        current_text = examples[idx]['text']

    # 编辑表单
    new_title = st.text_input(
        "标题",
        current_title,
        key=f"tea_title_{idx}",
        placeholder="例如：🌸 桂花乌龙",
        max_chars=50  # 限制标题长度
    )
    new_text = st.text_area(
        "内容",
        current_text,
        height=200,
        key=f"tea_text_{idx}",
        placeholder="请输入茶评描述...",
        max_chars=2000  # 限制内容长度
    )

    # 输入验证：去除首尾空格
    if new_title:
        new_title = new_title.strip()
    if new_text:
        new_text = new_text.strip()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 保存", type="primary", key=f"save_tea_{idx}"):
            if not new_title or not new_text:
                st.error("❌ 标题和内容不能为空")
            elif len(new_title) > 50:
                st.error("❌ 标题不能超过50个字符")
            elif len(new_text) > 2000:
                st.error("❌ 内容不能超过2000个字符")
            else:
                new_example = {"title": new_title, "text": new_text}

                if idx == -1:
                    # 新增
                    examples.append(new_example)
                    st.success("✅ 已添加新示例")
                else:
                    # 更新
                    examples[idx] = new_example
                    st.success("✅ 已保存修改")

                # 保存到文件和 session_state
                ResourceManager.save_tea_examples(examples)
                st.session_state.tea_examples = examples
                st.session_state.editing_tea_example_idx = None
                st.session_state.show_tea_examples = True  # 返回示例列表
                st.rerun()

    with col3:
        if st.button("❌ 取消", type="secondary", key=f"cancel_tea_{idx}"):
            st.session_state.editing_tea_example_idx = None
            st.session_state.show_tea_examples = True  # 返回主弹窗
            st.rerun()
