"""
visualization.py
================
可视化函数模块 - Windows 系统中文字体增强版
"""

import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 确保使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, FontManager
from scipy.interpolate import make_interp_spline
import logging
import platform

logger = logging.getLogger(__name__)


# ==========================================
# Windows 系统字体优先级配置
# ==========================================

# 根据操作系统调整字体优先级
if platform.system() == 'Windows':
    FONT_PREFERENCES = [
        "Microsoft YaHei",     # 微软雅黑（Windows 推荐）
        "SimHei",              # 黑体
        "SimSun",              # 宋体
        "KaiTi",               # 楷体
        "FangSong",            # 仿宋
        "Noto Sans CJK SC",    # 如果用户安装了
        "Noto Sans SC",        # 如果用户安装了
    ]
elif platform.system() == 'Darwin':  # macOS
    FONT_PREFERENCES = [
        "PingFang SC",         # 苹方
        "Heiti SC",            # 黑体-简
        "STHeiti",             # 华文黑体
        "Noto Sans CJK SC",
        "Noto Sans SC",
    ]
else:  # Linux
    FONT_PREFERENCES = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
    ]


# ==========================================
# 字体检测与缓存
# ==========================================

_font_cache = None


def _find_available_chinese_font() -> str:
    """
    跨平台动态检测系统中可用的中文字体

    Returns:
        str: 可用的中文字体名称
    """
    global _font_cache

    if _font_cache is not None:
        return _font_cache

    # Windows 系统直接返回微软雅黑（绕过 Matplotlib 字体检测）
    if platform.system() == 'Windows':
        logger.info("✅ Windows 系统，直接使用微软雅黑")
        _font_cache = "Microsoft YaHei"
        return "Microsoft YaHei"

    try:
        font_manager = FontManager()
        available_fonts = set(font_manager.get_names())

        logger.info(f"🔍 系统: {platform.system()}, 检测到 {len(available_fonts)} 个字体")

        # 按优先级查找可用字体
        for preferred_font in FONT_PREFERENCES:
            if preferred_font in available_fonts:
                logger.info(f"✅ 使用中文字体: {preferred_font}")
                _font_cache = preferred_font
                return preferred_font

        # 模糊匹配：查找包含中文字体关键词的字体
        fallback_keywords = ['cjk', 'chinese', 'sc', 'tc', 'noto', 'yahei', 'simhei', 'pingfang', 'wqy']
        fallback_fonts = [f for f in available_fonts if any(
            keyword in f.lower() for keyword in fallback_keywords
        )]

        if fallback_fonts:
            logger.warning(f"⚠️ 使用备选字体: {fallback_fonts[0]}")
            _font_cache = fallback_fonts[0]
            return fallback_fonts[0]

        # 未找到任何中文字体
        logger.error("❌ 未找到中文字体，将使用拼音标签")
        _font_cache = "DejaVu Sans"
        return _font_cache

    except Exception as e:
        logger.error(f"字体检测失败: {e}", exc_info=True)
        _font_cache = "DejaVu Sans"
        return _font_cache


def _get_chinese_font_prop(size: int = 12, weight: str = 'normal') -> FontProperties:
    """
    获取中文字体属性对象

    Args:
        size: 字体大小
        weight: 字体粗细 ('normal', 'bold', 'light')

    Returns:
        FontProperties: 字体属性对象
    """
    font_name = _find_available_chinese_font()
    return FontProperties(family=font_name, size=size, weight=weight)


# ==========================================
# Matplotlib 全局配置（作为后备）
# ==========================================

_available_font = _find_available_chinese_font()
plt.rcParams['font.sans-serif'] = [_available_font] + FONT_PREFERENCES
plt.rcParams['axes.unicode_minus'] = False

logger.info(f"📝 Matplotlib 字体: {_available_font}")


# ==========================================
# 中国风配色方案
# ==========================================
TEA_COLORS = {
    # 六因子专属颜色
    "优雅性": "#F0C75E",   # 缃色
    "辨识度": "#789262",    # 竹青
    "协调性": "#B36D61",    # 檀色
    "饱和度": "#845538",    # 赭石
    "持久性": "#5B7C99",    # 黛蓝
    "苦涩度": "#4A3728",   # 紫檀
    # 三段风味颜色
    "top": "#F0C75E",      # 缃色 - 前调(香)
    "mid": "#B36D61",      # 檀色 - 中调(味)
    "base": "#4A3728",     # 紫檀 - 后调(韵)
}


# ==========================================
# 1. 核心计算逻辑
# ==========================================

def calculate_section_scores(scores):
    """
    将六因子得分为：前调(Top)、中调(Mid)、尾调(Base)

    Args:
        scores: 六因子得分字典 (可能是简单格式或嵌套格式)

    Returns:
        tuple: (top, mid, base) 三个分数
    """
    # 辅助函数：安全获取分数，默认为 0
    def get(key):
        value = scores.get(key, 0)

        # 如果是字典结构，提取 'score' 键
        if isinstance(value, dict):
            return float(value.get('score', 0))

        # 如果是数字，直接转换
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    top = (get('优雅性') + get('辨识度')) / 2
    mid = (get('协调性') + get('饱和度')) / 2
    base = (get('持久性') + get('苦涩度')) / 2

    return top, mid, base


# ==========================================
# 2. 六因子雷达图 (Plotly)
# ==========================================

def plot_radar_chart(scores_data):
    """
    绘制单次评测的六因子雷达图

    Args:
        scores_data: 六因子得分数据

    Returns:
        plotly.graph_objects.Figure: 雷达图对象
    """
    categories = ["优雅性", "辨识度", "协调性", "饱和度", "持久性", "苦涩度"]
    values = [float(scores_data.get(c, 0)) for c in categories]

    # 使用中国风颜色
    line_color = "#4A5D53"  # 黛绿
    fill_color = "rgba(74, 93, 83, 0.4)"

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='当前风味',
        line_color=line_color,
        fillcolor=fill_color
    ))

    # 为每个因子设置不同颜色
    factor_colors = {
        "优雅性": "#F0C75E",
        "辨识度": "#789262",
        "协调性": "#B36D61",
        "饱和度": "#845538",
        "持久性": "#5B7C99",
        "苦涩度": "#4A3728"
    }

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10, color="#666"),
                linecolor="rgba(0,0,0,0.1)",
                gridcolor="rgba(74, 93, 83, 0.2)"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#4A5D53", family="Noto Sans SC"),
                rotation=90,
                direction="clockwise"
            )
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 247, 245, 0.5)"
    )
    return fig


# ==========================================
# 3. 三段风味形态图 (Matplotlib)
# ==========================================

def plot_flavor_shape(scores_data):
    """
    绘制基于 '前中后' 三调的茶汤形态图

    Args:
        scores_data: 包含六因子得分的数据

    Returns:
        matplotlib.figure.Figure: 风味形态图对象
    """
    top, mid, base = calculate_section_scores(scores_data)

    fig, ax = plt.subplots(figsize=(4, 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    y = np.array([1, 2, 3])
    x = np.array([base, mid, top])

    y_new = np.linspace(1, 3, 300)
    try:
        spl = make_interp_spline(y, x, k=2)
        x_smooth = spl(y_new)
    except:
        x_smooth = np.interp(y_new, y, x)

    x_smooth = np.maximum(x_smooth, 0.1)

    # 中国风配色 - 缃色(前)/檀色(中)/紫檀(后)
    colors = {
        'base': TEA_COLORS['base'],   # 紫檀 - 后调(韵)
        'mid': TEA_COLORS['mid'],     # 檀色 - 中调(味)
        'top': TEA_COLORS['top']      # 缃色 - 前调(香)
    }

    mask_base = (y_new >= 1.0) & (y_new <= 1.6)
    ax.fill_betweenx(y_new[mask_base], -x_smooth[mask_base], x_smooth[mask_base],
                     color=colors['base'], alpha=0.9, edgecolor=None)

    mask_mid = (y_new > 1.6) & (y_new <= 2.4)
    ax.fill_betweenx(y_new[mask_mid], -x_smooth[mask_mid], x_smooth[mask_mid],
                     color=colors['mid'], alpha=0.85, edgecolor=None)

    mask_top = (y_new > 2.4) & (y_new <= 3.0)
    ax.fill_betweenx(y_new[mask_top], -x_smooth[mask_top], x_smooth[mask_top],
                     color=colors['top'], alpha=0.8, edgecolor=None)

    # 轮廓线
    ax.plot(x_smooth, y_new, color='#4A5D53', linewidth=1.5, alpha=0.4)
    ax.plot(-x_smooth, y_new, color='#4A5D53', linewidth=1.5, alpha=0.4)

    # 分隔线
    ax.axhline(y=1.6, color='#4A5D53', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=2.4, color='#4A5D53', linestyle=':', alpha=0.3, linewidth=1)

    # ========== 显式指定中文字体 ==========

    font_name = _find_available_chinese_font()
    has_chinese_font = font_name != "DejaVu Sans"

    logger.debug(f"绘制风味形态图，使用字体: {font_name}")

    if has_chinese_font:
        # 有中文字体：使用中文标签 + 显式 FontProperties
        font_props = _get_chinese_font_prop(size=12, weight='bold')

        # 香
        ax.text(0, 2.7, f"香\n{top:.1f}",
                ha='center', va='center',
                color='#4A5D53',
                fontproperties=font_props)

        # 味
        ax.text(0, 2.0, f"味\n{mid:.1f}",
                ha='center', va='center',
                color='#4A5D53',
                fontproperties=font_props)

        # 韵
        ax.text(0, 1.3, f"韵\n{base:.1f}",
                ha='center', va='center',
                color='#F5F5F5',
                fontproperties=font_props)

        logger.debug("✅ 使用中文字体标签")
    else:
        # 降级方案：使用拼音标签
        ax.text(0, 2.7, f"Aroma\n{top:.1f}",
                ha='center', va='center',
                color='#4A5D53',
                fontsize=12, fontweight='bold')

        ax.text(0, 2.0, f"Taste\n{mid:.1f}",
                ha='center', va='center',
                color='#4A5D53',
                fontsize=12, fontweight='bold')

        ax.text(0, 1.3, f"Finish\n{base:.1f}",
                ha='center', va='center',
                color='#F5F5F5',
                fontsize=12, fontweight='bold')

        logger.warning("⚠️ 使用拼音标签（中文字体不可用）")

    ax.axis('off')
    ax.set_xlim(-10, 10)
    ax.set_ylim(0.8, 3.2)

    return fig
