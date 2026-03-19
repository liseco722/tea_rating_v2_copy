# ==========================================
# 知识库配置
# ==========================================

# 知识库模式：local（纯本地）、github（云端）、hybrid（混合）
KB_MODE = "local"

# 向量化批处理大小（优化性能）
KB_EMBEDDING_BATCH_SIZE = 50  # 从 10 增加到 50，减少 API 调用次数


# ==========================================
# 路径配置类
# ==========================================

class PathConfig:
    """路径管理类 —— 集中定义所有文件路径"""

    # 外部资源文件
    SRC_SYS_PROMPT = Path("sys_p.txt")

    # 运行时数据目录
    DATA_DIR = Path("./tea_data")
    RAG_DIR = Path("./tea_data/RAG")
    BACKUP_DIR = Path("./tea_data/RAG_backup")
    CONFIG_DIR = Path("./config")

    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RAG_DIR.mkdir(exist_ok=True)
        self.BACKUP_DIR.mkdir(exist_ok=True)
        self.GRAPHRAG_DIR = self.DATA_DIR / "graphrag_artifacts"
        self.GRAPHRAG_DIR.mkdir(exist_ok=True)

        # --- 知识库（RAG） ---
        self.kb_index = self.DATA_DIR / "kb.index"
        self.kb_chunks = self.DATA_DIR / "kb_chunks.pkl"
        self.kb_files = self.DATA_DIR / "kb_files.json"
        self.kb_metadata = self.DATA_DIR / "kb_metadata.json"  # 文件元数据缓存
        self.kb_vectors = self.DATA_DIR / "kb_vectors.pkl"

        # --- 判例库（基础 + 进阶） ---
        self.basic_case_data = self.DATA_DIR / "basic_case.json"
        self.supp_case_index = self.DATA_DIR / "supp_cases.index"
        self.supp_case_data = self.DATA_DIR / "supplementary_case.json"

        # --- 微调与 Prompt ---
        self.training_file = self.DATA_DIR / "deepseek_finetune.jsonl"
        self.prompt_config_file = self.DATA_DIR / "prompts.json"
        self.tea_examples_file = self.DATA_DIR / "tea_examples.json"

        # --- 模板与默认配置（位于 config） ---
        self.template_file = self.CONFIG_DIR / "template.xlsx"
        self.default_prompts = self.CONFIG_DIR / "default_prompts.json"


# 全局路径实例
PATHS = PathConfig()
