from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT_DIR / "logs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# ---- 原始语料（download_data.py 下载；文件已存在则跳过）----
# 仅保留 GAIIC 2022（京东商品标题实体识别）作为训练语料；天池数据集已移除。
GAIIC_FILE = RAW_DATA_DIR / "gaiic_train.txt"
# GAIIC 官方测试集（竞赛私有，通常无法下载；若存在则优先作为 test 切分）
GAIIC_TEST_FILE = RAW_DATA_DIR / "gaiic_test.txt"

# ---- 灌库产物（docker/init.sql：自包含建表 + 维表预制；生成脚本已随冻结交付移除）----
MYSQL_SEED_FILE = ROOT_DIR.parent.parent / "docker" / "init.sql"

MODEL_NAME = "google-bert/bert-base-chinese"

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

SAVE_STEPS = 500
EVAL_STEPS = 500
LOGS_STEPS = 100

# 单类型 TAG（下游图谱只存 Tag 文本，无 type 字段）。
# GAIIC 的多类型标签（B-1..B-54 脱敏编码）在 preprocess 中只取 BIO 前缀，塌缩到这 3 类。
LABELS = ["B", "I", "O"]

# ---- 预处理 ----
RANDOM_SEED = 42
VALID_RATIO = 0.1      # train/valid 切分比例，其余为 train
MAX_SENT_LEN = 200     # 超过此字数的句子直接丢弃（避免 bert 截断丢尾部实体）