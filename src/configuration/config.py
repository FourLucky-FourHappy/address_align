from pathlib import Path
# 目录配置
# 	根目录

ROOT_DIR = Path(__file__).parent.parent.parent
# 	源数据目录
RAW_DATA_DIR = ROOT_DIR / "data"/'raw'
# 	预处理目录
PROCESSED_DATA_DIR = ROOT_DIR / "data"/'processed'
# 	日志目录
LOG_DIR = ROOT_DIR / "logs"
# 	模型保存目录
MODELS_DIR = ROOT_DIR / "models"
# 	存放预训练模型目录
PRE_TRAINED_DIR = ROOT_DIR / 'pretrained'