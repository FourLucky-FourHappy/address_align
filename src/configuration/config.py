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

#源文件
RAW_DATA_FILE = 'data.jsonl'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'

# 超参数
TEST_RATE = 0.2

LABELS = [
        "O",
        "B-prov", "I-prov", "E-prov",
        "B-city", "I-city", "E-city",
        "B-district", "I-district", "E-district", "S-district",
        "B-town", "I-town", "E-town", "S-town",
        "B-detail", "I-detail", "E-detail", "S-detail",
        "B-name", "I-name", "E-name",
        "B-phone", "I-phone", "E-phone",
    ]
FIELD_NAMES = {
    "prov": "省份",
    "city": "城市",
    "district": "区县",
    "town": "街道",
    "detail": "详细地址",
    "name": "姓名",
    "phone": "电话",
}
