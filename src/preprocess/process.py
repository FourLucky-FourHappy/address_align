import pandas as pd
from sklearn.model_selection import train_test_split    # 划分数据集
from tqdm import tqdm
from configuration import config

def preprocess():
    print("Preprocessing data...")
    # 1.读取数据文件
    df = pd.read_json(config.RAW_DATA_DIR / config.RAW_DATA_FILE,lines=True,encoding='utf-8')

    # 2.数据提取和清洗
    df = df.dropna()

    # 3. 划分数据集
    train_df,test_df = train_test_split(df,test_size=config.TEST_RATE)
    train_df.to_json(config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILE, orient='records', lines=True,force_ascii=False)
    test_df.to_json(config.PROCESSED_DATA_DIR / config.TEST_DATA_FILE, orient='records', lines=True,force_ascii=False)
    print("数据预处理完成！")

if __name__ == '__main__':
    preprocess()