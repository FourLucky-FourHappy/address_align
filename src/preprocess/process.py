import pandas as pd
from sklearn.model_selection import train_test_split    # 划分数据集
from tqdm import tqdm
from configuration import config
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer

def preprocess():
    print("开始预处理")
    labels = [
        "O",
        "B-prov", "I-prov", "E-prov",
        "B-city", "I-city", "E-city",
        "B-district", "I-district", "E-district", "S-district",
        "B-town", "I-town", "E-town", "S-town",
        "B-detail", "I-detail", "E-detail", "S-detail",
        "B-name", "I-name", "E-name",
        "B-phone", "I-phone", "E-phone",
    ]
    label2id = {label: i for i, label in enumerate(labels)}
    dataset_dict = load_dataset("json",data_files=str(config.RAW_DATA_DIR/config.RAW_DATA_FILE))
    dataset = dataset_dict["train"]
    # print(dataset)
    dataset = dataset.filter(lambda x: len(x["labels"]) > 0)
    # 分词
    dataset_dict = dataset.train_test_split(test_size=config.TEST_RATE)
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / "roberta-small-wwm-chinese-cluecorpussmall",use_fast=True)

    def encode(example):
        # 将句子切分为文字和标签
        text = example["text"]
        tags = example["labels"]
        if len(text) != len(tags):
            raise ValueError("text and tags must have same length")
        text = [tokenizer.unk_token if char == "\uf881" else char for char in text]
        # 分词器进行划分
        encoded = tokenizer(text,is_split_into_words=True)
        #划分后,同一个词的多个id只进行一个损失计算
        aligned_labels = []
        previous_id = None
        for word_id in encoded.word_ids():
            if word_id is None or word_id == previous_id:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(label2id[tags[word_id]])
            previous_id = word_id
        encoded["labels"] = aligned_labels
        # print(encoded)
        return encoded

    encoded_dataset = dataset_dict.map(
        encode,
        remove_columns=dataset_dict["train"].column_names,
    )
    encoded_dataset.save_to_disk(config.PROCESSED_DATA_DIR)
    print("数据预处理完成！")

if __name__ == '__main__':
    preprocess()