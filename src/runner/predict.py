import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from configuration import config
from address.region_repository import RegionRepository
from address.service import AddressAlignmentService

class Predictor:
    def __init__(self, device, model, tokenizer):
        self.device = device
        # 模型与输入放在同一设备，预测时关闭 Dropout。
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer

    # 预测只做前向传播，关闭梯度追踪以减少显存开销。
    @torch.inference_mode()
    def predict(self, text: str | list[str]) -> list[str] | list[list[str]]:
        # 把单条地址也包装成一个 batch，使用同一套处理流程。
        is_str = isinstance(text, str)
        addresses = [text] if is_str else text
        if not addresses:
            return []

        # 按训练时的规则逐字拆分，并用 [UNK] 保留特殊字符 U+F881 的位置。
        characters = [
            [self.tokenizer.unk_token if char == "\uf881" else char for char in address]
            for address in addresses
        ]

        # 每个字符列表是一条地址，补齐后转换为 PyTorch 张量。
        inputs = self.tokenizer(
            characters,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
        )
        # 超长时直接报错，避免截掉地址结尾的门牌号等信息。
        max_length = self.model.config.max_position_embeddings
        if inputs["input_ids"].shape[-1] > max_length:
            raise ValueError(f"地址分词后超过模型支持的 {max_length} 个 token")

        inputs = inputs.to(self.device)
        outputs = self.model(**inputs)
        # logits 的形状是 [地址数, token 数, 标签数]，取最后一维的最高分标签。
        predictions = outputs.logits.argmax(dim=-1).tolist()
        id2label = self.model.config.id2label

        results = []
        for batch_index, (address, prediction_ids) in enumerate(zip(addresses, predictions)):
            # 被 tokenizer 略过的字符保留原位置（如空格），默认标为 O。
            labels = ["O"] * len(address)
            previous_id = None
            # word_ids() 提供每个 token 对应的原始字符下标。
            for word_id, label_id in zip(inputs.word_ids(batch_index=batch_index), prediction_ids):
                # 跳过特殊 token，只取每个原始字符的第一个子 token。
                if word_id is not None and word_id != previous_id:
                    labels[word_id] = id2label[label_id]
                previous_id = word_id
            results.append(labels)

        # 单条地址返回一份标签列表，批量输入则返回多份列表。
        return results[0] if is_str else results


def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载验证成绩最好的微调权重，沿用训练时的分词器。
    model_path = config.MODELS_DIR / "best"
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到微调模型：{model_path}。请先运行训练并保存最佳权重。")

    tokenizer_path = config.PRE_TRAINED_DIR / "roberta-small-wwm-chinese-cluecorpussmall"
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, local_files_only=True,
    )
    # 使用快速分词器，后面才能通过 word_ids() 找回字符位置。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    predictor = Predictor(device, model, tokenizer)

    texts = [
        "中国浙江省杭州市余杭区葛墩路27号楼傅婷15830444519",
        "北京市市辖区通州区永乐店镇27号楼汪明13334219987",
        "高霞 13139243427北京市市辖区东风街道27号楼",
        "新疆维吾尔自治区划阿拉尔市金杨镇27号楼 刘燕 14727827196",
        "甘肃省南市文县碧口镇27号楼陈桂兰 13939269190",
        "陕西省渭南市华阴市罗镇27号楼 赵鑫15687584092",
        "邱金凤18582166250西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "广州市花都区花东镇27号楼张荣   18736672007",
        "北京市拱辰街道27号楼2单元 陈桂兰 13939269190",
        "河北省唐山市玉田县林北仓镇27号楼5单元 张荣 18736672007",
        "河北省朔州市玉田县林南仓镇27号楼5单元 张荣 18736672007"
    ]
    service = AddressAlignmentService(predictor, RegionRepository())
    results = service.align_address(texts)
    for index, result in enumerate(results, start=1):
        print(f"\n第 {index} 条地址")
        # 展示时将内部字段名转成中文，缺失的字段显示为“无”。
        for field, name in config.FIELD_NAMES.items():
            print(f"- {name}：{result[field] or '无'}")

if __name__ == '__main__':
    predict()
