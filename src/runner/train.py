from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForTokenClassification,DataCollatorForTokenClassification
from configuration import config
from preprocess.dataset import get_dataset


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size:int = 32
    learning_rate: float = 5e-5
    output_dir: str = './models'
    log_dir: str = './logs'
    save_steps: int = 100  # 每隔多少个训练步检查一次验证指标
    validation_rate: float = 0.1
    early_stop_metric: str = 'loss'  # 用哪个验证指标判断模型有没有进步
    early_stop_patience: int = 3  # 连续多少次验证没有刷新历史最佳时停止

    def __post_init__(self):
        if self.early_stop_metric not in ('loss', 'accuracy', 'f1'):
            raise ValueError('early_stop_metric 必须是 loss、accuracy 或 f1')
        if self.early_stop_patience < 1:
            raise ValueError('early_stop_patience 必须大于等于 1')

class Trainer:
    def __init__(self,device,
                 model,
                 test_dataset,
                 collate_fn,
                 train_dataset = None,
                 training_config:TrainingConfig = TrainingConfig(),
                 *, valid_dataset=None):
        # 设备和模型
        self.device = device
        self.model = model.to(device)
        # 训练超参数
        self.training_config = training_config

        # 数据集和数据整理器
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
        # tensorboard
        self.writer = SummaryWriter(
            log_dir=str(Path(self.training_config.log_dir) / time.strftime('%Y-%m-%d_%H-%M-%S')))
        # 全局的step
        self.step = 1
        # 初始最佳分数设为负无穷，让第一次有效的验证结果成为历史最佳
        self.early_stop_best_score = -float('inf')
        self.early_stop_counter = 0  # 连续未刷新历史最佳的验证次数

    def _get_dataloader(self, dataset, *, shuffle=True):
        dataset.set_format(type='torch')
        generator = torch.Generator()
        generator.manual_seed(42)
        return DataLoader(dataset=dataset,
                          batch_size=self.training_config.batch_size,
                          shuffle=shuffle,
                          generator=generator,
                          collate_fn=self.collate_fn)
    def train(self):
        current_step = 0
        # 获取数据加载器
        dataloader = self._get_dataloader(self.train_dataset)
        for epoch in range(1,1 + self.training_config.epochs):
            for inputs in tqdm(dataloader,desc = f"Epoch {epoch}",position = 0):
                current_step += 1
                if current_step < self.step:
                    continue
                loss = self.train_one_step(inputs)
                if self.step % self.training_config.save_steps == 0:
                    tqdm.write(f'Epoch:{epoch},Step:{self.step},Loss:{loss}')
                    self.writer.add_scalar('Loss', loss, self.step)
                    if self.valid_dataset is not None:
                        metrics = self.evaluate(self.valid_dataset)
                        metrics_str = ' | '.join(f'{key}:{value:.4f}' for key, value in metrics.items())
                        tqdm.write(f'Validation (step {self.step}): {metrics_str}')
                        for key, value in metrics.items():
                            self.writer.add_scalar(f'Validation/{key}', value, self.step)
                        # 每次验证后检查早停，return 会退出整个训练循环
                        if self._should_stop(metrics):
                            patience = self.training_config.early_stop_patience
                            tqdm.write(f'早停：第 {self.step} 步，连续 {patience} 次验证未改善')
                            # 本步已完成，退出前也要递增步数并写出 TensorBoard 日志
                            self.step += 1
                            self.writer.flush()
                            return
                self.step += 1

    def train_one_step(self,inputs):
        self.model.train()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # 评估函数
    @torch.inference_mode()
    def evaluate(self, dataset=None):
        if dataset is None:
            dataset = self.test_dataset
        # 获取数据加载器
        dataloader = self._get_dataloader(dataset, shuffle=False)
        self.model.eval()
        # 累加当前地址的所有token的损失和
        total_loss = 0.0
        #模型对每个有效token的预测标签
        all_predictions = []
        #有效token的真实标签
        all_labels = []
        for inputs in dataloader:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # 获取标签
            labels = inputs["labels"]
            mask = labels != -100
            #获取真实标签的数量
            token_count = mask.sum().item()
            if token_count == 0:
                continue
            # 前向传播
            outputs = self.model(**inputs)
            #获取预测标签
            predictions = outputs.logits.argmax(dim=-1)
            # 计算总损失
            total_loss += outputs.loss.item() * token_count
            all_predictions.extend(predictions[mask].tolist())
            all_labels.extend(labels[mask].tolist())

        if not all_labels:
            raise ValueError('评估数据集中没有有效的标签')

        return {
            'loss': total_loss / len(all_labels),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        }

    def _should_stop(self, metrics):
        metric_name = self.training_config.early_stop_metric
        metric = metrics[metric_name]
        # loss 越小越好，取负后就能和 accuracy、f1 一样按越大越好比较
        score = -metric if metric_name == 'loss' else metric

        # 只有有效分数严格超过历史最佳时，才保存新权重并清零计数
        if isfinite(score) and score > self.early_stop_best_score:
            best_model_path = Path(self.training_config.output_dir) / 'best'
            self.model.save_pretrained(str(best_model_path))
            self.early_stop_best_score = score
            self.early_stop_counter = 0
            tqdm.write(f'保存最佳模型至 {best_model_path}')
            return False

        # 持平也算未改善，累计到阈值后通知 train() 停止
        self.early_stop_counter += 1
        return self.early_stop_counter >= self.training_config.early_stop_patience

def train():
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型地址
    model_path = config.PRE_TRAINED_DIR / "roberta-small-wwm-chinese-cluecorpussmall"
    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=True)
    id2label = dict(enumerate(config.LABELS))
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(config.LABELS),
        id2label=id2label,
        label2id=label2id
    )

    training_config = TrainingConfig(output_dir=str(config.MODELS_DIR))
    # 数据划分
    splits = get_dataset('train').train_test_split(
        test_size=training_config.validation_rate,
        seed=42,
    )
    trainer = Trainer(
        device=device,
        model=model,
        train_dataset=splits['train'],
        valid_dataset=splits['test'],
        test_dataset=get_dataset("test"),
        collate_fn=DataCollatorForTokenClassification(tokenizer),
        training_config=training_config,
    )
    trainer.train()

if __name__ == '__main__':
    train()
