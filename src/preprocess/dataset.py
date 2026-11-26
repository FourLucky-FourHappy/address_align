from datasets import load_from_disk
from transformers import AutoTokenizer,DataCollatorForTokenClassification
from configuration import config
from torch.utils.data import DataLoader

def get_dataset(dtype = 'train'):
    path = str(config.PROCESSED_DATA_DIR/ dtype)
    return load_from_disk(path)

def get_dataloader(tokenizer,dtype = 'train'):
    path = str(config.PROCESSED_DATA_DIR / dtype)
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')

    collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer,padding=True,return_tensors='pt')
    return DataLoader(dataset=dataset,batch_size=16,shuffle=True,collate_fn=collate_fn)

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / "roberta-small-wwm-chinese-cluecorpussmall",use_fast=True)
    dataloader = get_dataloader(tokenizer,'train')
    for batch in dataloader:
        for k,v in batch.items():
            print(k,v)
        break

