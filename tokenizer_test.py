from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"pretrained\roberta-small-wwm-chinese-cluecorpussmall")

# 直接处理，没有逐字拆分
res = tokenizer(["0000号"])["input_ids"]
print(res)
# [[101, 8241, 8129, 1384, 102]]
print([tokenizer.convert_ids_to_tokens(i) for i in res])
# [["[CLS]", "000", "##0", "号", "[SEP]"]]

# 分割成一个个字符，被当作多个序列处理
res = tokenizer(["0", "0", "0", "0", "号"])["input_ids"]
print(res)
# [[101, 121, 102], [101, 121, 102], [101, 121, 102], [101, 121, 102], [101, 1384, 102]]
print([tokenizer.convert_ids_to_tokens(i) for i in res])
# [['[CLS]', '0', '[SEP]'], ['[CLS]', '0', '[SEP]'], ['[CLS]', '0', '[SEP]'], ['[CLS]', '0', '[SEP]'], ['[CLS]', '号', '[SEP]']]

# 通过 is_split_into_words=True 让 tokenizer 将一个列表当作一个序列，并逐字处理
res = tokenizer(["0", "0", "0", "0", "号"], is_split_into_words=True)["input_ids"]
print(res)
# [101, 121, 121, 121, 121, 1384, 102]
print([tokenizer.convert_ids_to_tokens(i) for i in res])
# ['[CLS]', '0', '0', '0', '0', '号', '[SEP]']

# 或者逐字转换为 id
res = [tokenizer.convert_tokens_to_ids(i) for i in ["0", "0", "0", "0", "号"]]
res = [101] + res + [102]
print(res)
# [101, 121, 121, 121, 121, 1384, 102]
print([tokenizer.convert_ids_to_tokens(i) for i in res])
# ['[CLS]', '0', '0', '0', '0', '号', '[SEP]']

res = tokenizer([["中", "国", "杭", "州"], ["北", "京", "市"]], is_split_into_words=True)["input_ids"]
print(res)
