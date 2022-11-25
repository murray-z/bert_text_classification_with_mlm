# coding:utf-8
import json
import torch
from torch.utils.data import Dataset


class LabelDataSet(Dataset):
    def __init__(self,
                 tokenizer,
                 label_data_path,
                 label2id_path,
                 max_len=32,
                 data_type="train"):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = []
        self.labels = []
        with open(label_data_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == 1000:
                    break
                lis = line.strip().split("\t")
                self.texts.append(lis[1])
                self.labels.append(lis[0])

        if data_type == "train":
            label_dict = {label: idx for idx, label in enumerate(list(set(self.labels)))}
            with open(label2id_path, "w", encoding="utf-8") as f:
                json.dump(label_dict, f, ensure_ascii=False)
        else:
            with open(label2id_path, "r", encoding="utf-8") as f:
                label_dict = json.loads(f.read())

        self.labels = torch.tensor([label_dict[l] for l in self.labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.texts[idx], return_tensors="pt",
                                            padding="max_length", truncation=True,
                                            max_length=self.max_len)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids[0], attention_mask[0], token_type_ids[0], self.labels[idx]


class UnlabelDataSet(Dataset):
    def __init__(self,
                 tokenizer,
                 unlabel_data_path,
                 max_len):

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.texts = []
        with open(unlabel_data_path, encoding="utf-8") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(self.texts[idx], return_tensors="pt", padding="max_length",
                                            truncation=True, max_length=self.max_len)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return input_ids[0], attention_mask[0], token_type_ids[0]


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """构造MLM的输入、输出"""
    # 复制一份输入作为输出标签
    labels = input_ids.clone()

    # 和输入相同的以概率填充的矩阵
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    # 获得tokenizer中特殊字符mask
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                           for val in input_ids.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    # 将特殊字符所在位置以0填充
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 采用伯努利分布对文本位置进行采样，被采样概率为mlm_probability，
    # 被采样位置即为MLM需要预测的位置
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 将不需要预测的位置以-100进行填充，不计算损失
    labels[~masked_indices] = -100

    # 被采样位置 80% 用[mask]替换
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 被采样位置 10% 采用随机替换
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & (~indices_replaced)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 被采样位置剩余 10% 保持不变
    return input_ids, labels