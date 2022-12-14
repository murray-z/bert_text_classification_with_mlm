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
    """??????MLM??????????????????"""
    # ????????????????????????????????????
    labels = input_ids.clone()

    # ??????????????????????????????????????????
    probability_matrix = torch.full(input_ids.shape, mlm_probability)

    # ??????tokenizer???????????????mask
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                           for val in input_ids.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    # ??????????????????????????????0??????
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # ?????????????????????????????????????????????????????????????????????mlm_probability???
    # ?????????????????????MLM?????????????????????
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # ??????????????????????????????-100??????????????????????????????
    labels[~masked_indices] = -100

    # ??????????????? 80% ???[mask]??????
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # ??????????????? 10% ??????????????????
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & (~indices_replaced)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # ????????????????????? 10% ????????????
    return input_ids, labels