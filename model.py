import torch
import torch.nn as nn
import transformers
from transformers import BertForMaskedLM, BertTokenizer


class BertClsMLM(nn.Module):
    def __init__(self, model_name, hidden_size, label_size, dropout=0.1):
        super(BertClsMLM, self).__init__()
        self.backbone = BertForMaskedLM.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_size, label_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.backbone(input_ids, attention_mask, token_type_ids,
                               output_hidden_states=True, return_dict=True)
        cls_hidden = output.hidden_states[-1][:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.fc(cls_hidden)
        return logits

    def mlm_forward(self, input_ids, attention_mask, token_type_ids, labels):
        """MLM 计算损失"""
        outputs = self.backbone(input_ids, attention_mask, token_type_ids, labels=labels)
        return outputs.loss


if __name__ == '__main__':
    bert = BertClsMLM("bert-base-chinese", 768, 2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    inputs = tokenizer.batch_encode_plus(["The capital of France is [MASK].", "i loce you"], return_tensors="pt",
                                         truncation=True, padding=True)
    print(inputs)
    logits = bert(**inputs)
    print(logits)
