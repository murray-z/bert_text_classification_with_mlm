# coding:utf-8
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from model import BertClsMLM
from dataset import LabelDataSet, UnlabelDataSet, mask_tokens
from sklearn.metrics import classification_report
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 超参数
label_batch_size = 128
unlabel_batch_size = 128
lr = 2e-5
epochs = 20
max_len = 50
hidden_size = 768
label_size = 10

# 是否使用MLM
mlm = True
# LML Loss 权重
Lambda = 1.
# Linear warmup 步数比
warmup_ratio = 0.1
# grad clip
max_norm = 1.
num_works = 5
# 预训练bert名称
pretrained_model_name = "bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_train_path = "./data/label_data_train.txt"
label_val_path = "./data/label_data_val.txt"
label_test_path = "./data/label_data_test.txt"
unlabel_path = "./data/unlabel_data.txt"
save_model_path = "./model/best_weights.pth"
label2idx_path = "./data/label2id.json"


def val(model, data_loader, device):
    model.eval()
    acc_num = 0.
    all_num = 0.
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_data = batch_data[:-1]
            labels = batch_data[-1]
            preds = model(*input_data)
            preds = torch.argmax(preds, dim=1)
            acc_num += torch.sum(preds == labels)
            all_num += preds.size()[0]
    return acc_num/all_num


def final_test(model, data_loader, model_path, device, label2idx):
    idx2label = {idx: label for label, idx in label2idx.items()}
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            batch_data = [data.to(device) for data in batch_data]
            input_data = batch_data[:-1]
            labels = batch_data[-1]
            preds = model(*input_data)
            preds = torch.argmax(preds, dim=1)
            labels = labels.cpu().numpy().tolist()
            preds = preds.cpu().numpy().tolist()
            true_labels.extend(labels)
            pred_labels.extend(preds)
    true_labels = [idx2label[idx] for idx in true_labels]
    pred_labels = [idx2label[idx] for idx in pred_labels]

    table = classification_report(true_labels, pred_labels, digits=4)
    return table


def train(mlm=mlm):
    """
    :param mlm: 是否使用MLM
    :return:
    """
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    # 加载标注数据集
    print("load label train data ......")
    label_train_dataset = LabelDataSet(tokenizer, data_type="train", max_len=max_len,
                                       label2id_path=label2idx_path, label_data_path=label_train_path,)
    label_train_dataloader = DataLoader(label_train_dataset, batch_size=label_batch_size, shuffle=True,
                                        num_workers=num_works)

    # 分类标签id
    label2idx = json.loads(open(label2idx_path).read())

    print("load label val data ......")
    label_val_dataset = LabelDataSet(tokenizer, data_type="val", max_len=max_len,
                                     label2id_path=label2idx_path, label_data_path=label_val_path)
    label_val_dataloader = DataLoader(label_val_dataset, batch_size=label_batch_size, shuffle=False,
                                      num_workers=num_works)

    print("load label test data ......")
    label_test_dataset = LabelDataSet(tokenizer, data_type="test", max_len=max_len,
                                      label2id_path=label2idx_path, label_data_path=label_test_path)
    label_test_dataloader = DataLoader(label_test_dataset, batch_size=label_batch_size, shuffle=False,
                                       num_workers=num_works)

    if mlm:
        # 加载无标签数据集
        print("load unlabel data ......")
        unlabel_dataset = UnlabelDataSet(tokenizer, unlabel_data_path=unlabel_path, max_len=max_len)
        unlabel_dataloader = DataLoader(unlabel_dataset, batch_size=unlabel_batch_size, shuffle=False,
                                        num_workers=num_works)
        unlabel_iter = iter(unlabel_dataloader)

    # 模型
    print("set model ......")
    model = BertClsMLM(pretrained_model_name, hidden_size, label_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=model.parameters(), lr=lr)

    num_training_steps = len(label_train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(warmup_ratio * num_training_steps),
                                                num_training_steps=num_training_steps)
    best_acc = 0.

    print("start training ......")
    for epoch in range(epochs):
        for step, batch in enumerate(label_train_dataloader):
            optimizer.zero_grad()
            batch = [d.to(device) for d in batch]
            labels = batch[-1]
            logits = model(*batch[:3])
            loss_label = criterion(logits, labels)

            if mlm:
                try:
                    unlabel_input_ids, unlabel_attention_mask, unlabel_token_type_ids = next(unlabel_iter)
                    unlabel_input_ids, unlabel_labels = mask_tokens(unlabel_input_ids)
                    unlabel_batch = [d.to(device) for d in [unlabel_input_ids, unlabel_attention_mask,
                                                            unlabel_token_type_ids, unlabel_labels]]
                except:
                    unlabel_iter = iter(unlabel_dataloader)
                    unlabel_input_ids, unlabel_attention_mask, unlabel_token_type_ids = next(unlabel_iter)
                    unlabel_input_ids, unlabel_labels = mask_tokens(unlabel_input_ids, tokenizer=tokenizer)
                    unlabel_batch = [d.to(device) for d in [unlabel_input_ids, unlabel_attention_mask,
                                                            unlabel_token_type_ids, unlabel_labels]]

                loss_unlabel = model.mlm_forward(*unlabel_batch)

                all_loss = loss_label + Lambda * loss_unlabel
            else:
                all_loss = loss_label

            if step % 5 == 0:
                print("TRAIN => Epoch: {}  Step: {}  Loss: {}".format(epoch, step, all_loss.item()))

            all_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            scheduler.step()

        # 每个epoch验证一次
        val_acc = val(model, label_val_dataloader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
            print("VAL => Epoch: {} Best Val ACC: {}".format(epoch, best_acc))

    # 最有模型测试
    classification_reports = final_test(model, data_loader=label_test_dataloader, model_path=save_model_path,
                                        device=device, label2idx=label2idx)
    print("Test RES:")
    print(classification_reports)


if __name__ == '__main__':
    train()





