# 项目背景
> 文本分类时，我们通常会有部分标注数据以及大量的同领域非标注数据，
> 在训练模型时，可以将两部分数据同时使用。

> 有标签数据做监督学习，无标签数据做MLM

## 数据格式
- 每条数据一行
- 有标签数据
  - label\t文本
- 无标签数据
  - 文本

## 运行
```
# 超参在train.py设置
python train.py
```