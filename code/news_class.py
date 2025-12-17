import json
import os
import random
import re
import time
from collections import Counter

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# 设备选择：优先使用 Apple GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# 0.数据验证
def data_validation():
    print(f'{'-' * 30}开始数据检查{'-' * 30}')
    # 1. 读取数据
    train_data = pd.read_csv('../data/train.csv')
    val_data = pd.read_csv('../data/val.csv')
    test_data = pd.read_csv('../data/test.csv')
    # 2. 基本信息
    print('[train_data]', train_data.shape, train_data.head())
    print('[val_data]', val_data.shape, val_data.head())
    print('[test_data]', test_data.shape, test_data.head())
    # 3.列结构检查
    print(train_data.columns)
    print(val_data.columns)
    print(test_data.columns)
    assert 'text' in train_data.columns, 'train.csv 缺少 text 列'
    assert 'label' in train_data.columns, 'train.csv 缺少 label 列'
    assert 'text' in val_data.columns, 'val.csv 缺少 text 列'
    assert 'label' in val_data.columns, 'val.csv 缺少 label 列'
    assert 'text' in test_data.columns, 'test.csv 缺少 text 列'
    assert 'label' in test_data.columns, 'test.csv 缺少 label 列'
    # 4.缺失值检查
    train_text_nan = train_data['text'].isna().sum()
    val_text_nan = val_data['text'].isna().sum()
    test_text_nan = test_data['text'].isna().sum()
    train_label_nan = train_data['label'].isna().sum()
    val_label_nan = val_data['label'].isna().sum()
    print('[train_text_nan]', train_text_nan)
    print('[val_text_nan]', val_text_nan)
    print('[test_text_nan]', test_text_nan)
    print('[train_label_nan]', train_label_nan)
    print('[val_label_nan]', val_label_nan)
    print(f'{'-' * 30}数据检查结束{'-' * 30}')



# 1.数据清洗
def data_clean():
    """
    数据没有缺失值，可以不动，尽量不破坏原数据
    :return: 返回三个读取的数据文件
    """
    train_data = pd.read_csv('../data/train.csv')
    val_data = pd.read_csv('../data/val.csv')
    test_data = pd.read_csv('../data/test.csv')
    return train_data, val_data, test_data


# 2.分词，构建词表
def build_vocab():
    # 0.数据验证,获取数据
    train_data, val_data, test_data = data_clean()
    # 1.对train文本进行分词
    # 所有的分词列表，二维
    train_texts = []
    # 所有的分词列表，一维
    all_words = []
    # 遍历每行文本并分词，一行为一个分词列表
    for text in train_data['text']:
        # 每一行的分词结果
        participle = jieba.lcut(text)
        # print(participle)
        # 添加到总列表中
        # 保留二维的结构
        train_texts.append(participle)
        # 打平为一维
        all_words.extend(participle)
    print(f'train_texts：{len(train_texts)}')
    print(f'all_words：{len(all_words)}\t{all_words[:10]}')
    # 2.去重，集合去重转列表
    train_texts_unique = list(set(all_words))
    print(f'train_texts_unique：{len(train_texts_unique)}')
    # 3.词频，统计每个词出现的次数
    word_counter = Counter(all_words)
    print(f'word_counter：{len(word_counter)}')
    # 4.构建词表，全保留
    # word_to_idx = {word: idx for idx, word in enumerate(train_texts_unique)}
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    # 索引从2开始,追加元素
    id_idx = 2
    for word in train_texts_unique:
        word_to_idx[word] = id_idx
        id_idx += 1
    print(f'word_to_idx：{len(word_to_idx)},{type(word_to_idx)}')
    # 5.文本数值化，文本使用索引表示，存储在列表中，为了Dataset 和 label 对齐
    # train_texts_idx = [word_to_idx[word] for word in all_words]
    train_texts_idx = [
        [word_to_idx[word] for word in text]
        for text in train_texts
    ]
    print(f'train_texts_idx：{len(train_texts_idx)}\t{type(train_texts_idx)}\t{train_texts_idx[:5]}')
    return train_texts_idx, word_to_idx


def texts_to_indices(texts, word_to_idx):
    texts_idx = []
    for text in texts:
        words = jieba.lcut(text)
        texts_idx.append(
            [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
        )
    return texts_idx


# 3.数据集封装
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        """

        :param texts: 二维列表，每一条是一个词索引列表
        :param labels: 一维列表，对应每条文本的标签
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        """

        :return: 返回数据集一共有多少条样本
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        根据索引 idx，取出一条数据
        :param idx:
        :return:
        """
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


def collate_fn(batch):
    """
    batch: [(text1, label1), (text2, label2), ...]
    """
    texts, labels = zip(*batch)
    # 找 batch 中最长句子
    max_len = max(len(text) for text in texts)
    padded_texts = []
    for text in texts:
        padded = text + [0] * (max_len - len(text))
        padded_texts.append(padded)
    texts_tensor = torch.tensor(padded_texts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return texts_tensor, labels_tensor


# 4.构建神经网络模型
class NewsClassifier(nn.Module):
    def __init__(self, vocab_size, num_class):
        super().__init__()
        # 1.词嵌入层
        self.embedding = nn.Embedding(vocab_size, 128)
        # 2.循环网络层
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        # 3.输出层
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        x = self.embedding(x)  # [batch, seq_len, 128]
        _, hidden = self.rnn(x)  # hidden: [1, batch, 256]
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)  # [batch, num_classes]
        return out


# 5.训练模型，使用训练集train.csv
def train_model():
    # 1.获取数据
    train_data, val_data, test_data = data_clean()
    # 2.构建词表
    train_texts_idx, word_to_idx = build_vocab()
    labels = train_data['label'].tolist()
    # 3.数据集和数据加载器
    dataset = NewsDataset(train_texts_idx, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    # 4.初始化模型
    vocab_size = len(word_to_idx)
    num_class = len(set(labels))
    # 模型，使用GPU
    model = NewsClassifier(vocab_size, num_class).to(device)
    # 5.损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 6.循环训练模型
    epochs = 15
    for epoch in range(epochs):
        # 开始时间
        start = time.time()
        # 迭代次数
        iter_num = 0
        # 总损失
        total_loss = 0
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            # 梯度清零，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累加损失
            total_loss += loss.item()
            iter_num += 1
        # 打印本轮训练信息
        print(
            f'epoch [{epoch + 1}/{epochs}] '
            f'loss: {total_loss / iter_num:.4f} '
            f'time: {time.time() - start:.2f}s'
        )
    # 保存模型
    torch.save(model.state_dict(), '../model/news_class_rnn_h256_e15.pth')


# 6.模型验证，使用验证集val.csv
def evaluate_model():
    print(f'{'-' * 30}开始模型验证{'-' * 30}')
    # 1.加载数据
    train_data, val_data, test_data = data_clean()
    # 2.构建词表
    train_texts_idx, word_to_idx = build_vocab()
    # 3.处理验证集文本和标签
    val_texts = val_data['text'].tolist()
    val_labels = val_data['label'].tolist()
    # 文本转为词索引
    val_texts_idx = texts_to_indices(val_texts, word_to_idx)
    # 4.构建验证集 Dataset 和 DataLoader
    val_dataset = NewsDataset(val_texts_idx, val_labels)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    # 5.加载模型
    vocab_size = len(word_to_idx)
    num_class = len(set(val_labels))
    model = NewsClassifier(vocab_size, num_class).to(device)
    model_path = '../model/news_class_rnn_h128_e15.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 7.切换为评估模式
    model.eval()
    # 8.开始验证
    all_preds = []
    all_labels = []
    # 验证阶段不需要梯度
    with torch.no_grad():
        for texts, labels in val_dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f'[val_acc] {acc:.4f}\t[model_path]{model_path}')
    print(f'{'-' * 30}结束模型验证{'-' * 30}')
    return 0


# 7.模型测试，使用测试集test.csv
# def test_model():
#     pass


# 8.数据可视化
# def data_visualize():
#     pass


# 主函数
def main():
    # 0.数据验证
    data_validation()
    # 1.数据清洗
    # data_clean()
    # 2.分词，构建词表
    # build_vocab()
    # 5.训练模型
    train_model()
    # 6.评估模型
    # evaluate_model()


if __name__ == '__main__':
    main()
