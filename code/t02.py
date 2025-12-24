import json
import os
import random
import re
import time

start = time.time()
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

# 设置最大文本长度
MAX_LEN = 200


# 设备选择：优先使用 Apple GPU (MPS)
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print("Using device:", device)

def get_device():
    """
    自动选择运行设备：
    1. 优先 CUDA（NVIDIA 显卡，如 RTX 3060）
    2. 其次 MPS（Apple Silicon）
    3. 最后 CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


device = get_device()


# 工具函数，用于读取类别描述文件
def load_label_map(path='../data/class.txt'):
    """
    读取类别描述文件
    :param path: 类别描述文件路径
    :return: dict {label_id: label_name}
    """
    label_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            idx, name = line.strip().split(maxsplit=1)
            label_map[int(idx)] = name
        # print("label_map:", label_map)
    return label_map


# 0.数据验证
def data_validation():
    """
    只负责检查数据是否“能不能用”，不做任何处理
    :return:
    """
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
    # 数据可视化
    counts = train_data['label'].value_counts().sort_index()
    counts.plot(kind='bar')
    plt.title('Train Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    for i, v in enumerate(counts):
        plt.text(i, v + 100, str(v), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('../data/images/label_dist.png')
    plt.close()


# 1.数据读取和清洗
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
    """
    1.只用 train 构建词表：防止数据泄漏，防止验证集“偷看答案”
    2.使用jieba中文分词：使用lcut分词转列表
    3.main函数统一调用，返回值统一保存，消除隐式依赖
    :return:1.train_texts_idx：返回一个二维列表，存储所有文本转词表索引，保持原数据结构 2.word_to_idx：返回一个字典，作为唯一词的词表
    """
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
        participle = jieba.lcut(text)[:MAX_LEN]
        # print(participle)
        # 添加到总列表中
        # 保留二维的结构
        train_texts.append(participle)
        # 打平为一维
        all_words.extend(participle)
    # print(f'train_texts：{len(train_texts)}')
    # print(f'all_words：{len(all_words)}\t{all_words[:10]}')
    # 2.去重，集合去重转列表
    train_texts_unique = list(set(all_words))
    # print(f'train_texts_unique：{len(train_texts_unique)}')
    # 3.词频，统计每个词出现的次数
    word_counter = Counter(all_words)
    # print(f'word_counter：{len(word_counter)}')
    # 4.构建词表，全保留
    # word_to_idx = {word: idx for idx, word in enumerate(train_texts_unique)}
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    # 索引从2开始,追加元素
    id_idx = 2
    for word in train_texts_unique:
        word_to_idx[word] = id_idx
        id_idx += 1
    # print(f'word_to_idx：{len(word_to_idx)},{type(word_to_idx)}')
    # 5.文本数值化，文本使用索引表示，存储在列表中，为了Dataset 和 label 对齐
    # train_texts_idx = [word_to_idx[word] for word in all_words]
    train_texts_idx = [
        [word_to_idx[word] for word in text]
        for text in train_texts
    ]
    # print(f'train_texts_idx：{len(train_texts_idx)}\t{type(train_texts_idx)}\t{train_texts_idx[:5]}')
    return train_texts_idx, word_to_idx


# 验证 / 测试文本数值化
def texts_to_indices(texts, word_to_idx):
    """
    1.把“验证 / 测试集的原始文本（字符串）” 转成“模型能吃的整数序列”
    2.不建词表，不改词表，只用训练集学到的 word_to_idx
    :param texts:
    :param word_to_idx:
    :return: 1.texts_idx：返回一个二维列表，存储索引版本的
    """
    # 最终结果容器，二维列表
    texts_idx = []
    # 遍历每行文本
    for text in texts:
        # 中文分词，最大长度截断
        words = jieba.lcut(text)[:MAX_LEN]
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
    用于定义 DataLoader 在生成一个 batch 时，如何将多条样本进行组合、补齐和张量化
    :param batch:
    :return:
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
    """
    定义一个 PyTorch 的神经网络模型：
    1.词嵌入层
    2.循环网络层
    3.输出层
    """

    def __init__(self, vocab_size, num_class):
        """
        vocab_size: 词表大小，例如train中的test列有多少词
        num_class: 分类数量，例如train的label列有多少种类
        """
        super().__init__()
        # 1.词嵌入层
        # 参数：词表大小，词嵌入维度
        self.embedding = nn.Embedding(vocab_size, 128)
        # 2.循环网络层
        # self.rnn = nn.RNN(
        #     input_size=128,
        #     hidden_size=256,
        #     num_layers=1,
        #     batch_first=True
        # )
        # 参数：input_size: 输入的维度，hidden_size: 隐藏层的维度，num_layers: 循环网络的层数，batch_first: 是否使用 batch_size 为第一维
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        # 3.输出层
        # 参数：输入的维度，输出的维度
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        x = self.embedding(x)  # [batch, seq_len, 128]
        _, (hidden, _) = self.lstm(x)  # hidden: [1, batch, 256]
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)  # [batch, num_classes]
        return out


# 5.训练模型，使用训练集train.csv
def train_model(train_texts_idx, word_to_idx):
    # 1.获取数据
    train_data, val_data, test_data = data_clean()
    # 2.构建词表
    # train_texts_idx, word_to_idx = build_vocab()
    # 取出所有的行的标签存为列表
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
    # 分类的种类数量
    num_class = len(set(labels))
    # 模型，使用GPU
    model = NewsClassifier(vocab_size, num_class).to(device)
    # 5.损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 6.循环训练模型
    epochs = 20
    train_losses = []
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
        # 保存训练损失
        train_losses.append(total_loss / iter_num)
    # 绘制训练损失曲线
    plt.plot(train_losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('../data/images/train_loss.png')
    plt.close()
    # 保存模型
    torch.save(model.state_dict(), '../model/news_lstm_i128_h512_e20_0p001.pth')


# 6.模型验证，使用验证集val.csv
def evaluate_model(word_to_idx):
    print(f'{'-' * 30}开始模型验证{'-' * 30}')
    # 1.加载数据
    train_data, val_data, test_data = data_clean()
    # 2.构建词表
    # train_texts_idx, word_to_idx = build_vocab()
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
    # 验证集的标签用自己的，但是标签种类数量需要使用train的，种类数量需要对齐
    # num_class = len(set(val_labels))
    num_class = len(set(train_data['label']))
    model = NewsClassifier(vocab_size, num_class).to(device)
    model_path = '../model/news_lstm_i128_h512_e20_0p001.pth'
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
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,  # 数字太多，先关掉
        fmt='d',
        cmap='Blues'
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Validation Confusion Matrix')
    # 保存图片
    os.makedirs('../data', exist_ok=True)
    plt.savefig('../data/images/confusion_matrix.png')
    plt.close()


# 7.模型测试，使用测试集test.csv
def model_test(word_to_idx):
    """
    模型测试（无真实标签）：
    1. 使用训练好的模型对 test.csv 进行预测
    2. 将预测的类别 id 与中文类别名保存为 CSV
    """
    print(f'{'-' * 30}开始模型测试{'-' * 30}')

    # 1. 加载数据
    train_data, _, test_data = data_clean()
    # lable全0占位，只取test列
    test_texts = test_data['text'].tolist()
    # 2. 文本数值化
    test_texts_idx = texts_to_indices(test_texts, word_to_idx)
    # 3. 构建 Dataset & DataLoader
    dummy_labels = [0] * len(test_texts)  # 占位，不参与计算
    test_dataset = NewsDataset(test_texts_idx, dummy_labels)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    # 4. 加载模型
    vocab_size = len(word_to_idx)
    num_class = len(set(train_data['label'].tolist()))
    model = NewsClassifier(vocab_size, num_class).to(device)
    model_path = '../model/news_lstm_i128_h512_e20_0p001.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 5. 加载类别映射
    label_map = load_label_map()
    # 6. 推理
    all_preds = []
    with torch.no_grad():
        for texts, _ in test_dataloader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    # 7. 保存结果
    result_df = pd.DataFrame({
        'text': test_texts,
        'label': all_preds
    })
    save_path = '../data/李贺童2201140218.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'预测结果已保存至：{save_path}')
    print(f'{'-' * 30}结束模型测试{'-' * 30}')


# 测试函数
def _test():
    # 0.数据验证
    data_validation()
    # 1.数据清洗
    # data_clean()
    # 2.分词，构建词表
    train_texts_idx, word_to_idx = build_vocab()
    # 5.训练模型
    train_model(train_texts_idx, word_to_idx)
    # 6.评估模型
    evaluate_model(word_to_idx)
    # 7.模型测试
    model_test(word_to_idx)


if __name__ == '__main__':
    _test()
    print(f'耗时：{time.time() - start:.2f}s')
