import json
import os
import random
import re
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



# 0.数据验证
def data_validation():
    print(f'{'-' * 30}开始数据检查{'-' * 30}')

    # 1. 读取数据
    train_data = pd.read_csv('../data/train.csv')
    val_data = pd.read_csv('../data/val.csv')
    test_data = pd.read_csv('../data/test.csv')

    # 2. 基本信息
    print('[train_data.shape]', train_data.shape)
    print('[val_data.shape]', val_data.shape)
    print('[test_data.shape]', test_data.shape)

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

    # 返回数据
    return train_data, val_data, test_data


# 1.数据清洗
def data_clean():
    pass


# 2.分词，构建词表
def build_vocab():
    # 0.数据验证,获取数据
    train_data, val_data, test_data = data_validation()

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

    # 返回结果
    return train_texts_idx, word_to_idx


def texts_to_indices(texts, word_to_idx):
    texts_idx = []
    for text in texts:
        words = jieba.lcut(text)
        texts_idx.append([word_to_idx[word] for word in words])
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


# 4.构建神经网络模型
class NewsClassifier(nn.Module):
    pass


# 5.训练模型，使用训练集train.csv
def train_model():
    pass


# 6.模型验证，使用验证集val.csv
def evaluate_model():
    pass


# 7.模型测试，使用测试集test.csv
# def test_model():
#     pass


# 8.数据可视化
def data_visualize():
    pass


# 主函数
def main():
    # 0.数据验证
    # data_validation()

    # 1.数据清洗
    # data_clean()

    # 2.分词，构建词表
    build_vocab()

    #


if __name__ == '__main__':
    main()
