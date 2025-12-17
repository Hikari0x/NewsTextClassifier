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
    # 0.数据验证
    train_data, val_data, test_data = data_validation()

    # 1.对train文本进行分词
    # 定义：总所有的分词列表
    train_texts = []
    # 遍历每行文本并分词，一行为一个分词列表
    for text in train_data['text']:
        participle = jieba.lcut(text)
        # print(participle)
        # 添加到总列表中
        train_texts.extend(participle)
    print(f'train_texts：{len(train_texts)}')
    # 2.去重，集合去重转列表
    train_texts_unique = list(set(train_texts))
    print(f'train_texts_unique：{len(train_texts_unique)}')
    # 3.词频，统计每个词出现的次数
    word_counter = Counter(train_texts)
    print(f'word_counter：{len(word_counter)}')
    # 4.构建词表，全保留
    word_to_idx= {word: idx for idx, word in enumerate(train_texts_unique)}
    print(f'word_to_idx：{len(word_to_idx)},{type(word_to_idx)}')


# 3.数据集封装
class NewsDataset(Dataset):
    pass


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


if __name__ == '__main__':
    main()
