import json
import os
import random
import re

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

    # 验证数据读取是否正常
    print('读取训练集train.csv')
    train_data = pd.read_csv('../data/train.csv')
    print(train_data.head())
    print(train_data.shape)
    print('训练集train.csv读取成功')
    print('读取验证集val.csv')
    test_data = pd.read_csv('../data/val.csv')
    print(test_data.head())
    print(test_data.shape)
    print('验证集val.csv读取成功')
    print('读取测试集test.csv')
    test_data = pd.read_csv('../data/test.csv')
    print(test_data.head())
    print(test_data.shape)
    print('测试集test.csv读取成功')

    # 检查数据是否能够转成DataFream
    print(f'{'-' * 30}数据检查结束{'-' * 30}')


# 1.数据清洗
def data_clean():
    pass


# 2.分词，构建词表
def build_vocab():
    pass


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
    data_validation()

    # 1.数据清洗
    # data_clean()

    # 2.分词，构建词表
    # build_vocab()


if __name__ == '__main__':
    main()