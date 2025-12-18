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

from news_class import (NewsClassifier, NewsDataset, build_vocab, data_clean, data_validation,evaluate_model)


# def test_model():
#     pass

def dm01():
    """
    val_data的分类种类数量
    :return:
    """
    val_data = pd.read_csv('../data/val.csv')
    print(f'val_data{list(set(val_data['label']))}')


def dm02():
    """
    train_data的分类种类数量
    :return:
    """
    train_data = pd.read_csv('../data/train.csv')
    print(f'train_data{list(set(train_data["label"]))}')
    labels = train_data['label'].tolist()
    num_class = len(set(train_data['label']))
    print(f'num_class：{num_class}')


def dm03():
    """
    数据可视化
    :return:
    """
    train_data, val_data, test_data = data_clean()
    train_texts_idx, word_to_idx = build_vocab()
    evaluate_model(word_to_idx)
    # 类别分布图
    train_data['label'].value_counts().sort_index().plot(
        kind='bar',
        title='Train Label Distribution'
    )
    plt.savefig('../data/images/label_dist.png')
    plt.close()


def dm04():
    data_validation()


if __name__ == '__main__':
    # build_vocab()
    # dm01()
    # dm02()
    # dm03()
    dm04()