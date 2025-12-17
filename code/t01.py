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

from news_class import (NewsClassifier, NewsDataset, build_vocab, data_clean, data_validation)


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


if __name__ == '__main__':
    # build_vocab()
    dm01()
    dm02()
