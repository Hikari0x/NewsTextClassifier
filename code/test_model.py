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

from news_class import (NewsClassifier, NewsDataset, build_vocab, data_clean,
                        data_validation)


def test_model():
    pass
