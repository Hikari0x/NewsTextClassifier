# News Text Classification
A PyTorch-based news text classification project using LSTM for multi-class classification.
## 1. Project Background
News text classification is a common natural language processing task that aims to automatically categorize news articles into predefined topics such as finance, sports, or politics.
This project is implemented as a course assignment and focuses on building a complete text classification pipeline, including data preprocessing, model training, evaluation, and inference.

## 2. Project Structure

```
NewsTextClassifier
├── code							# 核心代码目录
│   ├── data_visualize.ipynb		# 数据分布与训练结果可视化分析
│   ├── news_class.py				# 主程序，包含数据处理、模型定义、训练、验证与测试流程
│   └── t01.py						# 辅助测试脚本
├── data							# 数据集与预测结果
│   ├── class.txt					# 类别编号与中文类别名称映射
│   ├── images						# 数据分析相关图片
│   ├── test.csv					# 测试集（无真实标签）
│   ├── train.csv					# 训练集
│   ├── val.csv						# 验证集
│   └── 李贺童2201140218.csv		# 测试集预测结果文件
├── doc								# 项目文档
│   ├── images						# Markdown图片目录
│   │   └── ProjectReport			# Markdown图片字目录，对应具体的文件
│   ├── ProjectReport.md			# 项目实验报告（Markdown）
│   ├── ProjectReport.pdf			# 项目实验报告（PDF）
│   └── 新闻分类分类任务书.pdf		# 项目任务书
├── model							# 模型存放目录
│   └── news_lstm_i128_h512_e20_0p001.pth	# 模型名称，规范参数命名
└── README.md						# 项目说明
```
## 3. Environmental dependence
- Python 3.13.9
- PyTorch
- NumPy
- pandas
## 4. Usage instructions
1. Clone the repository
## 5. Results
## 6. Version History
### v1.0 (2025-12-18)
- Implemented LSTM-based news text classifier
- Jieba tokenization + custom vocabulary
- Train / Validation / Test pipeline completed
- Test predictions exported to CSV with label names
- After multiple experiments, the current model parameters represent the optimal solution, with an accuracy rate of 89%.