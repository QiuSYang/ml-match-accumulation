# -*- coding: utf-8 -*
import sys
import os

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 预训练模型路径
PRE_MODEL_PATH = os.path.join(sys.path[0], 'data', 'input', 'model')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
MODEL_PATH_BEST = os.path.join(sys.path[0], 'data', 'output', 'model', 'best')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 词汇表路径
VOCAB_PATH = os.path.join(sys.path[0], 'data', 'input', 'model', 'vocab_small.txt')


class PredictConfig(object):
    """预测参数设置"""
    temperature = 1.0
    repetition_penalty = 1.0
    topk = 8
    topp = 0.0
    max_len = 100
