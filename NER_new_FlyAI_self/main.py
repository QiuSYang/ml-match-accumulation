# -*- coding: utf-8 -*
import os
import logging
import argparse
from flyai.dataset import Dataset
from model import Model
import datetime
import shutil
import kashgari
from kashgari.embeddings import BERTEmbedding
import kashgari.tasks.labeling as labeling
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

_logger = logging.getLogger(__name__)


# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# 模型操作辅助类
modelpp = Model(dataset)


class NerNet:
    """默认使用bert word2vector, BiLSTM+CRF进行实体抽取"""
    provides = ["entities"]

    # 默认参数
    defaults = {
        "bert_model_path": None,
        "sequence_length": "auto",
        "layer_nums": 4,
        "trainable": False,
        "labeling_model": "BiLSTM_CRF_Model",
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.2,
        "patience": 5,
        "factor": 0.5,  # factor of reduce learning late everytime
        "verbose": 1,
        "use_cudnn_cell": False
    }

    def __init__(self, config_dict={}):
        self.defaults.update(config_dict)

        bert_model_path = self.defaults.get('bert_model_path')
        sequence_length = self.defaults.get('sequence_length')
        layer_nums = self.defaults.get('layer_nums')
        trainable = self.defaults.get('trainable')
        use_cudnn_cell = self.defaults.get('use_cudnn_cell')

        # 设置是否使用cudnn进行加速训练，True为加速训练，false反之
        # 训练使用cudnn加速，那么inference也只能使用cudnn加速
        kashgari.config.use_cudnn_cell = use_cudnn_cell

        self.labeling_model = self.defaults.get('labeling_model')

        self.bert_embedding = BERTEmbedding(bert_model_path,
                                            task=kashgari.LABELING,
                                            layer_nums=layer_nums,
                                            trainable=trainable,
                                            sequence_length=sequence_length)

        labeling_model = eval("labeling." + self.labeling_model)
        # load 模型结构
        self.model = labeling_model(self.bert_embedding)

    def train(self, datasets):
        """ 训练函数
        :param datasets:
        :return:
        """
        epochs = self.defaults.get('epochs')
        batch_size = self.defaults.get('batch_size')
        validation_split = self.defaults.get('validation_split')
        patience = self.defaults.get('patience')
        # 训练学习率下降的改变因子
        factor = self.defaults.get('factor')
        verbose = self.defaults.get('verbose')

        # 设置回调状态
        checkpoint = ModelCheckpoint(
            'entity_weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            verbose=verbose)


