# -*- coding: utf-8 -*
import os
import path
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


class NerNet(object):
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

        self.bert_model_path = self.defaults.get('bert_model_path')
        self.sequence_length = self.defaults.get('sequence_length')
        self.layer_nums = self.defaults.get('layer_nums')
        self.trainable = self.defaults.get('trainable')
        self.use_cudnn_cell = self.defaults.get('use_cudnn_cell')

        # 设置是否使用cudnn进行加速训练，True为加速训练，false反之
        # 训练使用cudnn加速，那么inference也只能使用cudnn加速
        kashgari.config.use_cudnn_cell = self.use_cudnn_cell

        self.labeling_model = self.defaults.get('labeling_model')

        # self.bert_embedding = BERTEmbedding(bert_model_path,
        #                                     task=kashgari.LABELING,
        #                                     layer_nums=layer_nums,
        #                                     trainable=trainable,
        #                                     sequence_length=sequence_length)
        #
        # labeling_model = eval("labeling." + self.labeling_model)
        # # load 模型结构
        # self.model = labeling_model(self.bert_embedding)

        self.bert_embedding = None
        self.model = None

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

        x_train, y_train, x_val, y_val = self.get_dataset(datasets)

        self.bert_embedding = BERTEmbedding(self.bert_model_path,
                                            task=kashgari.LABELING,
                                            layer_nums=self.layer_nums,
                                            trainable=self.trainable,
                                            sequence_length=self.sequence_length)

        labeling_model = eval("labeling." + self.labeling_model)
        # load 模型结构
        self.model = labeling_model(self.bert_embedding)

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
        log_dir = path.LOG_PATH
        if os.path.exists(log_dir):
            # 路径已经存在删除路径
            shutil.rmtree(log_dir)
        tensor_board = TensorBoard(
            log_dir=log_dir,
            batch_size=batch_size)

        # 训练模型
        self.model.fit(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr, tensor_board]
        )

    def get_dataset(self, datasets):
        x_train, y_train, x_val, y_val = dataset.get_all_data()
        x_train_list, y_train_list = [], []
        x_val_list, y_val_list = [], []
        for i in range(len(x_train)):
            x_train_list.append(x_train[i].get('source').strip().split(' '))
            y_train_list.append(y_train[i].get('target').strip().split(' '))
        for i in range(len(x_val)):
            x_val_list.append(x_val[i].get('source').strip().split(' '))
            y_val_list.append(y_val[i].get('target').strip().split(' '))

        self.sequence_length = max(max(map(len, x_train_list)), max(map(len, x_val_list)))

        return x_train_list, y_train_list, x_val_list, y_val_list

    def get_pre_train_model(self):
        # 必须使用该方法下载模型，然后加载
        from flyai.utils import remote_helper
        path = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_wwm_ext_L-12_H-768_A-12.zip')

        return path


if __name__ == "__main__":
    # 超参
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
    args = parser.parse_args()
    # 数据获取辅助类
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    # 模型操作辅助类
    modelpp = Model(dataset)

    config = {"bert_model_path": path.BERT_PATH,
              "sequence_length": 512,
              "batch_size": 32,
              "epochs": args.EPOCHS,
              "use_cudnn_cell": True}

    net = NerNet(config_dict=config)
    net.get_pre_train_model()

    net.train(dataset)

    modelpp.save_model(ner_net=net)

