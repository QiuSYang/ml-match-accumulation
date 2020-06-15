# -*- coding: utf-8 -*
import numpy
import os
import tensorflow as tf
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants
import numpy as np
from path import MODEL_PATH
import time
TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''

        return self.data.to_categorys()

    def predict_all(self, datas):

        ratings = []
        for data in datas:
            ratings.append(self.data.to_categorys())
        return ratings

    def save_model(self, session, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        if overwrite:
            self.delete_file(path)

        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path, name))
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    def batch_iter(self, x, y, batch_size=128):
        '''
        生成批次数据
        :param x: 所有验证数据x
        :param y: 所有验证数据y
        :param batch_size: 每批的大小
        :return: 返回分好批次的数据
        '''
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
