# -*- coding: utf-8 -*
import numpy
import os
import tensorflow as tf
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants
import numpy as np
import path
import time
import shutil
import kashgari


class Model(Base):
    def __init__(self, data):
        self.data = data

    def predict(self, **data):
        '''
        使用模型
        :param data: 模型的输入参数
        :return:
        '''
        # load 已经训练好的模型
        load_model = kashgari.utils.load_model(model_path=path.MODEL_PATH)

        data_list = [data.get('source').strip().split(' ')]
        single_predict_result = load_model.predict(data_list)

        return single_predict_result[0]

    def predict_all(self, datas):
        """ 使用模型连续预测
        :param datas:
        :return:
        """
        # load 已经训练好的模型
        load_model = kashgari.utils.load_model(model_path=path.MODEL_PATH)

        test_data_list = []
        for data in datas:
            test_data_list.append(data.get('source').strip().split(' '))

        predict_result = load_model.predict(test_data_list)

        return predict_result

    def save_model(self, ner_net):
        """
        :param ner_net: NerNet类对象
        :return:
        """
        ner_net.model.save(path.MODEL_PATH)

        remove_file = os.path.join(path.MODEL_PATH, 'model_weights.h5')
        os.remove(remove_file)
        # 将损失最小的模型拷贝到对应路径
        shutil.move('entity_weights.h5', path.MODEL_PATH)
        os.rename(os.path.join(path.MODEL_PATH, 'entity_weights.h5'), os.path.join(path.MODEL_PATH, 'model_weights.h5'))

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
