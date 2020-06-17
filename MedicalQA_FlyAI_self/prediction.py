# -*- coding: utf-8 -*
import os
import tensorflow as tf
from path import DATA_PATH, MODEL_PATH
from data_helper import load_dict, text2id
from modelNet import get_tensor_name
from flyai.framework import FlyAI


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        self.sour2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalQA/ask_fr.dict'))
        self.targ2id, self.id2targ = load_dict(os.path.join(DATA_PATH, 'MedicalQA/ans_fr.dict'))
        self.session = tf.Session()
        tf.saved_model.loader.load(self.session,
                                   [tf.saved_model.tag_constants.SERVING], os.path.join(MODEL_PATH, 'best'))
        self.source_input = self.session.graph.get_tensor_by_name(get_tensor_name('inputs'))
        self.source_seq_len = self.session.graph.get_tensor_by_name(get_tensor_name('source_sequence_length'))
        self.target_seq_length = self.session.graph.get_tensor_by_name(get_tensor_name('target_sequence_length'))
        self.predictions = self.session.graph.get_tensor_by_name(get_tensor_name('predictions'))

    def predict(self, department, title, ask):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"department": "心血管科", "title": "心率为72bpm是正常的吗",
                                    "ask": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"answer": "心脏不舒服一般是..."}
        '''
        sour_x, sour_len = text2id(title+ask, self.sour2id)

        feed_dict = {self.source_input: [sour_x],
                     self.source_seq_len: [sour_len],
                     self.target_seq_length: [5] * len([sour_x])}
        predict = self.session.run(self.predictions, feed_dict=feed_dict)

        result_list = list()
        for item in predict[0]:
            if item != self.targ2id['_eos_']:
                result_list.append(self.id2targ[item])
        result = ''.join(result_list)

        return {'answer': result}