# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import os
from path import MODEL_PATH


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        pass

    def predict(self, question):
        '''
        模型预测返回结果
        :参数示例 question='老是胡思乱想该怎么办'
        :return: 返回预测结果格式具体说明如下： 最多返回三个生成式答案组成的list
        评估中该样本的F1值是返回三个生成式预测答案分别和真实答案计算F1值的最大值
        '''

        answer1 = '哈哈'
        answer2 = '嗯嗯'
        answer3 = question
        return [answer1, answer2, answer3]