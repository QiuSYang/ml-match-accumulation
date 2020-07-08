# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""

import argparse
from flyai.dataset import Dataset
from model import Model, _stack_it
from path import MODEL_PATH, DATA_PATH
import os
import json
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
下载模版之后需要把当前样例项目的app.yaml复制过去哦～
第一次使用请看项目中的：FLYAI项目详细文档.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
进入Dataset类中可查看方法说明
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
x_train, y_train, x_val, y_val = dataset.get_all_processor_data()

all_classes = 40

def get_xy(json_path_list, label_list):
    # ['draws/draw_490253.json', ...] [ 9, ...]
    out_list = []
    for i in range(len(json_path_list)):
        out_dict = {}
        json_path = os.path.join(DATA_PATH, json_path_list[i])
        with open(json_path) as f:
            draw = json.load(f)
        out_dict['drawing'] = draw['drawing']
        out_dict['label'] = label_list[i]
        out_list.append(out_dict)
    out_df = pd.DataFrame(out_list) # 加载数据
    out_df['drawing'] = out_df['drawing'].map(_stack_it)
    return out_df


train_df = get_xy(x_train, y_train)
valid_df = get_xy(x_val, y_val)

def get_Xy(in_df):
    X = np.stack(in_df['drawing'], 0)
    y = np.zeros((X.shape[0], all_classes))
    for i, index in enumerate(in_df['label'].values):
        y[i][index] = 1
    return X, y

train_X, train_y = get_Xy(train_df)
valid_X, valid_y = get_Xy(valid_df)
print(train_X.shape, train_y.shape)
print(valid_X.shape, valid_y.shape)
'''
实现自己的网络结构
'''

stroke_read_model = Sequential()
stroke_read_model.add(BatchNormalization(input_shape=(None,)+train_X.shape[2:]))
stroke_read_model.add(Conv1D(48, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(64, (5,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Conv1D(96, (3,)))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = True))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(LSTM(128, return_sequences = False))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(512))
stroke_read_model.add(Dropout(0.3))
stroke_read_model.add(Dense(all_classes, activation = 'softmax'))
stroke_read_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
stroke_read_model.summary()

'''
dataset.get_step() 获取数据的总迭代次数
'''
KERAS_MODEL_NAME = "model.h5"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, KERAS_MODEL_NAME), monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only=False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
                                   verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=5)
callbacks_list = [checkpoint, early, reduceLROnPlat]
stroke_read_model.fit(x=train_X, y=train_y,
                      validation_data=(valid_X, valid_y),
                      batch_size=2,
                      epochs=1,
                      callbacks=callbacks_list)
