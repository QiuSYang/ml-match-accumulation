# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
from keras.models import load_model
from path import MODEL_PATH, DATA_PATH
import json
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
KERAS_MODEL_NAME = "model.h5"

STROKE_COUNT = 196
all_classes = 40

def _stack_it(raw_strokes):
    """preprocess the string and make
    a standard Nx3 stroke vector"""
    # unwrap the list
    in_strokes = [(xi,yi,i)
                  for i,(x,y) in enumerate(raw_strokes)
                  for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1 # since 0 is no stroke
    # pad the strokes with zeros
    return pad_sequences(c_strokes.swapaxes(0, 1),
                         maxlen=STROKE_COUNT,
                         padding='post').swapaxes(0, 1)

class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_path = os.path.join(MODEL_PATH, KERAS_MODEL_NAME)

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):
        model = load_model(self.model_path)
        out_list = []
        for data in datas:
            json_path = self.dataset.predict_data(**data)[0]
            json_path = os.path.join(DATA_PATH, json_path)
            out_dict = {}
            with open(json_path) as f:
                draw = json.load(f)
            out_dict['drawing'] = draw['drawing']
            out_list.append(out_dict)
        out_df = pd.DataFrame(out_list) # 加载数据
        out_df['drawing'] = out_df['drawing'].map(_stack_it)
        sub_vec = np.stack(out_df['drawing'].values, 0)
        sub_pred = model.predict(sub_vec, verbose=True, batch_size=64)
        labels = np.argmax(sub_pred, axis=1)
        return labels

    '''
    保存模型的方法
    '''

    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        model.save(os.path.join(path, name))
