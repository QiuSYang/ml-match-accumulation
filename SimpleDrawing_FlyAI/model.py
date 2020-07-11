# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
# from keras.models import load_model
from path import MODEL_PATH, DATA_PATH
import json
import pandas as pd
import numpy as np
import torch
from net import Net
MODEL_NAME = "best.pkl"

NUM_CLASSES = 40


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = Net(num_classes=NUM_CLASSES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载以训练模型
        self.load_model(path=MODEL_PATH, model_name=MODEL_NAME)
        self.model.to(self.device)

    def predict_all(self, datas):
        """评估的时候会调用该方法实现评估得分"""
        self.model.eval()
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
        # out_df['drawing'] = out_df['drawing'].map(_stack_it)
        # sub_vec = np.stack(out_df['drawing'].values, 0)
        # sub_pred = model.predict(sub_vec, verbose=True, batch_size=64)
        # labels = np.argmax(sub_pred, axis=1)
        # return labels

    def load_model(self, path, model_name=MODEL_NAME):
        """加载模型的方法"""
        self.model.load_state_dict(torch.load(os.path.join(path, model_name)))
