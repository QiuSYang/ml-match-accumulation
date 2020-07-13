# -*- coding: utf-8 -*
import os
import logging
from flyai.model.base import Base
from path import MODEL_PATH, DATA_PATH
import json
import math
import pandas as pd
import numpy as np
import torch
from net import Net
from main import data_transforms
from PIL import Image

MODEL_NAME = "best.pkl"
NUM_CLASSES = 40
BATCH_SIZE = 64  # 每次预测多少张图片
expand_border = [(-1, -1), (-1, 0), (-1, 1),
                              (0, -1), (0, 0), (0, 1),
                              (1, -1), (1, 0), (1, 1)]


def xy_to_image(xy_coordinates,
                image_height=300, image_width=300, grid_expand=False):
    y, x = [], []
    for xy in xy_coordinates:
        # 有所有坐标入列(横纵坐标)
        y.extend(xy[0])
        x.extend(xy[1])
    x_max, x_min = max(x), min(x)
    y_max, y_min = max(y), min(y)
    width, height = x_max - x_min + 1, y_max - y_min + 1
    if x_max > image_width or y_max > image_height:
        # 最大矩形框边界大于初始图像宽高
        image_width = max(x_max, y_max) + 10
        image_height = max(x_max, y_max) + 10

    d_width = math.floor((image_width - width) / 2.0 + 0.5)
    d_height = math.floor((image_height - height) / 2.0 + 0.5)

    x = np.array(x) + d_width  # 全部加上偏移量
    y = np.array(y) + d_height

    # 初始化图像数据
    image_data = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for i in range(len(x)):
        if grid_expand:
            for dx, dy in expand_border:
                y_i, x_i = y[i] + dy, x[i] + dx
                # 边界判定
                if y_i < 0:
                    y_i = 0
                elif y_i >= image_height:
                    y_i = image_height - 1
                if x_i < 0:
                    x_i = 0
                elif x_i >= image_width:
                    x_i = image_width - 1

                image_data[y_i, x_i, :] = 255
        else:
            image_data[y[i], x[i], :] = 255

    return image_data


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
        images = []
        for data in datas:
            json_path = self.dataset.predict_data(**data)[0]
            json_path = os.path.join(DATA_PATH, json_path)
            with open(json_path) as f:
                draw = json.load(f)
            image_numpy = xy_to_image(draw.get('drawing'), grid_expand=False)
            PIL_image = Image.fromarray(image_numpy)
            image = data_transforms['val'](PIL_image)
            images.append(image)

        print("start predict.")
        turn = math.ceil(len(images)/float(BATCH_SIZE))

        predicts = []
        with torch.no_grad():
            for index in range(turn):
                try:
                    # 这种访问方式, 超过最大索引, 也只会访问到最后一个元素
                    batch_images = images[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                except IndexError:
                    # 索引越界, 从当前位置取到最后
                    batch_images = image[index*BATCH_SIZE:]
                input_images = torch.stack(batch_images, dim=0)
                input_images = input_images.to(self.device)

                batch_predicts = self.model(input_images)
                predicts.extend(batch_predicts.data.cpu().numpy().tolist())

        labels = np.argmax(np.array(predicts), axis=1)

        return labels

    def load_model(self, path, model_name=MODEL_NAME):
        """加载模型的方法"""
        self.model.load_state_dict(torch.load(os.path.join(path, model_name)))
