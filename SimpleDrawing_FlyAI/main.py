# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import os
import logging
import json
import math
import cv2
import argparse
import flyai
from flyai.dataset import Dataset as fly_Dataset
from path import MODEL_PATH, DATA_PATH
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from net import Net

"""
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
"""

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
_logger = logging.getLogger(__name__)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CustomDataset(Dataset):
    def __init__(self, json_path_list, label_list, data_type='val',
                 image_height=300, image_width=300, grid_expand=False):
        """
        :param json_path_list: 所有文件列表, json文件包含一个字段‘drawing’, 包含所有绘制点x, y坐标点
        :param label_list: 每个json文件对应的label
        :param image_height: 坐标在图像上表示的图像高度
        :param image_width: 坐标在图像上表示的图像宽度
        :param grid_expand: 是否需要将对应单点8邻域扩张
        """
        self.json_path_list = json_path_list
        self.label_list = label_list
        self.grid_expand = grid_expand
        self.expand_border = [(-1, -1), (-1, 0), (-1, 1),
                              (0, -1), (0, 0), (0, 1),
                              (1, -1), (1, 0), (1, 1)]
        self.data_type = data_type
        self.image_height = image_height
        self.image_width = image_width
        self.image_show = False

    def __getitem__(self, index):
        json_path = os.path.join(DATA_PATH, self.json_path_list[index])
        with open(json_path, mode='r', encoding='utf-8') as f:
            # load json file content
            xy_coordinates_dict = json.load(f)
        # 将坐标转为图像数据
        image_data = self.xy_to_image(xy_coordinates=xy_coordinates_dict.get('drawing'))
        PIL_image = Image.fromarray(image_data)
        image = data_transforms[self.data_type](PIL_image)
        label = self.label_list[index]

        return image, label

    def __len__(self):
        return len(self.json_path_list)

    def xy_to_image(self, xy_coordinates):
        y, x = [], []
        for xy in xy_coordinates:
            # 有所有坐标入列(横纵坐标)
            y.extend(xy[0])
            x.extend(xy[1])
        # 寻找边界
        x_max, x_min = max(x), min(x)
        y_max, y_min = max(y), min(y)
        width, height = x_max - x_min + 1, y_max - y_min + 1
        if x_max > self.image_width or y_max > self.image_height:
            # 最大矩形框边界大于初始图像宽高
            self.image_width = math.ceil(max(x_max, y_max)/10.0)*10 + 10
            self.image_height = math.ceil(max(x_max, y_max)/10.0)*10 + 10

        d_width = math.floor((self.image_width - width) / 2.0 + 0.5)
        d_height = math.floor((self.image_height - height) / 2.0 + 0.5)

        x = np.array(x) + d_width  # 全部加上偏移量
        y = np.array(y) + d_height

        # 初始化图像数据
        image_data = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        for i in range(len(x)):
            if self.grid_expand:
                for dx, dy in self.expand_border:
                    y_i, x_i = y[i] + dy, x[i] + dx
                    # 边界判定
                    if y_i < 0:
                        y_i = 0
                    elif y_i >= self.image_height:
                        y_i = self.image_height - 1
                    if x_i < 0:
                        x_i = 0
                    elif x_i >= self.image_width:
                        x_i = self.image_width - 1

                    image_data[y_i, x_i, :] = 255
            else:
                image_data[y[i], x[i], :] = 255

        if self.image_show:
            cv2.imshow('image_data', image_data)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return image_data


class Main(object):
    def __init__(self, args):
        """
        :param args: 超参数
        """
        self.args = args
        """
        flyai库中的提供的数据处理方法
        传入整个数据训练多少轮，每批次批大小
        进入Dataset类中可查看方法说明
        """
        # self.dataset = fly_Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH)
        self.dataset = fly_Dataset()
        self.x_train, self.y_train, self.x_val, self.y_val = self.dataset.get_all_processor_data()
        # 初始化模型
        self.model = Net(num_classes=self.args.NUM_CLASSES)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()

    def get_loader(self, json_path_list, label_list,
                   image_height=300, image_width=300, grid_expand=False):
        """生产data loader"""
        dataset = CustomDataset(json_path_list, label_list,
                                image_height=image_height, image_width=image_width,
                                grid_expand=grid_expand)

        # temp = dataset[0]

        return DataLoader(dataset,
                          batch_size=self.args.BATCH,
                          num_workers=0,
                          shuffle=True)

    def train(self):
        """模型训练"""
        train_data_loader = self.get_loader(self.x_train, self.y_train, grid_expand=False)
        self.model.train()

        # 设置优化器
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        _logger.info("start training.")
        # 记录最佳评估loss和accuracy
        best_loss = np.inf
        best_accuracy = -1
        for epoch in range(self.args.EPOCHS):
            running_loss = 0.0
            running_accuracy = 0
            for batch_idx, (images, labels) in enumerate(train_data_loader):
                # 将数据拷贝到GPU上
                images = images.to(self.device)
                # tensor 类型转换
                labels = torch.as_tensor(labels, dtype=torch.long)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.model(images)
                _, predicts = torch.max(outputs, dim=1)
                # 计算损失
                loss = self.criterion(outputs, labels)

                # backward
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_accuracy += torch.sum(predicts == labels.data)

            # scheduler step update learning ratio
            scheduler.step()
            _logger.info("every epoch loss: {}, accuracy: {}.".format(running_loss,
                                                                      running_accuracy))

            # 进行评估, 获取最佳评估损失，保存best模型
            evaluate_loss, evaluate_accuracy = self.evaluate()
            _logger.info("every epoch evaluate loss: {}, accuracy: {}.".format(evaluate_loss,
                                                                               evaluate_accuracy))
            if evaluate_loss < best_loss:
                _logger.info("save best model.")
                best_loss = evaluate_loss
                self.save_model(MODEL_PATH, '{}.pkl'.format('best'))

        _logger.info("finish trained.")

        return self.model

    def evaluate(self):
        """模型评估"""
        val_data_loader = self.get_loader(self.x_val, self.y_val, grid_expand=False)
        self.model.eval()

        _logger.info("start evaluating.")
        eval_loss_total = 0.0
        eval_accuracy_total = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_data_loader):
                # 将数据拷贝到GPU上
                images = images.to(self.device)
                # tensor 类型转换
                labels = torch.as_tensor(labels, dtype=torch.long)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, predicts = torch.max(outputs, dim=1)
                # 计算损失
                loss = self.criterion(outputs, labels)

                # statistics
                eval_loss_total += loss.item() * images.size(0)
                eval_accuracy_total += torch.sum(predicts == labels.data)

        _logger.info("finish evaluated.")

        return eval_loss_total, eval_accuracy_total

    def save_model(self, model_path, model_name):
        """保存模型"""
        if not os.path.exists(model_path):
            # 路径不存在，则创建路径
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), os.path.join(model_path, model_name))


if __name__ == "__main__":
    # 项目的超参
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=32, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
    parser.add_argument("-n", "--NUM_CLASSES", default=40, type=int, help="number of categories")
    args = parser.parse_args()

    main = Main(args)
    main.train()
