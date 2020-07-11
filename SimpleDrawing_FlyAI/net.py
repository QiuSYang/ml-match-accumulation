# -*- coding: utf-8 -*
from torch import nn
from torchvision import models


class Net(nn.Module):
    def __init__(self, num_classes, model_name='resnet18'):
        super(Net, self).__init__()
        # 使用resnet网络做迁移学习(获取迁移学习模型对象)
        resnet = eval("models." + model_name)
        self.model = resnet(pretrained=True)
        # 获取全连接层的输入
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, inputs):
        outputs = self.model(inputs)

        return outputs
