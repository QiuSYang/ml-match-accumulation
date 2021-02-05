"""
# 解决图像分类问题类不平衡问题优化 trick
# 参考链接：https://www.flyai.com/n/51410
# 数据增强：采用随机裁剪，随机旋转，随机翻转，随机擦除
"""
import os
import numpy as np
import logging
import torch
from torch import nn
import torchvision

logger = logging.getLogger(__name__)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """数据预处理：每个类别的样本个数不一样，
    故采用Imbalanced Dataset Sampler调整每个类别的权重最后使得整个样本群每个类别平衡。"""
    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            try:
                label = self._get_label(dataset, idx)
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
            except:
                pass

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return np.argmax(dataset.labels[idx])

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class CrossEntropyLabelSmooth(nn.Module):
    """基于CrossEntropy进行Label Smooth操作"""
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
