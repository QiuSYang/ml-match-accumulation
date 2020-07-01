# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from sklearn.model_selection import train_test_split
from path import DATA_PATH, MODEL_PATH
import pandas as pd
from modelNet import *
from data_helper import *
import transformers
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config
import torch

PAD = "[PAD]"


class Main(FlyAI):
    def __init__(self, args, pad_id=0):
        self.args = args
        self.args.cuda = torch.cuda.is_available()
        # 为CPU设置种子用于生成随机数，以使得结果是确定的
        # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        # 当得到比较好的结果时我们通常希望这个结果是可以复现
        if self.args.seed:
            self.set_random_seed()

        self.pad_id = pad_id
        # 构建模型, n_ctx 上下文的最大长度
        self.model, self.n_ctx = self.build_model()

    def set_random_seed(self):
        """
        设置训练的随机种子
        """
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)

        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        # 数据下载解压路径：./data/input/MedicalQA
        data_helper.download_from_ids("MedicalQA")
        print('=='*8+'数据下载完成！'+'=='*8)

    def deal_with_data(self):
        # 加载数据集
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalQA/train.csv'))
        # 划分训练集、测试集
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.001, random_state=6, shuffle=True)

        # 获取训练数据的ids
        self.train_dataset_ids = get_sequence_ids(self.train_data, tokenizer=self.tokenizer, n_ctx=self.n_ctx)
        # 获取测试数据的ids
        self.test_dataset_ids = get_sequence_ids(self.test_data, tokenizer=self.tokenizer, n_ctx=self.n_ctx)
        # 计算每个epoch的batch数
        self.steps_per_epoch = int(len(self.train_dataset_ids) / args.BATCH)

        print('=='*8+'数据处理完成！'+'=='*8)

    def build_model(self):
        """创建GPT-2生成模型
        """
        # 使用bert tokenizer # 初始化tokenizer
        self.tokenizer = BertTokenizer(vocab_file=args.vocab_path)
        # tokenizer的字典大小
        self.vocab_size = len(self.tokenizer)

        self.pad_id = self.tokenizer.convert_tokens_to_ids(PAD)

        if args.pretrained_model:
            # 如果指定了预训练的GPT2模型
            model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        else:
            # 若没有指定预训练模型，则初始化模型
            model_config = GPT2Config(args.model_config)
            model = GPT2LMHeadModel(config=model_config)

        # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
        model.resize_token_embeddings(self.vocab_size)

        print('model config:\n{}'.format(model.config.to_json_string()))

        return model, model.config.to_dict().get("n_ctx")

    def train(self):
        pass

    def evaluate(self):
        pass

    def calculate_loss_and_accuracy(self, outputs, labels, device):
        """
        计算非pad_id的平均loss和准确率
        :param outputs:
        :param labels:
        :param device:
        :return:
        """
        logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
        # 用前n-1个token，预测出第n个token
        # 用第i个token的prediction_score用来预测第i+1个token。
        # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，
        # shift_labels表示第[1，n-1]的label
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(device)

        # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id, reduction='sum')
        temp = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

        # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]
        _, preds = shift_logits.max(dim=-1)

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        not_ignore = shift_labels.ne(self.pad_id)
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets

        return loss, accuracy

    def collate_fn(self, batch):
        """
        计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
        :param batch:
        :return:
        """
        input_ids = []
        btc_size = len(batch)
        max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
        # 计算该batch中input的最大长度
        for btc_idx in range(btc_size):
            if max_input_len < len(batch[btc_idx]):
                max_input_len = len(batch[btc_idx])
        # 使用pad_id对小于max_input_len的input_id进行补全
        for btc_idx in range(btc_size):
            input_len = len(batch[btc_idx])
            input_ids.append(batch[btc_idx])
            input_ids[btc_idx].extend([self.pad_id] * (max_input_len - input_len))
        return torch.tensor(input_ids, dtype=torch.long)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
    parser.add_argument('--model_config',
                        default='GPT2_wiki/config.json',
                        type=str, help='选择模型参数')
    parser.add_argument('--vocab_path',
                        default='vocabulary/vocab_small.txt',
                        type=str, help='选择词库')
    parser.add_argument('--pretrained_model',
                        default='GPT2_wiki/',
                        type=str, help='预训练的GPT2模型的路径')
    args = parser.parse_args()

    main = Main(args)
    # main.download_data()
    main.deal_with_data()
    # main.train()

    exit(0)
