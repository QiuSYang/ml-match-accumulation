# -*- coding: utf-8 -*-

import os
import time
import argparse
import random
import numpy as np
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from sklearn.model_selection import train_test_split
from path import DATA_PATH, MODEL_PATH, MODEL_PATH_BEST, PRE_MODEL_PATH
import pandas as pd
from data_helper import *
import transformers
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config
import torch
from torch.utils.data import DataLoader

PAD = "[PAD]"


class Main(FlyAI):
    def __init__(self, args, pad_id=0):
        self.args = args
        self.args.cuda = torch.cuda.is_available()
        # 为CPU设置种子用于生成随机数，以使得结果是确定的
        # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        # 当得到比较好的结果时我们通常希望这个结果是可以复现
        # 设置GPU
        self.device = 'cuda' if args.cuda else 'cpu'
        if self.args.seed:
            self.set_random_seed()

        self.pad_id = pad_id
        # 构建模型, n_ctx 上下文的最大长度
        self.model, self.n_ctx = self.build_model()

        # 设置multi-GPU计算
        self.model.to(self.device)
        self.multi_gpu = False
        if self.args.cuda and torch.cuda.device_count() > 1:
            print('use GPUs to training.')
            self.model = torch.nn.DataParallel(self.model)
            self.multi_gpu = True

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
        self.steps_per_epoch = int(len(self.train_dataset_ids) / self.args.BATCH)

        print('=='*8+'数据处理完成！'+'=='*8)

    def build_model(self):
        """创建GPT-2生成模型
        """
        # 使用bert tokenizer # 初始化tokenizer
        self.tokenizer = BertTokenizer(vocab_file=self.args.vocab_path)
        # temp = self.tokenizer.convert_tokens_to_ids('')
        # print(self.tokenizer.convert_ids_to_tokens(temp))
        # tokenizer的字典大小
        self.vocab_size = len(self.tokenizer)

        self.pad_id = self.tokenizer.convert_tokens_to_ids(PAD)

        if self.args.pretrained_model:
            # 如果指定了预训练的GPT2模型
            model = GPT2LMHeadModel.from_pretrained(self.args.pretrained_model)
        else:
            # 若没有指定预训练模型，则初始化模型
            model_config = GPT2Config(self.args.model_config)
            model = GPT2LMHeadModel(config=model_config)

        # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
        model.resize_token_embeddings(self.vocab_size)

        print('model config:\n{}'.format(model.config.to_json_string()))

        return model, model.config.to_dict().get("n_ctx")

    def train(self):
        """模型训练"""
        train_dataset = MedicalQADataset(self.train_dataset_ids)
        train_data_loader = DataLoader(train_dataset, batch_size=self.args.BATCH,
                                       shuffle=True, collate_fn=self.collate_fn)

        self.model.train()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.EPOCHS = args.max_steps // (len(train_data_loader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_data_loader) // self.args.gradient_accumulation_steps * self.args.EPOCHS
        self.args.warmup_steps = int(t_total * self.args.warmup_proportion)

        # # Prepare optimizer and schedule (linear warmup and decay)
        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': args.weight_decay},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0}]

        optimizer = transformers.AdamW(self.model.parameters(), lr=self.args.lr, correct_bias=True)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=self.args.warmup_steps,
                                                                 num_training_steps=-1)

        print("Training start.")
        # 用于统计每次梯度累计的loss
        running_loss = 0
        # 统计一共训练了多少个step
        overall_step = 0
        # 记录 out of memory的次数
        oom_time = 0
        # 记录最佳评估loss和accuracy
        best_loss = np.inf
        best_accuracy = -1
        for epoch in range(self.args.EPOCHS):
            epoch_start_time = time.clock()
            for batch_idx, input_ids in enumerate(train_data_loader):
                # 注意：GPT2模型的forward()函数, 是对于给定的context, 生成一个token，而不是生成一串token
                # GPT2Model的输入为n个token_id时, 输出也是n个hidden_state, 使用第n个hidden_state预测第n+1个token
                input_ids = input_ids.to(self.device)
                # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
                try:
                    outputs = self.model(input_ids=input_ids)
                    loss, accuracy = self.calculate_loss_and_accuracy(outputs,
                                                                      labels=input_ids,
                                                                      device=self.device)
                    if self.multi_gpu:
                        loss = loss.mean()
                        accuracy = accuracy.mean()
                    if self.args.gradient_accumulation > 1:
                        loss = loss / self.args.gradient_accumulation
                        accuracy = accuracy / self.args.gradient_accumulation
                    loss.backward()
                    # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # 进行一定step的梯度累计之后，更新参数
                    if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                        running_loss += loss.item()
                        # update parameter
                        optimizer.step()
                        # 进行 warn up
                        scheduler.step()
                        # 清空梯度信息
                        optimizer.zero_grad()
                        overall_step += 1
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        print("WARNING: ran out of memory,times: {}.".format(oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        print(str(exception))
                        raise exception
            print('Saving model for epoch {}'.format(epoch + 1))
            model_path = os.path.join(MODEL_PATH, 'model_epoch_{}'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(model_path)
            print('The epoch {} finished.'.format(epoch + 1))
            epoch_finish_time = time.clock()
            print('Time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

            # 每个epoch评估一次模型, 交叉验证
            eval_loss, eval_accuracy = self.evaluate()
            if eval_loss < best_loss:
                best_loss = eval_loss
                if not os.path.exists(MODEL_PATH_BEST):
                    os.mkdir(MODEL_PATH_BEST)
                print('Save best model.')
                model_to_save.save_pretrained(MODEL_PATH_BEST)

        print('Training finished.')

    def evaluate(self):
        """模型评估"""
        test_dataset = MedicalQADataset(self.test_dataset_ids)
        test_data_loader = DataLoader(test_dataset, batch_size=self.args.BATCH,
                                      shuffle=True, collate_fn=self.collate_fn)

        self.model.eval()
        print('start evaluating model.')
        eval_loss_total = 0.0
        eval_accuracy_total = 0.0
        with torch.no_grad():
            for batch_idx, input_ids in enumerate(test_data_loader):
                input_ids.to(self.device)
                outputs = self.model(input_ids=input_ids)
                loss, accuracy = self.calculate_loss_and_accuracy(outputs,
                                                                  labels=input_ids,
                                                                  device=self.device)
                eval_loss_total += loss
                eval_accuracy_total += accuracy
                if self.multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if self.args.gradient_accumulation > 1:
                    loss = loss / self.args.gradient_accumulation
                    accuracy = accuracy / self.args.gradient_accumulation
                print("evaluate batch {} ,loss {} ,accuracy {}.".format(batch_idx, loss, accuracy))
            print("finishing evaluating.")

        # 返回平均验证loss和accuracy
        return eval_loss_total/len(test_data_loader), eval_accuracy_total/len(test_data_loader)

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
        """计算该batch中的所有sample的最长的input,
        并且将其他input的长度向其对齐
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


def get_pre_train_model():
    """加载预训练模型"""
    # 必须使用该方法下载模型，然后加载
    from flyai.utils import remote_helper
    # 下载到项目中的data/input/文件夹，默认会自动解压，具体文件路径可以下之后查看使用
    path = remote_helper.get_remote_date('https://www.flyai.com/m/gpt-2-chinese-wiki.zip')

    return path


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
    parser.add_argument('--lr', default=1.5e-4,
                        type=float, help='学习率')
    parser.add_argument('--log_step', default=1,
                        type=int, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation',
                        default=1, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0,
                        type=float, help='梯度剪切参数')
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=None,
                        help='设置种子用于生成随机数，以使得训练的结果是确定的')

    args = parser.parse_args()

    pre_train_model_path = get_pre_train_model()

    pre_train_model_path = PRE_MODEL_PATH

    # 更新预训练模型参数
    args.model_config = os.path.join(pre_train_model_path, 'config.json')
    args.vocab_path = os.path.join(pre_train_model_path, 'vocab_small.txt')
    args.pretrained_model = pre_train_model_path

    main = Main(args)
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)
