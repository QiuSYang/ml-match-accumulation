# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

from transformers import (
    BertTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)

from path import HyperParametersConfig, DataArguments, ModelArguments
from custom_model import CustomGPTGeneration
from dataset import PsychologicalQADataset

logger = logging.getLogger(__name__)

"""
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
"""


class Main(FlyAI):
    """
    项目中必须继承FlyAI类，否则线上运行会报错。
    """
    def __init__(self, args):
        self.args = args

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("PsychologicalQA")

    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        pass

    def train(self):
        """
        训练模型，必须实现此方法
        :return:
        """
        config = HyperParametersConfig(epochs=self.args.EPOCHS,
                                       batch_size=self.args.BATCH)

        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        # config_dict = HyperParametersConfig().__dict__
        # print(config_dict)
        model_args, data_args, training_args = parser.parse_dict(config.__dict__)

        logger.info("Load pre-training model.")
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
        model = CustomGPTGeneration.from_pretrained(model_args.model_name_or_path)

        # Get datasets
        logger.info("Loading dataset.")
        train_dataset = PsychologicalQADataset(data_args.dataset_path,
                                               tokenizer=tokenizer,
                                               max_sequence_len=data_args.max_sequence_len)

        logger.info("Initialize Trainer.")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
        )

        logger.info("Training start.")
        if training_args.do_train:
            trainer.train()
            trainer.save_model()
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s %(filename)s:%(lineno)s] %(message)s', level=logging.INFO,
                        filename=None, filemode='a')

    # # 项目的超参，不使用可以删除
    parser = argparse.ArgumentParser(description="Set train parameter.")
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    args = parser.parse_args()

    main = Main(args)
    main.download_data()
    main.train()
