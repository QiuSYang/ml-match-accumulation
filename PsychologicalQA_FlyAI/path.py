# -*- coding: utf-8 -*
import sys
import os
import logging
import platform
from dataclasses import dataclass, field
from typing import Optional
import torch
from flyai.utils import remote_helper

logger = logging.getLogger(__name__)
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
OUTPUT_PATH = os.path.join(sys.path[0], 'data', 'output')


@dataclass
class DataArguments:
    """数据预处理超参数"""
    dataset_path: str = field(
        metadata={"help": "原始数据路径."}
    )
    max_sequence_len: int = field(
        default=512, metadata={"help": "序列最大长度"}
    )
    max_condition_len: int = field(
        default=100, metadata={"help": "条件序列最大长度"}
    )
    max_target_len: int = field(
        default=50, metadata={"help": "目标序列最大长度"}
    )
    is_right_pad: bool = field(
        default=True, metadata={"help": "Padding补齐方式"}
    )
    is_condition_first: bool = field(
        default=False, metadata={"help": "条件文本是否放在最前面"}
    )
    is_unilm_mask: bool = field(
        default=False, metadata={"help": "是否构造unilm attention mask"}
    )


@dataclass
class ModelArguments:
    """模型超参数"""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


class HyperParametersConfig(object):
    def __init__(self, epochs=10, batch_size=4):
        # decoder data
        self.max_sequence_len = 512  # context_len + condition_len = target_len
        self.max_target_len = 50
        self.dataset_path = os.path.join(DATA_PATH, "PsychologicalQA/train.csv")

        # train
        self.seed = 2021
        self.model_name_or_path = self.get_pre_train_model_path()  # "thu-coai/CDial-GPT2_LCCC-base"
        self.output_dir = os.path.join(OUTPUT_PATH, "models")
        self.logging_dir = os.path.join(OUTPUT_PATH, "logs")

        gpu_nums = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.per_device_train_batch_size = batch_size
        self.per_device_eval_batch_size = batch_size
        self.gradient_accumulation_steps = 8
        # self.max_steps = 15000
        self.num_train_epochs = epochs
        self.warmup_steps = 500
        self.save_steps = 1000
        # self.eval_steps = 1000
        self.logging_steps = 150

        self.local_rank = -1

        self.do_train = True
        # self.do_eval = False
        # self.evaluation_strategy = "epoch"  # "steps"
        # self.evaluate_during_training = True
        self.prediction_loss_only = True

    def get_pre_train_model_path(self):
        """FLYAI框架下获取预训练模型的PATH"""
        path = remote_helper.get_remote_data('https://www.flyai.com/m/CDial-GPT2_LCCC-base.zip')
        if platform.system() == "Linux":
            path_list = path.split('/')
            path = "/".join(path_list[:-1])
        elif platform.system() == "Windows":
            path_list = path.split('\\')
            path = "\\".join(path_list[:-1])

        logger.info("Model path: {}".format(path))
        return path
