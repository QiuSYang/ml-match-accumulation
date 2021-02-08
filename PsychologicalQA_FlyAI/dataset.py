"""
# 数据处理脚本
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class PsychologicalQADataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_sequence_len=512):
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        self.samples = self.get_samples(data_path)

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)

    def get_samples(self, data_path):
        """获取数据集并转换"""
        csv_data = pd.read_csv(data_path, header=0)
        # print(csv_data.head())

        samples = []
        for row in csv_data.itertuples():
            question = row.question.replace("\n", " ").replace("\t", " ").replace("\\", "")
            answer = row.answer.replace("\n", " ").replace("\t", " ").replace("\\", "")
            question_tokens = self.tokenizer.tokenize(question)
            answer_tokens = self.tokenizer.tokenize(answer)

            max_question_len = self.max_sequence_len - len(answer_tokens) - 3  # 三个特殊token
            if len(question_tokens) > max_question_len:
                question_tokens = question_tokens[-max_question_len:]  # 从后向前截断

            tokens = ["[CLS]"] + question_tokens + ["[SEP]"] + answer_tokens + ["[SEP]"]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = ids[:-1]
            target_ids = ids[1:]
            token_type_ids = [0] * (len(question_tokens) + 2) + [1] * len(answer_tokens)
            mask_ids = [1.0] * len(input_ids)
            assert len(input_ids) == len(token_type_ids) == len(mask_ids)

            extra = self.max_sequence_len - len(input_ids)
            if extra > 0:
                input_ids += [self.pad_token_id] * extra
                target_ids += [-100] * extra
                mask_ids += [0.0] * extra
                token_type_ids += [1] * extra

            samples.append({
                "input_ids": torch.tensor(input_ids).long(),
                "attention_mask": torch.tensor(mask_ids).float(),
                "token_type_ids": torch.tensor(token_type_ids).long(),
                "labels": torch.tensor(target_ids).long()
            })

        return samples


if __name__ == "__main__":
    model_name_or_path = "thu-coai/CDial-GPT2_LCCC-base"
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, "data/input/PsychologicalQA/train.csv")

    dataset = PsychologicalQADataset(data_path=data_path, tokenizer=tokenizer)
