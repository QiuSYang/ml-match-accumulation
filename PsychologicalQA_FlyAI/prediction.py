# -*- coding: utf-8 -*
import torch
import logging
from flyai.framework import FlyAI
from transformers import BertTokenizer
from custom_model import CustomGPTGeneration
from path import HyperParametersConfig

logger = logging.getLogger(__name__)


class Prediction(FlyAI):
    def __init__(self):
        self.config = HyperParametersConfig()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.output_dir)
        self.model = self.load_model()

    def load_model(self):
        """
        模型初始化，必须在此方法中加载模型
        """
        model = CustomGPTGeneration.from_pretrained(self.config.output_dir)
        model.to(self.config.device)

        return model

    def predict(self, question):
        """
        模型预测返回结果
        :参数示例 question='老是胡思乱想该怎么办'
        :return: 返回预测结果格式具体说明如下： 最多返回三个生成式答案组成的list
        评估中该样本的F1值是返回三个生成式预测答案分别和真实答案计算F1值的最大值
        """

        self.model.eval()
        with torch.no_grad():
            inputs = self.text_to_ids(question, self.tokenizer)
            for key, value in inputs.items():
                inputs[key] = value.unsqueeze(dim=0).to(self.config.device)

            current_len = inputs["input_ids"].size(1)
            bos_token_id = self.tokenizer._convert_token_to_id("[CLS]")
            pad_token_id = self.tokenizer._convert_token_to_id("[PAD]")
            eos_token_id = self.tokenizer._convert_token_to_id("[SEP]")
            results = self.model.generate(
                max_length=current_len + self.config.max_target_len,
                num_beams=3,
                early_stopping=True,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                num_return_sequences=3,
                **inputs
            )

            predicts_ids = results.data.cpu().numpy().tolist()
            results = []
            for i, predict_ids in enumerate(predicts_ids):
                result_text = self.ids_to_text(predict_ids[current_len:], self.tokenizer)
                results.append(result_text)

        return results

    @staticmethod
    def text_to_ids(question, tokenizer, max_question_len=416):
        question_tokens = tokenizer.tokenize(question)
        if len(question_tokens) > max_question_len:
            question_tokens = question_tokens[-max_question_len:]  # 从后向前截断

        tokens = ["[CLS]"] + question_tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(input_ids)
        mask_ids = [1.0] * len(input_ids)
        assert len(input_ids) == len(token_type_ids) == len(mask_ids)

        return {
                "input_ids": torch.tensor(input_ids).long(),
                "attention_mask": torch.tensor(mask_ids).float(),
                "token_type_ids": torch.tensor(token_type_ids).long()
        }

    @staticmethod
    def ids_to_text(ids, tokenizer):
        valid_ids = []
        for id_ in ids:
            if int(id_) == tokenizer._convert_token_to_id("[SEP]"):
                break
            else:
                valid_ids.append(int(id_))
        text = "".join(tokenizer.convert_ids_to_tokens(valid_ids))
        text = text.replace("[UNK]", "").replace("#", "")

        # Deduplication
        qlist = []
        pre_char = ""
        for char in text:
            if char != pre_char:
                qlist.append(char)
            pre_char = char
        text = "".join(qlist)

        return text
