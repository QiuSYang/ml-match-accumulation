# -*- coding: utf-8 -*
import os
import argparse
from path import DATA_PATH, MODEL_PATH, VOCAB_PATH, MODEL_PATH_BEST
from data_helper import load_dict, text2id
from flyai.framework import FlyAI
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2LMHeadModel


def set_predict_args():
    """设置预测使用超参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, help='生成的temperature')
    parser.add_argument('--repetition_penalty', default=1.0, type=float,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--topk', default=8, type=int, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, help='最高积累概率')
    parser.add_argument('--max_len', type=int, default=100, help='每个utterance的最大长度,超过指定长度则进行截断')

    return parser.parse_args()


class Prediction(FlyAI):
    def load_model(self):
        """
        模型初始化，必须在构造方法中加载模型
        """
        self.args = set_predict_args()
        self.args.cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.args.cuda else 'cpu'

        self.tokenizer = BertTokenizer(vocab_file=VOCAB_PATH)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH_BEST)
        self.n_ctx = self.model.config.to_dict().get("n_ctx")
        self.model.to(self.device)

    def predict(self, department, title, ask):
        """
        模型预测返回结果
        :param input: 评估传入样例 {"department": "心血管科", "title": "心率为72bpm是正常的吗",
                                    "ask": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"answer": "心脏不舒服一般是..."}
        """
        # 每个input以[CLS]为开头
        input_ids = [self.tokenizer.cls_token_id]
        if department:
            input_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in department])
            # 不同属性之间使用[SEP]隔开
            input_ids.append(self.tokenizer.sep_token_id)
        if title:
            input_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in title])
            # 不同属性之间使用[SEP]隔开
            input_ids.append(self.tokenizer.sep_token_id)
        if ask:
            input_ids.extend([self.tokenizer.convert_tokens_to_ids(word) for word in ask])
            # 不同属性之间使用[SEP]隔开
            input_ids.append(self.tokenizer.sep_token_id)

        # 输入的最大长度不能超过
        if len(input_ids) > self.n_ctx:
            # 预留最后一位保存[SEP]
            input_ids = input_ids[:self.n_ctx-1]
            input_ids.append(self.tokenizer.sep_token_id)

        current_input_tensor = torch.tensor(input_ids).long().to(self.device)

        generated = []
        # 最多生成max_len个token
        for i in range(self.args.max_len):
            outputs = self.model(input_ids=current_input_tensor)
            next_token_logits = outputs[0][-1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for index in set(generated):
                next_token_logits[index] /= self.args.repetition_penalty
            next_token_logits = next_token_logits / self.args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = self.top_k_top_p_filtering(next_token_logits,
                                                         top_k=self.args.topk,
                                                         top_p=self.args.topp)

            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素
            # 权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == self.tokenizer.sep_token_id:
                # 遇到[SEP]则表明response生成结束
                break
            generated.append(next_token.item())
            # 将预测新token加入current_input_tensor
            current_input_tensor = torch.cat((current_input_tensor, next_token), dim=0)

        text_list = self.tokenizer.convert_ids_to_tokens(generated)

        text = ''.join(text_list)

        return {'answer': text}

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
            # ...表示其他维度由计算机自行推断
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits


if __name__ == "__main__":
    if not os.path.exists(VOCAB_PATH):
        VOCAB_PATH = './vocabulary/vocab_small.txt'
    if not os.path.exists(MODEL_PATH_BEST):
        MODEL_PATH_BEST = './GPT2_wiki'

    predict = Prediction()
    predict.load_model()

    result = predict.predict(department='',
                             title='孕妇经常胃痛会影响到胎儿吗',
                             ask="我怀上五个多月了,自从怀上以来就经常胃痛(两个胸之间往下一点儿是胃吧?)有时痛十几分钟,"
                                 "有时痛半个钟,每次都痛得好厉害,不过痛过这一阵之后就不痛了,"
                                 "我怀上初期饮食不规律,经常吃不下东西,会不会是这样引发的呀?"
                                 "我好忧心对胎儿有影响,该怎么办呢?可有食疗的方法可以纾解一下痛呢?")
    print(result)
