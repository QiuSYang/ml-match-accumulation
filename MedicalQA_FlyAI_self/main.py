# -*- coding: utf-8 -*-

import argparse
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from sklearn.model_selection import train_test_split
from path import DATA_PATH, MODEL_PATH
import pandas as pd
from modelNet import *
from data_helper import *


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()


class Main(FlyAI):
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        # 数据下载解压路径：./data/input/MedicalQA
        data_helper.download_from_ids("MedicalQA")
        print('=='*8+'数据下载完成！'+'=='*8)

    def deal_with_data(self):
        # 创建字典
        # creatDict(os.path.join(DATA_PATH, 'train.csv'))

        # 超参数
        self.sour2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalQA/ask_fr.dict'))
        self.targ2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalQA/ans_fr.dict'))
        # 加载数据集
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalQA/train.csv'))
        # 划分训练集、测试集
        self.train_data, self.test_data = train_test_split(self.data, test_size=0.001, random_state=6, shuffle=True)
        # 计算每个epoch的batch数
        self.steps_per_epoch = int(len(self.train_data.index) / args.BATCH)
        self.source_train, self.target_train = read_data(self.train_data, self.sour2id, self.targ2id)

        print('=='*8+'数据处理完成！'+'=='*8)

    def train(self):
        decoder_vocab_size = len(self.targ2id)
        # RNN Size
        rnn_size = 64
        # Number of Layers
        num_layers = 3
        # Embedding Size
        encoding_embedding_size = 64
        decoding_embedding_size = 64
        # Learning Rate
        learning_rate = 0.001

        # 构造graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)
            input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = \
                get_inputs()

            training_decoder_output, predicting_decoder_output = \
                seq2seq_model(input_data,
                              targets,
                              target_sequence_length,
                              max_target_sequence_length,
                              source_sequence_length,
                              source_vocab_size=len(self.sour2id),
                              rnn_size=rnn_size,
                              num_layers=num_layers,
                              encoding_embedding_size=encoding_embedding_size,
                              decoding_embedding_size=decoding_embedding_size,
                              targ2id=self.targ2id,
                              batch_size=args.BATCH)

            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name="masks")

            # accuracy
            logits_flat = tf.reshape(training_logits, [-1, decoder_vocab_size])
            predict = tf.cast(tf.reshape(tf.argmax(logits_flat, 1), [tf.shape(input_data)[0], -1]),
                              tf.int32, name='predict')
            corr_target_id_cnt = tf.cast(tf.reduce_sum(
                tf.cast(tf.equal(tf.cast(targets, tf.float32), tf.cast(predict, tf.float32)),
                        tf.float32) * masks), tf.int32)
            ans_accuracy = corr_target_id_cnt / tf.reduce_sum(target_sequence_length)

            with tf.name_scope("optimization"):
                cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
                optimizer = tf.train.AdamOptimizer(lr)

                # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，
                # 返回一个以元组(gradient, variable)组成的列表
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                    grad is not None]
                # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，
                # 返回一个应用指定的梯度的操作Operation，对global_step做自增操作
                train_op = optimizer.apply_gradients(capped_gradients)
                # summary_op = tf.summary.merge([tf.summary.scalar("loss", cost)])

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1, args.EPOCHS + 1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(get_batches(
                        self.target_train, self.source_train, args.BATCH, self.sour2id['_pad_'],
                        self.targ2id['_pad_']
                )):
                    _, train_loss = sess.run([train_op, cost], feed_dict={
                        input_data: sources_batch,
                        targets: targets_batch,
                        lr: learning_rate,
                        target_sequence_length: targets_lengths,
                        source_sequence_length: sources_lengths})
                    print('Epoch: {} | CurStep: {}| Train Loss: {}'.
                          format(str(epoch_i) + "/" + str(args.EPOCHS),
                                 str(batch_i+1) + "/" + str(self.steps_per_epoch), train_loss))

                    # 实现自己的保存模型逻辑
                    if batch_i % 10 == 0:
                        save_model(sess, MODEL_PATH, overwrite=True)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)