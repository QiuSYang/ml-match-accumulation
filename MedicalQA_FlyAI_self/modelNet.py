# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.layers.core import Dense


# 输入层
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# Encoder
"""
在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给LSTM进行处理。
在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。
"""


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_output, encoder_state = \
        tf.compat.v1.nn.dynamic_rnn(cell, encoder_embed_input,
                                    sequence_length=source_sequence_length, dtype=tf.float32)

    return encoder_output, encoder_state


def process_decoder_input(data, phonem_dict, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], phonem_dict['_sos_']), ending], 1)

    return decoder_input


def decoding_layer(phonem_dict, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    '''
    构造Decoder层
    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''

    # 1. Embedding
    target_vocab_size = len(phonem_dict)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # 4. Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([phonem_dict['_sos_']], dtype=tf.int32),
                               [tf.shape(target_sequence_length)[0]], name='start_token')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens,
                                                                     phonem_dict['_eos_'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = \
            tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                              impute_finished=True, maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


# 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
def seq2seq_model(input_data, targets, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, rnn_size, num_layers,
                  encoding_embedding_size, decoding_embedding_size, targ2id, batch_size):
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    decoder_input = process_decoder_input(targets, targ2id, batch_size)

    training_decoder_output, predicting_decoder_output = decoding_layer(targ2id,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


def save_model(session, path, overwrite=False):
    '''
    保存模型
    :param session: 训练模型的sessopm
    :param path: 要保存模型的路径
    :param name: 要保存模型的名字
    :param overwrite: 是否覆盖当前模型
    :return:
    '''

    def delete_file(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    if overwrite:
        delete_file(path)
    print(path)

    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path, 'best'))
    builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
    builder.save()


def get_tensor_name(name):
    return name + ":0"