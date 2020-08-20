"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from preprocessing import convert_to_unicode
from tensorflow.keras.layers import Embedding, Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.models import Model

import csv
import logging
import numpy as np
import os
import random
import tensorflow as tf

logger = logging.getLogger('xtremedistil')

# set seeds for random number generator for reproducibility
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def read_word_embedding(pretrained_word_embedding_file, vocab):

    word_embedding = {}
    inv_vocab = {v: k for k, v in vocab.items()}
    indx = 0

    if 'muse' in pretrained_word_embedding_file.lower() or 'glove' in pretrained_word_embedding_file.lower():
        with tf.io.gfile.GFile(pretrained_word_embedding_file, "r") as f:
            reader = csv.reader(f, delimiter=" ", quotechar=None)
            for line in reader:
                if len(line[0].strip()) == 0 or len(line) == 2:
                    continue 
                word = convert_to_unicode(line[0])
                coefs = np.asarray(line[1:], dtype='float32')
                word_embedding[word] = coefs
    else:  
        with open(pretrained_word_embedding_file, "r") as f:
            for line in f:
                word = inv_vocab[indx]
                coefs = np.asarray(line.strip().split(" "), dtype='float32')
                word_embedding[word] = coefs
                indx += 1

    logger.info ("Word embeddings loaded for {} words".format(len(word_embedding)))

    return word_embedding

def construct_word_embedding_layer(word_index, word_emb, input_length):
    num_words = len(word_index)
    word_emb_dim = len(next(iter(word_emb.values())))
    embedding_matrix = np.zeros((num_words + 2, word_emb_dim))
    for word, i in word_index.items():
        embedding_vector = word_emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, word_emb_dim)

    embedding_layer = Embedding(num_words + 2, word_emb_dim, weights=[embedding_matrix], input_length=input_length, mask_zero=False, name='word_embedding')

    return embedding_layer

def construct_bilstm_student_model(word_index, word_emb, timesteps, bilstm_hidden_size, dense_hidden_size, dropout, dense_act_func, classes, stage, do_NER):

    word_emb_dim = len(next(iter(word_emb.values())))
    emb_layer = construct_word_embedding_layer(word_index, word_emb, timesteps)
    if do_NER:
        rnn_layer = Bidirectional(LSTM(bilstm_hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), name="bilstm")
    else:
        rnn_layer = Bidirectional(LSTM(bilstm_hidden_size, dropout=0.2, recurrent_dropout=0.2), name="bilstm")
    dense_layer = Dense(dense_hidden_size, activation=dense_act_func, name="dense")

    mi = Input(shape=(timesteps,), dtype='int32', name="input")    
    x1 = emb_layer(mi)
    x1 = Dropout(dropout)(x1)    
    x1 = rnn_layer(x1)
    x1 = Dropout(dropout)(x1)    
    x1 = dense_layer(x1)

    if stage == 1:
        return Model(inputs=mi, outputs=x1)
    else:
        x1 = Dropout(dropout)(x1)
        mo = Dense(classes, activation="linear", name="output")(x1)
        return Model(inputs=mi, outputs=mo)


def construct_transformer_student_model(word_index, word_emb, mask_zero, timesteps, bilstm_hidden_size, dense_hidden_size, dropout, dense_act_func, classes, stage):

    emb_size = len(next(iter(word_emb.values())))
    num_heads = 8
    ff_dim = 300

    emb_layer = TokenAndPositionEmbedding(timesteps, len(word_index), emb_size)
    dense_layer = Dense(dense_hidden_size, activation=dense_act_func, name="dense")
    transformer_block = TransformerBlock(emb_size, num_heads, ff_dim)

    mi = Input(shape=(timesteps,), dtype='int32', name="input")    
    x1 = emb_layer(mi)
    x1 = Dropout(dropout)(x1)    
    x1 = transformer_block(x1)
    x1 = Dropout(dropout)(x1)
    # x1 = transformer_block(x1)
    # x1 = Dropout(dropout)(x1)
    # x1 = transformer_block(x1)
    # x1 = Dropout(dropout)(x1)
    x1 = dense_layer(x1)

    if stage == 1:
        return Model(inputs=mi, outputs=x1)
    else:
        x1 = Dropout(dropout)(x1)
        mo = Dense(classes, activation="linear")(x1)
        return Model(inputs=mi, outputs=mo)


def compile_model(model, strategy, stage):

    #construct student models for different stages
    with strategy.scope():
        if stage == 1 or stage == 2:
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['mse'], metrics=['mse'])
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    return model