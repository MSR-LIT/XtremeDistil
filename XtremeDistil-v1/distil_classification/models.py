"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for neural network models for different stages of distillation.
"""

import random
import logging as logger
import numpy as np
from keras import backend as K
from keras.initializers import RandomUniform
from keras.layers import Embedding, Input, LSTM, Bidirectional, TimeDistributed, Dropout, Dense, Conv1D, Lambda, \
    RepeatVector, Activation, Flatten, Permute, Add, concatenate, merge, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.regularizers import l2
from numpy.random import seed
from tensorflow import set_random_seed
from keras.regularizers import l1, l2
import tensorflow as tf
import attention
from keras.utils import multi_gpu_model

# set seeds for random number generator for reproducibility
random.seed(42)
np.random.seed(42)
seed(42)
set_random_seed(42)

# set seeds for random number generator for reproducibility
random.seed(42)
np.random.seed(42)
seed(42)
set_random_seed(42)

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

def get_word_embedding(pretrained_word_embedding_file):
    word_embedding = {}
    f = open(pretrained_word_embedding_file, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            continue
        word_embedding[word] = coefs
    f.close()

    print ("Word embeddings loaded for " + str(len(word_embedding)) + " words.")
    return word_embedding

def construct_word_embedding_layer(word_index, word_emb, emb_size, trainable, mask_zero, input_length):
    num_words = len(word_index)
    embedding_matrix = np.zeros((num_words + 1, emb_size))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = word_emb.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, emb_size)

    embedding_layer = Embedding(num_words + 1, emb_size, weights=[embedding_matrix], input_length=input_length,
                                trainable=trainable, mask_zero=mask_zero, name='word_embedding')
    return embedding_layer


def construct_bilstm_model_soft_network_stage(word_index, word_emb, emb_size, trainable, mask_zero, timesteps, bilstm_hidden_size, dense_hidden_size, dropout, classes, stage=None):
    emb_layer = construct_word_embedding_layer(word_index, word_emb, emb_size, False, mask_zero, timesteps)
    rnn_layer = Bidirectional(LSTM(bilstm_hidden_size, dropout=0.2, recurrent_dropout=0.2, return_sequences=True), name="bilstm", trainable=False)

    dense_layer = Dense(dense_hidden_size, activation=gelu, name="dense", trainable=False)

    mi = Input(shape=(timesteps,), dtype='int32', name="input")    
    x1 = emb_layer(mi)
    x1 = Dropout(dropout)(x1)    
    x1 = rnn_layer(x1)
    x1 = GlobalMaxPooling1D()(x1)
    x1 = Dropout(dropout)(x1)
    x1 = dense_layer(x1)
    if stage == 1:
        return  Model(inputs=mi, outputs=x1)
    elif stage == 2:
        x1 = Dropout(dropout)(x1)
        mo = Dense(classes, activation="linear", name='output')(x1)
        return Model(inputs=mi, outputs=mo)
    else:
        x1 = Dropout(dropout)(x1)
        mo = Dense(classes, activation="softmax", name='output')(x1)
        return Model(inputs=mi, outputs=mo)

    return None