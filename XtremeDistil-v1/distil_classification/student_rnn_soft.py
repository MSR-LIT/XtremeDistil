"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for knowledge distillation using recurrent neural networks.
"""

import random
import logging 
import numpy as np
from collections import defaultdict
from numpy.random import seed
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing import sequence
from string import punctuation
from tensorflow.python.client import device_lib

import models
import sys
import re
import math

from keras import backend
from keras import optimizers

import keras as K

import tensorflow as tf

from sklearn.utils import shuffle
from tensorflow import set_random_seed
import tokenization

from keras.utils import multi_gpu_model, to_categorical

# from keras.utils import plot_model

logger = logging.getLogger(__name__)

# set seeds for random number generator for reproducibility
random.seed(42)
np.random.seed(42)
seed(42)
set_random_seed(42)
from keras.preprocessing.text import Tokenizer

#MAX_SEQUENCE_LENGTH = 220

class CosineLRSchedule:
    """
    Cosine annealing with warm restarts, described in paper
    "SGDR: stochastic gradient descent with warm restarts"
    https://arxiv.org/abs/1608.03983

    Changes the learning rate, oscillating it between `lr_high` and `lr_low`.
    It takes `period` epochs for the learning rate to drop to its very minimum,
    after which it quickly returns back to `lr_high` (resets) and everything
    starts over again.

    With every reset:
        * the period grows, multiplied by factor `period_mult`
        * the maximum learning rate drops proportionally to `high_lr_mult`

    This class is supposed to be used with
    `keras.callbacks.LearningRateScheduler`.
    """
    def __init__(self, lr_high: float, lr_low: float, initial_period: int = 50,
                 period_mult: float = 2, high_lr_mult: float = 0.97):
        self._lr_high = lr_high
        self._lr_low = lr_low
        self._initial_period = initial_period
        self._period_mult = period_mult
        self._high_lr_mult = high_lr_mult

    def __call__(self, epoch, lr):
        return self.get_lr_for_epoch(epoch)

    def get_lr_for_epoch(self, epoch):
        assert epoch >= 0
        t_cur = 0
        lr_max = self._lr_high
        period = self._initial_period
        result = lr_max
        for i in range(epoch + 1):
            if i == epoch:  # last iteration
                result = (self._lr_low +
                          0.5 * (lr_max - self._lr_low) *
                          (1 + math.cos(math.pi * t_cur / period)))
            else:
                if t_cur == period:
                    period *= self._period_mult
                    lr_max *= self._high_lr_mult
                    t_cur = 0
                else:
                    t_cur += 1
        return result

def trim(wordpiece, MAX_SEQUENCE_LENGTH):

    trimmed_tokens = []
    
    if len(wordpiece) > MAX_SEQUENCE_LENGTH - 2:
        trimmed_tokens = wordpiece[:MAX_SEQUENCE_LENGTH - 2]
    else:
        diff = MAX_SEQUENCE_LENGTH  - 2 - len(wordpiece)
        trimmed_tokens = wordpiece + ["[PAD]"]*diff
    
    trimmed_tokens.insert(0,"[CLS]")
    trimmed_tokens.extend(["[SEP]"])
    
    try:
        assert len(trimmed_tokens) == MAX_SEQUENCE_LENGTH
    except AssertionError:
        print (str(len(trimmed_tokens)) +" "+str(len(wordpiece)))
        raise

    return(trimmed_tokens)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def generate_sequence_data(MAX_SEQUENCE_LENGTH, input_file, vocab_file, train=False, teacher_examples=None, teacher_lines=None,tokenizer=None):
    texts = []
    labels = []
    texts_teacher = []
    labels_teacher = []
    layer_teacher = []

    if not input_file and not teacher_file:
        print ("no training input.")
        sys.exit(1)

    wordpiece_tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    if input_file:
        label_count = defaultdict(int)
        with open(input_file, encoding="ISO-8859-1") as f:
            for line in f:
                tok = line.strip().split('\t')
                line = ' '.join(trim(wordpiece_tokenizer.tokenize(tok[0].strip()), MAX_SEQUENCE_LENGTH))
                texts.append(line)
                labels.append(int(tok[1].strip()))
                label_count[int(tok[1].strip())] += 1
        for key in label_count.keys():
            print ("Count of instances with label {} is {}".format(key, label_count[key]))

    if train and teacher_examples:
        for i, prediction in enumerate(teacher_examples):
            line = ' '.join(trim(wordpiece_tokenizer.tokenize(teacher_lines[i][0]), MAX_SEQUENCE_LENGTH))
            texts_teacher.append(line)
            layer_teacher.append(prediction['output_layer'])
            labels_teacher.append(np.log(prediction['probabilities']/(1-prediction['probabilities'])))

    y = np.array(labels)
    y_teacher = np.array(labels_teacher)
    y_layer_teacher = np.array(layer_teacher)

    if train:        
        tokenizer = Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts(texts+texts_teacher)
        print ("Size of tokenizer word index ", str(len(tokenizer.word_index)))

    x_sequences = tokenizer.texts_to_sequences(texts)
    x_teacher_sequences = tokenizer.texts_to_sequences(texts_teacher)

    X = sequence.pad_sequences(x_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    X_teacher = sequence.pad_sequences(x_teacher_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    print ('X shape:', X.shape)
    print ('X_teacher shape:', X_teacher.shape)
    print ('y shape:', y.shape)
    print ('y_teacher shape:', y_teacher.shape)
    print ('y_layer_teacher shape:', y_layer_teacher.shape)

    return X, y, X_teacher, y_teacher, y_layer_teacher, tokenizer


def generator(X, y, X_teacher, y_teacher, y_layer_teacher, batch_size):

  start = 0
  end = int(batch_size/2)
  start_t = 0
  end_t = int(batch_size/2)

  while 1:

    if end > len(X):
        X_batch = X[len(X)-int(batch_size/2):len(X)]
        y_batch = y[len(y)-int(batch_size/2):len(y)]
        start = 0
        end = int(batch_size/2)
        X, y = shuffle(X, y, random_state=42)
    else:
        X_batch = X[start:end] 
        y_batch = y[start:end]
        start = end
        end += int(batch_size/2)

    if end_t > len(X_teacher):
        X_teacher_batch = X_teacher[len(X_teacher)-int(batch_size/2):len(X_teacher)]
        y_teacher_batch = y_teacher[len(y_teacher)-int(batch_size/2):len(y_teacher)]
        y_layer_teacher_batch = y_layer_teacher[len(y_layer_teacher)-int(batch_size/2):len(y_layer_teacher)]
        start_t = 0
        end_t = int(batch_size/2)
        X_teacher, y_teacher, y_layer_teacher = shuffle(X_teacher, y_teacher, y_layer_teacher, random_state=42)
    else:
        X_teacher_batch = X_teacher[start_t:end_t]
        y_teacher_batch = y_teacher[start_t:end_t]
        y_layer_teacher_batch = y_layer_teacher[start_t:end_t]
        start_t = end_t
        end_t += int(batch_size/2)

    yield [X_batch, X_teacher_batch], [y_batch, y_teacher_batch]


class MetricsCallback(K.callbacks.Callback):

    def __init__(self, valid_data, stage=2):
        super().__init__()
        self.valid_data = valid_data        
        self.stage = stage

    def evaluate_acc(self, x_test, y_test):

        if self.stage == 2:
            y_pred = self.model.predict([x_test, x_test])
        else:
            y_pred = self.model.predict(x_test)

        cor1, cor2, total = 0, 0, 0
        for i in range(len(y_test)):
            if self.stage == 2:
                if np.argmax(y_pred[0][i]) == np.argmax(y_test[i]):
                    cor1 += 1
                if np.argmax(y_pred[1][i]) == np.argmax(y_test[i]):
                    cor2 += 1
            else:
                if np.argmax(y_pred[i]) == np.argmax(y_test[i]):
                    cor1 += 1
            total += 1

        print ("Acc ", float(cor1)/total, float(cor2)/total)
        return max(float(cor1)/total, float(cor2)/total)


    def on_epoch_end(self, epoch, logs={}):
        
        #calculate accuracy
        x_dev = self.valid_data[0]
        y_dev = self.valid_data[1]

        x_test = self.valid_data[2]
        y_test = self.valid_data[3]

        print ("**Test result**")
        test_measure = self.evaluate_acc(x_test, y_test)

def train_model(MAX_SEQUENCE_LENGTH, vocab_indx_dict, batch_size, x_train, y_train, x_teacher, y_teacher, y_layer_teacher, x_dev, y_dev, x_test, y_test, pretrained_word_embedding_file, model_dir, dense_hidden_size, s1_loss, s_opt):
        
        word_emb = models.get_word_embedding(pretrained_word_embedding_file)
        emb_size = 300
        trainable = True
        mask_zero = False #flatten does not support masking
        bilstm_hidden_size = 600
        dropout = 0.4
        epoch = 200
        valid_split = 0.1

        gpus = get_available_gpus()
        if gpus == 0:
            gpus = 1
        print ("Number of gpus ", gpus)

        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_teacher, y_teacher, y_layer_teacher = shuffle(x_teacher, y_teacher, y_layer_teacher, random_state=42)

        # x_dev = np.array(x_dev)
        # y_dev = np.array(y_dev)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_teacher = np.array(x_teacher)
        y_teacher = np.array(y_teacher)
        y_layer_teacher = np.array(y_layer_teacher)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        labels = set(y_train)
        print ("Class labels ", labels)
        if 0 not in labels:
            #starting indx of labels is set to 0
            for i in range(len(y_train)):
                y_train[i] -= 1
            for i in range(len(y_test)):
                y_test[i] -= 1
            # for i in range(len(y_dev)):
            #     y_dev[i] -= 1

        y_train = to_categorical(y_train, len(labels))
        y_test = to_categorical(y_test, len(labels))
        # y_dev = to_categorical(y_dev, len(labels))

        print("X Train Shape " + str(x_train.shape) + ' ' + str(y_train.shape))
        print("X Teacher Train Shape " + str(x_teacher.shape) + ' ' + str(y_teacher.shape) + ' ' + str(y_layer_teacher.shape))
        print("X Test Shape " + str(x_test.shape) + ' ' + str(y_test.shape))

        #for debugging print top 10 test samples
        rev_word_map = {value: key for key, value in vocab_indx_dict.items()}
        print ("Top 10 test examples.")
        for i in range(10):
            st = ""
            for val in x_test[i]:
                if val in rev_word_map:
                    st += rev_word_map[val] + " "
            print (str(y_test[i]) + "\t" + st)

        print ("Top 10 train examples.")
        for i in range(10):
            st = ""
            for val in x_train[i]:
                if val in rev_word_map:
                    st += rev_word_map[val] + " "
            print (str(y_train[i]) + "\t" + st)

        print ("Top 10 teacher examples.")
        for i in range(10):
            st = ""
            for val in x_teacher[i]:
                if val in rev_word_map:
                    st += rev_word_map[val] + " "
            print (str(y_teacher[i]) + "\t" + st)
        


        earlyStopping = K.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, restore_best_weights=True, verbose=1)

        tunable_layers = ["word_embedding", "bilstm", "dense"]

        if 'adam' in s_opt.lower():
            optimizer = 'Adam'
            lr_scheduler = K.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1)
            callbacks=[earlyStopping, lr_scheduler]
        elif 'adadelta' in s_opt.lower():
            optimizer = 'Adadelta'
            callbacks=[earlyStopping]
        else:
            print ('Optimizer not supported')
            sys.exit(1)

        if 'kld' in s1_loss.lower():
            loss  = 'kld'
            metric = 'kullback_leibler_divergence'
        elif 'cosine' in s1_loss.lower():
            loss = 'cosine'
            metric = 'cosine'
        elif 'mse' in s1_loss.lower():
            loss = 'mse'
            metric = 'mse'
        else:
            print ('Optimizer not supported')
            sys.exit(1)

       
        # stage 1
        model = models.construct_bilstm_model_soft_network_stage(vocab_indx_dict, word_emb, emb_size, trainable, mask_zero, MAX_SEQUENCE_LENGTH, bilstm_hidden_size, dense_hidden_size, dropout, len(labels), stage=1)
        for layer in tunable_layers:
            model.get_layer(layer).trainable = True
        if gpus > 1:
            parallel_model = multi_gpu_model(model, gpus=gpus)
        else:
            parallel_model = model
        parallel_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        parallel_model.fit(x_teacher, y_layer_teacher, batch_size=batch_size*gpus, verbose=2, epochs=epoch, validation_split=0.1, callbacks=callbacks)

        for stage in range(2,4):

            print ("Starting stage {}".format(stage))

            if stage == 2:
                input_ = x_teacher
                output_ = y_teacher
                loss = 'mse'
            elif stage == 3:
                input_ = x_train
                output_ = y_train
                loss = 'categorical_crossentropy'

            cur_model = models.construct_bilstm_model_soft_network_stage(vocab_indx_dict, word_emb, emb_size, trainable, mask_zero, MAX_SEQUENCE_LENGTH, bilstm_hidden_size, dense_hidden_size, dropout, len(labels), stage=stage)

            for layer_name in tunable_layers:
                print ("Transferring weights for layer ", layer_name)
                cur_model.get_layer(layer_name).set_weights(model.get_layer(layer_name).get_weights())

            for layer_name in reversed(tunable_layers):

                #accessing the layers from top to bottom
                print ("Unfreezing layer ", layer_name)
                cur_model.get_layer(layer_name).trainable = True
                    
                if gpus > 1:
                    parallel_model = multi_gpu_model(cur_model, gpus=gpus)
                else:
                    parallel_model = cur_model

                parallel_model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])
                print(parallel_model.summary())
                parallel_model.fit(input_, output_, batch_size=batch_size*gpus, verbose=2, epochs=epoch, callbacks=callbacks, validation_data=(x_test, y_test))

            cur_model.save(model_dir + '/model-stage-2.h5')
            model = cur_model

        # if 0 not in labels:
        #     #starting indx of labels is set to 0
        #     for i in range(len(y_tune)):
        #         y_tune[i] -= 1
        # y_tune = to_categorical(y_tune, len(labels))
