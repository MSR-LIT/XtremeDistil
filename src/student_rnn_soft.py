# Copyright 2020 Authors of ACL Submission ID: 2162

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

from keras import backend as BK
#from keras import optimizers

import keras as K
import tensorflow as tf

import csv
import os

from sklearn.utils import shuffle
#from tensorflow import set_random_seed
import tokenization
import conlleval

from keras.utils import multi_gpu_model, to_categorical

# from keras.utils import plot_model

logger = logging.getLogger(__name__)

# set seeds for random number generator for reproducibility
random.seed(42)
np.random.seed(42)
seed(42)
#set_random_seed(42)
from keras.preprocessing.text import Tokenizer

def get_lang_list():
    return ['en', 'he', 'de', 'ru', 'ta', 'ar', 'pl', 'lt', 'fa', 'el', 'fr', 'it', 'hu', 'es', 'ca', 'mk', 'fi', 'hi', 'et', 'uk', 'hr', 'lv', 'tr', 'nl', 'vi', 'pt', 'sk', 'bn', 'bg', 'da', 'no', 'ro', 'cs', 'sq', 'af', 'id', 'sl', 'bs', 'sv', 'ms', 'tl']


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

def trim(tokenizer, words, labels, MAX_SEQUENCE_LENGTH, teacher=None):

    trimmed_tokens = []
    trimmed_labels = []
    sample_weight = None

    for i, word in enumerate(words):
        token = tokenizer.tokenize(word)
        trimmed_tokens.extend(token)
        if not teacher:
            label = labels[i]
            for j,_ in enumerate(token):
                if j==0:
                    trimmed_labels.append(label)
                else:
                    trimmed_labels.append("X")

    if not teacher:
        try:
            assert len(trimmed_tokens) == len(trimmed_labels)
        except AssertionError:
            print (str(len(trimmed_tokens)) +" "+str(len(trimmed_labels)))
            raise


    if len(trimmed_tokens) > MAX_SEQUENCE_LENGTH - 1:
        trimmed_tokens = trimmed_tokens[:MAX_SEQUENCE_LENGTH - 1]
        sample_weight = np.ones(len(trimmed_tokens)+1)
        if not teacher:
            trimmed_labels = trimmed_labels[:MAX_SEQUENCE_LENGTH - 1]
    else:
        diff = MAX_SEQUENCE_LENGTH  - 1 - len(trimmed_tokens)
        trimmed_tokens = trimmed_tokens + ["[PAD]"]*diff
        sample_weight = np.concatenate((np.ones(MAX_SEQUENCE_LENGTH-diff), np.zeros(diff)))
        if not teacher:
            trimmed_labels = trimmed_labels + ["[PAD]"]*diff
    
    trimmed_tokens.insert(0,"[CLS]")
    if not teacher:
        trimmed_labels.insert(0,"[CLS]")

    try:
        assert len(trimmed_tokens) == MAX_SEQUENCE_LENGTH
        assert len(sample_weight) == MAX_SEQUENCE_LENGTH

        if not teacher:
            assert len(trimmed_labels) == MAX_SEQUENCE_LENGTH
    except AssertionError:
        print (str(len(trimmed_tokens)) +" "+str(len(trimmed_labels)))
        raise

    return(trimmed_tokens, trimmed_labels, sample_weight)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def get_labels(label_file):
    label_map = []
    with open(label_file) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            line = line.strip()
            label_map.append(line)
    label_map.extend(["[PAD]","X","[CLS]"])
    return label_map

def generate_sequence_data(output_dir, MAX_SEQUENCE_LENGTH, input_file, label_map, wordpiece_tokenizer, train=False, distil_tokenizer=None, teacher_examples=None):
    texts = []
    texts_lang = defaultdict(list)
    labels = []
    labels_lang = defaultdict(list)
    sample_wt_lang = defaultdict(list)
    sample_wt = []

    texts_teacher = []
    sample_wt_teacher = []

    if not input_file and not teacher_file:
        print ("no training input.")
        sys.exit(1)

    lang_list = get_lang_list()

    if input_file:
        label_count = defaultdict(int)
        with tf.gfile.Open(input_file, "r") as f:
          reader = csv.reader(f, delimiter="\t", quotechar=None)
          for line in reader:
            if len(line[0].strip()) == 0:
                continue 
            text = tokenization.convert_to_unicode(line[0])
            _labels = tokenization.convert_to_unicode(line[1])

            lang = _labels.split(" ")[0].split(":")[0]
            #assert lang in lang_list

            #remove lang from labels
            _labels = [l.split(":")[1] if ":" in l else l for l in _labels.split(" ")]

            line, _labels, wt = trim(wordpiece_tokenizer, text.split(" "), _labels, MAX_SEQUENCE_LENGTH, teacher=False)
            texts.append(' '.join(line))
            sample_wt.append(wt)            
            labels.append([label_map.index(label) for label in _labels])

            texts_lang[lang].append(texts[-1])
            labels_lang[lang].append(labels[-1])
            sample_wt_lang[lang].append(wt)

            # labels.append(int(tok[1].strip()))
            for label in _labels:
                label_count[label] += 1

    if train and teacher_examples:
        i_count = 0
        for i in range(len(teacher_examples)):
            text = tokenization.convert_to_unicode(teacher_examples[i][0])
            line, _ , wt = trim(wordpiece_tokenizer, text.split(" "), None, MAX_SEQUENCE_LENGTH, teacher=True)
            texts_teacher.append(' '.join(line))
            sample_wt_teacher.append(wt)

    y = np.array(labels)
    y_lang = defaultdict()
    for lang in lang_list:
        y_lang[lang] = np.array(labels_lang[lang])
        sample_wt_lang[lang] = np.array(sample_wt_lang[lang])

    sample_wt = np.array(sample_wt)
    sample_wt_teacher = np.array(sample_wt_teacher)

    if train:        
        distil_tokenizer = Tokenizer(filters='', lower=False, oov_token="[UNK]")
        distil_tokenizer.fit_on_texts(texts+texts_teacher)

        distil_tokenizer.word_index["[PAD]"] = 0

        print ("Size of tokenizer word index ", str(len(distil_tokenizer.word_index)))

    x_sequences = distil_tokenizer.texts_to_sequences(texts)
    x_teacher_sequences = distil_tokenizer.texts_to_sequences(texts_teacher)
    
    x_lang_sequences = defaultdict()

    for lang in lang_list:
        x_lang_sequences[lang] = distil_tokenizer.texts_to_sequences(texts_lang[lang])
        x_lang_sequences[lang] = sequence.pad_sequences(x_lang_sequences[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    X = sequence.pad_sequences(x_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    X_teacher = sequence.pad_sequences(x_teacher_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    print ('X shape:', X.shape)
    print ('X_teacher shape:', X_teacher.shape)
    print ('y shape:', y.shape)

    y = to_categorical(y, len(label_map))
    for lang in lang_list:
        y_lang[lang] = to_categorical(y_lang[lang], len(label_map))
    #print (tokenizer.word_index)

    return X, y, X_teacher, distil_tokenizer, sample_wt, sample_wt_teacher, x_lang_sequences, y_lang, sample_wt_lang


def generator(X, y, X_teacher, y_teacher, y_layer_teacher, X_wt, X_wt_teacher, batch_size):

  start = 0
  end = int(batch_size/2)
  start_t = 0
  end_t = int(batch_size/2)

  while 1:

    if end > len(X):
        X_batch = X[len(X)-int(batch_size/2):len(X)]
        X_wt_batch = X_wt[len(X_wt)-int(batch_size/2):len(X_wt)]
        y_batch = y[len(y)-int(batch_size/2):len(y)]
        # print ('Length exceeded : ', len(X)-int(batch_size/2), len(X), '##', len(y)-int(batch_size/2), len(y))
        start = 0
        end = int(batch_size/2)
        X, y, X_wt = shuffle(X, y, X_wt, random_state=42)
    else:
        X_batch = X[start:end] 
        y_batch = y[start:end]
        X_wt_batch = X_wt[start:end]
        start = end
        end += int(batch_size/2)

    if end_t > len(X_teacher):
        X_teacher_batch = X_teacher[len(X_teacher)-int(batch_size/2):len(X_teacher)]
        y_teacher_batch = y_teacher[len(y_teacher)-int(batch_size/2):len(y_teacher)]
        y_layer_teacher_batch = y_layer_teacher[len(y_layer_teacher)-int(batch_size/2):len(y_layer_teacher)]
        X_wt_teacher_batch = X_wt_teacher[len(X_wt_teacher)-int(batch_size/2):len(X_wt_teacher)]
        start_t = 0
        end_t = int(batch_size/2)
        X_teacher, y_teacher, y_layer_teacher, X_wt_teacher = shuffle(X_teacher, y_teacher, y_layer_teacher, X_wt_teacher, random_state=42)
    else:
        X_teacher_batch = X_teacher[start_t:end_t]
        X_wt_teacher_batch = X_wt_teacher[start_t:end_t]
        y_teacher_batch = y_teacher[start_t:end_t]
        y_layer_teacher_batch = y_layer_teacher[start_t:end_t]
        start_t = end_t
        end_t += int(batch_size/2)

    yield [X_batch, X_teacher_batch], [y_batch, y_teacher_batch], [X_wt_batch, X_wt_teacher_batch]


def evaluate(model, x_test, y_test, labels, MAX_SEQUENCE_LENGTH):

    total = []
    for lang in get_lang_list():# x_test.keys():    
        y_pred = model.predict(x_test[lang])
        pred_tags_all = []
        true_tags_all = []
        for i, seq in enumerate(y_pred):
            for j in range(MAX_SEQUENCE_LENGTH):
                indx = np.argmax(y_test[lang][i][j])
                true_label = labels[indx]
                if "[PAD]" in true_label or "[CLS]" in true_label in true_label:
                    continue
                true_tags_all.append(true_label)
                indx = np.argmax(seq[j])
                pred_label = labels[indx]
                pred_tags_all.append(pred_label)
        prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, verbose=False)
        print ("Lang {} scores {} {} {}".format(lang, prec, rec, f1))
        total.append(f1)
    print ("All f-scores {}".format(total))
    print ("Overall average f-score mean {} and variance {}".format(np.mean(total), np.var(total)))

class MetricsCallback(K.callbacks.Callback):

    def __init__(self, valid_data, labels, MAX_SEQUENCE_LENGTH):
        super().__init__()
        self.valid_data = valid_data        
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        
        #calculate accuracy
        x_dev = self.valid_data[0]
        y_dev = self.valid_data[1]

        x_test = self.valid_data[2]
        y_test = self.valid_data[3]

        # print ("**Validation result**")
        # dev_measure = evaluate(self.model, x_dev, y_dev, self.labels, self.MAX_SEQUENCE_LENGTH, lang=self.lang)
        #getting the f1-score
        #logs['val_categorical_accuracy'] =  dev_measure

        print ("**Test result**")
        # test_measure = self.evaluate_acc(x_test, y_test)
        #getting the f1-score
        test_measure = evaluate(self.model, x_test, y_test, self.labels, self.MAX_SEQUENCE_LENGTH)


def _to_categorical(y, label_size):

    arr = np.zeros((len(y), len(y[0]), label_size))
    for i, seq in enumerate(y):
        for j, k in enumerate(seq):
            arr[i][j][k] = 1
    return np.array(arr)


def soften_logits(y, temperature):
    for i, seq in enumerate(y):
        for j, vec in enumerate(seq):
            y[i][j] = np.exp(vec/temperature)/np.sum(np.exp(vec/temperature))
    return y

def count_parameters(model):
    trainable_count = int(np.sum([BK.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([BK.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable_count, non_trainable_count


def init_model(MAX_SEQUENCE_LENGTH, labels, vocab_indx_dict, batch_size, word_emb, emb_size, dense_hidden_size, bilstm_hidden_size, dropout, s1_loss, s1_opt, s2_opt, x_dev, y_dev, x_test, y_test, stage=None):
        
        mask_zero = False
        epoch_stage_1 = 400
        epoch_stage_2 = 400  
        epoch_stage_3 = 400 
        valid_split = 0.1

        print ("Dense size ", dense_hidden_size)

        gpus = get_available_gpus()
        if gpus == 0:
            gpus = 1
        print ("Number of gpus ", gpus)

        lang_list = get_lang_list()

        earlyStopping = K.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, restore_best_weights=True, verbose=1)

        if stage == 1:
            if 'adam' in s1_opt.lower():
                optimizer = 'Adam'
                lr_scheduler = K.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1)
                callbacks = [earlyStopping, lr_scheduler]
            elif 'adadelta' in s1_opt.lower():
                optimizer = 'Adadelta'
                callbacks = [earlyStopping]
            if 'kld' in s1_loss.lower():
                loss = 'kullback_leibler_divergence'
            elif 'mse' in s1_loss.lower():
                loss = 'mse'
            shared_model = models.construct_bilstm_model_soft_network_stage(vocab_indx_dict, word_emb, emb_size, mask_zero, MAX_SEQUENCE_LENGTH, bilstm_hidden_size, dense_hidden_size, dropout, len(labels), lang_list, stage=1)
            if gpus > 1:
                shared_parallel_model = multi_gpu_model(shared_model, gpus=gpus)
            else:
                shared_parallel_model = shared_model
            shared_parallel_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])

        elif stage == 2:
            # metrics_callback = MetricsCallback(valid_data=(x_dev, y_dev, x_test, y_test), labels=labels, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
            if 'adam' in s2_opt.lower():
                optimizer = 'Adam'
                lr_scheduler = K.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1)
                callbacks = [earlyStopping, lr_scheduler]
            elif 'adadelta' in s2_opt.lower():
                optimizer = 'Adadelta'
                callbacks = [earlyStopping]

            shared_model = models.construct_bilstm_model_soft_network_stage(vocab_indx_dict, word_emb, emb_size, mask_zero, MAX_SEQUENCE_LENGTH, bilstm_hidden_size, dense_hidden_size, dropout, len(labels), lang_list, stage=2)
            if gpus > 1:
                shared_parallel_model = multi_gpu_model(shared_model, gpus=gpus)
            else:
                shared_parallel_model = shared_model
            shared_parallel_model.compile(optimizer=optimizer, loss=['mse'],metrics=['mse'])

        elif stage == 3:
            # metrics_callback = MetricsCallback(valid_data=(x_dev, y_dev, x_test, y_test), labels=labels, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
            if 'adam' in s2_opt.lower():
                optimizer = 'Adam'
                lr_scheduler = K.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1)
                callbacks = [earlyStopping, lr_scheduler]
            elif 'adadelta' in s2_opt.lower():
                optimizer = 'Adadelta'
                callbacks = [earlyStopping]

            shared_model = models.construct_bilstm_model_soft_network_stage(vocab_indx_dict, word_emb, emb_size, mask_zero, MAX_SEQUENCE_LENGTH, bilstm_hidden_size, dense_hidden_size, dropout, len(labels), lang_list, stage=3)
            if gpus > 1:
                shared_parallel_model = multi_gpu_model(shared_model, gpus=gpus)
            else:
                shared_parallel_model = shared_model
            shared_parallel_model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['categorical_accuracy'])

        print(shared_model.summary())
        print ("Parameters ", count_parameters(shared_model))

        return shared_model, shared_parallel_model, callbacks
