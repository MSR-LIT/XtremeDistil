"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from collections import defaultdict
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import csv
import logging
import numpy as np
import six
import tensorflow as tf

logger = logging.getLogger('xtremedistil')

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def sequence_tag(tokenizer, words, labels, special_tokens, MAX_SEQUENCE_LENGTH, teacher=None):

    trimmed_tokens = []
    trimmed_labels = []

    prev = []
    for i in range(1, len(words)+1):
        cur = tokenizer.tokenize(' '.join(words[:i]))
        diff = cur[len(prev):]
        trimmed_tokens.extend(diff)
        if not teacher:
            for j,_ in enumerate(diff):
                if j==0:
                    trimmed_labels.append(labels[i-1])
                else:
                    trimmed_labels.append("X")
        prev = cur

    if not teacher:
        try:
            assert len(trimmed_tokens) == len(trimmed_labels)
        except AssertionError:
            logger.error ("Dimension mismatch {} {}".format(len(trimmed_tokens), len(trimmed_labels)))
            raise

    if len(trimmed_tokens) > MAX_SEQUENCE_LENGTH - 2:
        trimmed_tokens = trimmed_tokens[:MAX_SEQUENCE_LENGTH - 2]
        if not teacher:
            trimmed_labels = trimmed_labels[:MAX_SEQUENCE_LENGTH - 2]
        trimmed_tokens.insert(0, special_tokens["bos_token"])
        trimmed_tokens.extend([special_tokens["eos_token"]])
        if not teacher:
            trimmed_labels.insert(0, special_tokens["bos_token"])
            trimmed_labels.extend([special_tokens["eos_token"]])
    else:
        diff = MAX_SEQUENCE_LENGTH  - 2 - len(trimmed_tokens)
        trimmed_tokens = [special_tokens["bos_token"]] + trimmed_tokens + [special_tokens["eos_token"]] + [special_tokens["pad_token"]]*diff
        if not teacher:
            trimmed_labels = [special_tokens["bos_token"]] + trimmed_labels + [special_tokens["eos_token"]] + [special_tokens["pad_token"]]*diff

    try:
        assert len(trimmed_tokens) == MAX_SEQUENCE_LENGTH
        if not teacher:
            assert len(trimmed_labels) == MAX_SEQUENCE_LENGTH
    except AssertionError:
        logger.error ("Dimension mismatch {} {}".format(len(trimmed_tokens), len(trimmed_labels)))
        raise

    return(trimmed_tokens, trimmed_labels)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def get_labels(label_file, special_tokens=None):
    label_map = []
    with open(label_file) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            line = line.strip()
            label_map.append(line)
    if special_tokens is not None:
        label_map.extend([special_tokens["pad_token"], "X", special_tokens["eos_token"], special_tokens["bos_token"]])
    return label_map

def generate_sequence_data(MAX_SEQUENCE_LENGTH, input_file, label_map, pt_tokenizer, special_tokens, train=False, distil_tokenizer=None, teacher_file=None, distil=False, do_NER=False):

    texts = []
    texts_lang = defaultdict(list)
    labels = []
    labels_lang = defaultdict(list)
    texts_teacher = []
    lang_list = set()
    label_count = defaultdict(int)

    with tf.io.gfile.GFile(input_file, "r") as f:

      reader = csv.reader(f, delimiter="\t", quotechar=None)
      for line in reader:
        text = convert_to_unicode(line[0])
        label = convert_to_unicode(line[1])
        first_tag = label.split(" ")[0]
        if ":" in first_tag:
            lang = first_tag.split(":")[0]
        else:
            lang = "default"
        lang_list.add(lang)

        #remove lang from labels
        label = [l.split(":")[1] if ":" in l else l for l in label.split(" ")]

        if do_NER:
            text, label = sequence_tag(pt_tokenizer, text.split(" "), label, special_tokens, MAX_SEQUENCE_LENGTH, teacher=False)
        else:
            text = pt_tokenizer.tokenize(text)

        texts.append(' '.join(text))
        labels.append([label_map.index(l) for l in label])

        texts_lang[lang].append(texts[-1])
        labels_lang[lang].append(labels[-1])

        for l in label:
            label_count[l] += 1

    if teacher_file:
        i_count = 0
        with tf.io.gfile.GFile(teacher_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line in reader:
                if len(line)==0:
                    continue
                text = convert_to_unicode(line[0])
                text, _ = sequence_tag(pt_tokenizer, text.split(" "), None, special_tokens, MAX_SEQUENCE_LENGTH, teacher=True)
                texts_teacher.append(' '.join(text))

    y = np.array(labels)
    y_lang = defaultdict()
    x_lang_sequences = defaultdict()

    for lang in lang_list:
        y_lang[lang] = np.array(labels_lang[lang])

    if train and distil:        
        logger.info ("*************** Init Distil Tokenizer ***************")
        distil_tokenizer = Tokenizer(filters='', lower=False, oov_token="[UNK]")
        distil_tokenizer.fit_on_texts(texts+texts_teacher)
        distil_tokenizer.word_index[special_tokens["pad_token"]] = 0
        logger.info ("Size of tokenizer word index {}".format(len(distil_tokenizer.word_index)))

    if distil:
        x_sequences = distil_tokenizer.texts_to_sequences(texts)
        x_teacher_sequences = distil_tokenizer.texts_to_sequences(texts_teacher)

        for lang in lang_list:
            x_lang_sequences[lang] = distil_tokenizer.texts_to_sequences(texts_lang[lang])
            x_lang_sequences[lang] = sequence.pad_sequences(x_lang_sequences[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    else:
        x_sequences = np.array([pt_tokenizer.convert_tokens_to_ids(text.split(' ')) for text in texts])
        x_teacher_sequences = np.array([pt_tokenizer.convert_tokens_to_ids(text.split(' ')) for text in texts_teacher])

        for lang in lang_list:
            x_lang_sequences[lang] = np.array([pt_tokenizer.convert_tokens_to_ids(text.split(' ')) for text in texts_lang[lang]])
            x_lang_sequences[lang] = sequence.pad_sequences(x_lang_sequences[lang], padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    X = sequence.pad_sequences(x_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
    X_teacher = sequence.pad_sequences(x_teacher_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    logger.info ("X shape {}".format(X.shape))
    logger.info ("y shape {}".format(y.shape))
    logger.info ("X_teacher shape {}".format(X_teacher.shape))

    if distil:
        if train:
            return X, y, X_teacher, distil_tokenizer
        else:
            return X, y, x_lang_sequences, y_lang
    else:
        return X, y, X_teacher, x_lang_sequences, y_lang
