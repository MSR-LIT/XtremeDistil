#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2019 The BioNLP-HZAU
# Copyright 2020 Authors of ACL Submission ID: 2162
# Time:2019/10/07
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import _pickle as cPickle
import csv
from absl import flags,logging
import modeling
import optimization
import tokenization
import tensorflow as tf
import metrics
import numpy as np
import conlleval
from collections import defaultdict
from sklearn.utils import shuffle
from timeit import default_timer as timer

import sys
from student_rnn_soft import generate_sequence_data, init_model, evaluate, count_parameters
from tensorflow.python.client import device_lib
from models import get_word_embedding
from keras.utils import multi_gpu_model

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("distil_task", None, "Name of the task to distil.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the output will be written.")

flags.DEFINE_string(
    "model_dir", None,
    "The model directory where the BERT checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "pred_file", None,
    "The file for generating predictions.")

flags.DEFINE_string(
    "bert_prediction_file", "",
    "The BERT file with generated predictions.")

flags.DEFINE_string(
    "word_embedding_file", None,
    "Pre-trained word embedding file")

flags.DEFINE_string(
    "shared_model_file", "",
    "Shared model file with distil weights.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_distil", False,
    "Whether to distil BERT.")

flags.DEFINE_string("task", None, help="name of the task")

# flags.DEFINE_string("teacher", None, help="which teacher to choose")

flags.DEFINE_string("s1_loss", 'mse', help="which loss to use for stage 1")

flags.DEFINE_string("s1_opt", 'adam', help="which optimizer to use for stage 1")

flags.DEFINE_string("s2_opt", 'adam', help="which optimizer to use for stage 2")

flags.DEFINE_string("path", None, help="path of data")


flags.DEFINE_integer("bilstm_hidden_size", 600, "hidden state size for bilstm")

flags.DEFINE_integer("distil_batch_size", 256, "batch size for distillation")

flags.DEFINE_float("dropout_rate", 0.4, "dropout rate")

flags.DEFINE_integer("teacher_layer", None, "teacher layer to distil from")

flags.DEFINE_integer("distil_teacher_batch_size", 300000, "batch size for distillation")

flags.DEFINE_float("alpha", 10.0, "weighing distillation losses")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("crf", "True", "use crf!")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        print ('HERE ', input_file)
        with tf.gfile.Open(input_file, "r") as f:
          reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
          lines = []
          for line in reader:
            if len(line[0].strip()) == 0:
                continue 
            #strip lang tags
            # print ("PREV ", line)
            if len(line) > 1 and len(line[1]) > 0:
                line[1] = ' '.join([val.split(":")[1] if ':' in val else val for val in line[1].split(" ")])
            # print ("CUR ", line)
            lines.append(line)
          return lines


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_pred_examples(self, pred_file):
        """See base class."""
        print ('Prediction File  ', pred_file)

        lines = self._read_tsv(pred_file)
        #print (lines)
        examples = self._create_example(lines, "test")
        return lines, examples


    def get_labels(self, data_dir):
        """See base class."""
        labels = [x for y in self._read_tsv(os.path.join(data_dir, "labels.tsv")) for x in y]
        labels.extend(["[PAD]","X","[CLS]"])
        print ("HERE ", labels)
        return labels

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          guid = "%s-%s" % (set_type, i)
          if set_type == "test":
            text_a = tokenization.convert_to_unicode(line[0])
            if len(line) == 2 and len(line[1].strip()) > 0:
              label = tokenization.convert_to_unicode(line[1])
            else:
              label = ' '.join(["X"]*len(text_a.split(' ')))
          else:
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text=text_a, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')

    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")

    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(label_map["[PAD]"])
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 10:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens %s" % ' '.join(ntokens))
        # logging.info("tokens: %s" % " ".join(
        #     [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(label_list[x]) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    lang_tokens = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)

        label = example.label.split(" ")[0]
        if ":" in label:
            lang = label.split(":")[0]
        else:
            lang = "default"
        lang_tokens.extend([lang]*len(ntokens))

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels, lang_tokens

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    #TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels,num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params =trans ,sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
   
    return loss,transition

def softmax_layer(logits,labels,num_labels,mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict, probabilities


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
        )

    output_layer = model.get_sequence_output()
    teacher_layer = model.get_teacher_layer(FLAGS.teacher_layer)
    #output_layer_pooled = model.get_pooled_output()
    #output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer,num_labels)
    # TODO test shape
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    if 'true' in FLAGS.crf.lower():
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits, predict, viterbi_score, teacher_layer)
    else:
        loss,predict, probabilities  = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict, probabilities, teacher_layer)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if 'true' in FLAGS.crf.lower():
            print ("************ CRF")
            (total_loss, logits,predicts, viterbi_score, output_layer) = create_model(bert_config, is_training, input_ids, mask, segment_ids, label_ids,num_labels, use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts, probabilities, output_layer) = create_model(bert_config, is_training, input_ids, mask, segment_ids, label_ids,num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names=None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                 init_string)

        def binarize_fn(logits):
            predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            return predictions

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                predictions={"logits": logits},
                #eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions={"logits": logits, "output_layer": output_layer}, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
            

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def main(_):
    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    # if not FLAGS.do_train and not FLAGS.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels(FLAGS.data_dir)

    wordpiece_tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        _,_,_ = filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, wordpiece_tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        logging.info("Running eval on test files")
        eval_examples = processor.get_test_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        batch_tokens,batch_labels,lang_tokens = filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, wordpiece_tokenizer, eval_file)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        pred_tags_dict = defaultdict(list)
        true_tags_dict = defaultdict(list)
        pred_tags_all = []

        result = estimator.predict(input_fn=predict_input_fn)
        for i, prediction in enumerate(result):
            logits = prediction["logits"]
            for j in range(FLAGS.max_seq_length):
                label = label_list[np.argmax(logits[j])]
                pred_tags_all.append(label)

        true_tags = []
        pred_tags = []

        assert len(batch_labels) == len(pred_tags_all)

        for i, label in enumerate(batch_labels):
            lang = lang_tokens[i]           

            if "[PAD]" in label_list[label] or "[CLS]" in label_list[label]:
                continue

            true_tags_dict[lang].append(label_list[label])
            pred_tags_dict[lang].append(pred_tags_all[i])
            
            true_tags.append(label_list[label])
            pred_tags.append(pred_tags_all[i])

        print (len(pred_tags))
        print (len(true_tags))
        print (pred_tags[:10])
        print (true_tags[:10])

        scores = []
        for lang in pred_tags_dict.keys():
            prec, rec, f1 = conlleval.evaluate(true_tags_dict[lang], pred_tags_dict[lang], verbose=True)
            print ("***Lang***", lang, prec, rec, f1)
            scores.append(f1)

        prec, rec, f1 = conlleval.evaluate(true_tags, pred_tags, verbose=True)
        print ("***Overall***", prec, rec, f1)
        print ("***Average***", np.mean(scores))

    if FLAGS.do_predict:
        label2id = {}
        id2label = {}
        for (i,label) in enumerate(label_list):
            label2id[label] = i
            id2label[i] = label

        gpus = get_available_gpus()
        if gpus == 0:
            gpus = 1
        print ("Number of gpus ", gpus)

        predict_lines, predict_examples = processor.get_pred_examples(FLAGS.pred_file)
        
        print ("Number of teacher examples: {}".format(len(predict_lines)))
        print ("Teacher layer: {}".format(FLAGS.teacher_layer))
        print ("Task " , FLAGS.distil_task)
        print ("Max Seq Length ", FLAGS.max_seq_length)
        print ("Batch Size ", FLAGS.distil_batch_size)
        print ("Dropout rate ", FLAGS.dropout_rate)
        print ("Alpha ", FLAGS.alpha)
        print ("Path ", FLAGS.path)
        print ("Output / Model Directory ", FLAGS.output_dir)
        print ("Directory of script ", os.path.dirname(os.path.abspath(__file__)))

        x_train, y_train, x_teacher, distil_tokenizer, x_wt, x_wt_teacher, x_lang, y_lang, x_wt_lang = generate_sequence_data(FLAGS.output_dir, FLAGS.max_seq_length, FLAGS.path+"/datasets/"+FLAGS.distil_task+"/train.tsv", label_list, wordpiece_tokenizer, train=True, teacher_examples=predict_lines)

        dense_hidden_size = 768 #y_layer_teacher.shape[2]
        print ("Dimension of teacher's representation layer : ", str(dense_hidden_size))

        # x_dev, y_dev = None, None
        x_dev, y_dev, _, _, _, _, x_dev_lang, y_dev_lang, _, = generate_sequence_data(FLAGS.output_dir, FLAGS.max_seq_length, FLAGS.path+"/datasets/"+FLAGS.distil_task+"/dev.tsv", label_list, wordpiece_tokenizer, distil_tokenizer=distil_tokenizer)

        x_test, y_test, _, _, _, _, x_test_lang, y_test_lang, _, = generate_sequence_data(FLAGS.output_dir, FLAGS.max_seq_length, FLAGS.path+"/datasets/"+FLAGS.distil_task+"/test.tsv", label_list, wordpiece_tokenizer, distil_tokenizer=distil_tokenizer)

        print("X Train Shape " + str(x_train.shape) + ' ' + str(x_wt.shape) + ' ' + str(y_train.shape))
        print("X Dev Shape " + str(x_dev.shape) + ' ' + str(y_dev.shape))
        print("X Test Shape " + str(x_test.shape) + ' ' + str(y_test.shape))

        pretrained_word_embedding_file = FLAGS.path+'/pre-trained-data/'+FLAGS.word_embedding_file.strip()
        # pretrained_word_embedding_file = FLAGS.path+'/pre-trained-data/mbase_sample_1.txt'

        print (pretrained_word_embedding_file)

        word_emb, emb_size = get_word_embedding(pretrained_word_embedding_file, wordpiece_tokenizer.vocab)

        model_1, model_stage_1, callbacks_stage_1 = init_model(FLAGS.max_seq_length , label_list, distil_tokenizer.word_index, FLAGS.distil_batch_size, word_emb, emb_size, dense_hidden_size, FLAGS.bilstm_hidden_size, FLAGS.dropout_rate, FLAGS.s1_loss, FLAGS.s1_opt, FLAGS.s2_opt, x_dev_lang, y_dev_lang, x_test_lang, y_test_lang, stage = 1)

        init_dense_weights_1 = model_1.get_layer('dense').get_weights()

        model_2, model_stage_2, callbacks_stage_2 = init_model(FLAGS.max_seq_length , label_list, distil_tokenizer.word_index, FLAGS.distil_batch_size, word_emb, emb_size, dense_hidden_size, FLAGS.bilstm_hidden_size, FLAGS.dropout_rate, FLAGS.s1_loss, FLAGS.s1_opt, FLAGS.s2_opt, x_dev_lang, y_dev_lang, x_test_lang, y_test_lang, stage = 2)

        model_3, model_stage_3, callbacks_stage_3 = init_model(FLAGS.max_seq_length , label_list, distil_tokenizer.word_index, FLAGS.distil_batch_size, word_emb, emb_size, dense_hidden_size, FLAGS.bilstm_hidden_size, FLAGS.dropout_rate, FLAGS.s1_loss, FLAGS.s1_opt, FLAGS.s2_opt, x_dev_lang, y_dev_lang, x_test_lang, y_test_lang, stage = 3)

        shared_layers = ['dense', 'bilstm', 'word_embedding']

        def compile_parallel_model(model, stage=None):
            if stage == 1:
                if 'adam' in FLAGS.s1_opt.lower():
                    optimizer = 'Adam'
                elif 'adadelta' in FLAGS.s1_opt.lower():
                    optimizer = 'Adadelta'
                if 'kld' in FLAGS.s1_loss.lower():
                    loss = 'kullback_leibler_divergence'
                elif 'mse' in FLAGS.s1_loss.lower():
                    loss = 'mse'
                if gpus > 1:
                    parallel_model = multi_gpu_model(model, gpus=gpus)
                else:
                    parallel_model = model
                parallel_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])
            elif stage == 2:
                #show all trainable and non-trainable parameters
                if 'adam' in FLAGS.s2_opt.lower():
                    optimizer = 'Adam'
                elif 'adadelta' in FLAGS.s2_opt.lower():
                    optimizer = 'Adadelta'
                if gpus > 1:
                    parallel_model = multi_gpu_model(model, gpus=gpus)
                else:
                    parallel_model = model
                
                parallel_model.compile(optimizer=optimizer, loss=['mse'], metrics=['mse'])
            elif stage == 3:
                #show all trainable and non-trainable parameters
                if 'adam' in FLAGS.s2_opt.lower():
                    optimizer = 'Adam'
                elif 'adadelta' in FLAGS.s2_opt.lower():
                    optimizer = 'Adadelta'
                if gpus > 1:
                    parallel_model = multi_gpu_model(model, gpus=gpus)
                else:
                    parallel_model = model
                
                parallel_model.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['categorical_accuracy'])
            else:
                print ("Wrong stage")
                sys.exit(1)

            print(model.summary())
            print ("Parameters ", count_parameters(model))
            return parallel_model

        for stage in range(1,10):

            print ("*** Starting stage {}".format(stage))
        
            if stage == 2:
                #copy weights from model_stage_1
                print ("Copying weights from model stage 1")
                assert not np.array_equal(init_dense_weights_1, model_1.get_layer('dense').get_weights()) 

                for layer in shared_layers:
                    model_2.get_layer(layer).set_weights(model_1.get_layer(layer).get_weights())
                    model_2.get_layer(layer).trainable = False
                model_stage_2 = compile_parallel_model(model_2, stage=2)
            elif stage == 6:
                #copy weights from model_stage_2
                print ("Copying weights from model stage 2")
                for layer in shared_layers:
                    model_3.get_layer(layer).set_weights(model_2.get_layer(layer).get_weights())
                    model_3.get_layer(layer).trainable = False
                model_stage_3 = compile_parallel_model(model_3, stage=3)
            elif stage > 2 and stage < 6:
                print ("Unfreezing layer {}".format(shared_layers[stage-3]))
                model_2.get_layer(shared_layers[stage-3]).trainable = True
                model_stage_2 = compile_parallel_model(model_2, stage=2)
            elif stage > 6:
                print ("Unfreezing layer {}".format(shared_layers[stage-7]))
                model_3.get_layer(shared_layers[stage-7]).trainable = True
                model_stage_3 = compile_parallel_model(model_3, stage=3)

            start_teacher = 0

            while start_teacher < len(predict_lines) and stage < 6:

                end_teacher = min(start_teacher + FLAGS.distil_teacher_batch_size, len(predict_lines))

                print ("Teacher indices from {} to {}".format(start_teacher, end_teacher))

                predict_file = os.path.join(FLAGS.output_dir, "predict_{}.tf_record".format(start_teacher))
                if not os.path.exists(predict_file):
                    _,_,_ = filed_based_convert_examples_to_features(predict_examples[start_teacher:end_teacher], label_list, FLAGS.max_seq_length, wordpiece_tokenizer, predict_file)

                logging.info("***** Running BERT prediction*****")
                logging.info("  Num examples = %d", len(predict_examples[start_teacher:end_teacher]))
                logging.info("  Batch size = %d", FLAGS.predict_batch_size)

                predict_input_fn = file_based_input_fn_builder(
                    input_file=predict_file,
                    seq_length=FLAGS.max_seq_length,
                    is_training=False,
                    drop_remainder=False)

                result = estimator.predict(input_fn=predict_input_fn)

                y_teacher = []
                y_layer_teacher = []
                for i, prediction in enumerate(result):
                    y_teacher.append(prediction['logits'])
                    y_layer_teacher.append(prediction['output_layer'])
                y_teacher = np.array(y_teacher)
                y_layer_teacher = np.array(y_layer_teacher)

                print("X Teacher Train Shape " + str(x_teacher[start_teacher:end_teacher].shape) + ' ' + str(y_teacher[start_teacher:end_teacher].shape) + ' ' + str(y_layer_teacher[start_teacher:end_teacher].shape))

                if stage == 1:
                    model_file = os.path.join(FLAGS.output_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
                    if os.path.exists(model_file):
                        print ("Loadings weights for stage 1 from {}".format(model_file))
                        model_1.load_weights(model_file)
                        model_stage_1 = compile_parallel_model(model_1, stage=1)
                    else:
                        model_stage_1.fit(x_teacher[start_teacher:end_teacher], y_layer_teacher, batch_size=FLAGS.distil_batch_size*gpus, verbose=2, epochs=80, callbacks=callbacks_stage_1, validation_split=0.1)
                        model_1.save_weights(model_file)
                elif stage > 1 and stage < 6:
                    model_file = os.path.join(FLAGS.output_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
                    if os.path.exists(model_file):
                        print ("Loadings weights for stage 2 from {}".format(model_file))
                        model_2.load_weights(model_file)
                        model_stage_2 = compile_parallel_model(model_2, stage=2)
                    else:
                        model_stage_2.fit(x_teacher[start_teacher:end_teacher], y_teacher, batch_size=FLAGS.distil_batch_size*gpus, verbose=2, epochs=80, callbacks=callbacks_stage_2, validation_split=0.1)
                        model_2.save_weights(model_file)

                start_teacher = end_teacher

            if stage > 1 and stage < 6:
                evaluate(model_2, x_test_lang, y_test_lang, label_list, FLAGS.max_seq_length)

            if stage > 5:

                model_file = os.path.join(FLAGS.output_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
                if os.path.exists(model_file):
                    print ("Loadings weights for stage 3 from {}".format(model_file))
                    model_3.load_weights(model_file)
                    model_stage_3 = compile_parallel_model(model_3, stage=3)
                else:
                    x_train, y_train, x_wt = shuffle(x_train, y_train, x_wt)
                    model_stage_3.fit(x_train, y_train, batch_size=FLAGS.distil_batch_size*gpus, verbose=2, epochs=80, callbacks=callbacks_stage_3, validation_split=0.1)
                    model_3.save_weights(model_file)

                evaluate(model_3, x_test_lang, y_test_lang, label_list, FLAGS.max_seq_length)

        evaluate(model_3, x_test_lang, y_test_lang, label_list, FLAGS.max_seq_length)

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()