"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from evaluation import ner_evaluate, classify_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher, get_word_embedding
from scheduler import CosineLRSchedule, create_learning_rate_scheduler
from preprocessing import generate_sequence_data, get_labels
from transformers import *
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model

import argparse
import logging
import models
import numpy as np
import os
import random
import sys
import tensorflow as tf



#set random seeds
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

#logging
logger = logging.getLogger('xtremedistil')
logging.basicConfig(level = logging.INFO)



if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()

	#required arguments
	parser.add_argument("--task", required=True, help="name of the task")
	parser.add_argument("--model_dir", required=True, help="path of model directory")
	parser.add_argument("--seq_len", required=True, type=int, help="sequence length")
	parser.add_argument("--transfer_file", required=True, help="transfer data for distillation")

	#task
	parser.add_argument("--do_NER", action="store_true", default=False, help="whether to perform NER")

	#student model parameters (optional)
	parser.add_argument("--word_emb_dim", nargs="?", type=int, default=300, help="word embedding dimension")
	parser.add_argument("--word_emb_file", nargs="?", default=None, help="word embedding file for student model")
	parser.add_argument("--bilstm_hidden_size", nargs="?", type=int, default=600, help="hidden state size for bilstm")
	parser.add_argument("--dropout_rate", nargs="?", type=float, default=0.2, help="dropout rate for student model")
	parser.add_argument("--dense_act_func", nargs="?", default=models.gelu, help="Activation function for the student representation layer")

	#teacher model parameters (optional)
	parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model to distil")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-multilingual-cased", help="teacher model checkpoint to load to pre-trained weights")
	parser.add_argument("--teacher_layer", nargs="?", type=int, default=6, help="intermediate teacher layer to distil from")

	#batch sizes and epochs (optional)
	parser.add_argument("--distil_batch_size", nargs="?", type=int, default=512, help="batch size for distillation")
	parser.add_argument("--ft_batch_size", nargs="?", type=int, default=32, help="batch size for distillation")
	parser.add_argument("--ft_epochs", nargs="?", type=int, default=32, help="epochs for fine-tuning")
	parser.add_argument("--distil_epochs", nargs="?", type=int, default=80, help="epochs for distillation")
	parser.add_argument("--distil_chunk_size", nargs="?", type=int, default=300000, help="transfer data partition size (reduce if OOM)")


	args = vars(parser.parse_args())
	logger.info(args)

	task_name = args["task"]
	max_seq_length = args["seq_len"]
	distil_batch_size = args["distil_batch_size"]
	model_dir = args["model_dir"]
	word_emb_file = args["word_emb_file"]
	word_emb_dim = args["word_emb_dim"]
	bilstm_hidden_size = args["bilstm_hidden_size"]
	dropout_rate = args["dropout_rate"]
	ft_epochs = args["ft_epochs"]
	distil_epochs = args["distil_epochs"]
	ft_batch_size = args["ft_batch_size"]
	distil_batch_size = args["distil_batch_size"]
	teacher_layer = args["teacher_layer"]
	distil_chunk_size = args["distil_chunk_size"]
	pt_teacher = args["pt_teacher"]
	pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
	do_NER = args["do_NER"]
	transfer_file = args["transfer_file"]
	dense_act_func = args["dense_act_func"]

	logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))

	#Get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == pt_teacher:
			TFModel, Tokenizer, Config = MODELS[indx]

	#get pre-trained tokenizer and special tokens
	pt_tokenizer = Tokenizer.from_pretrained(pt_teacher_checkpoint)
	special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)

	#get labels
	if do_NER:
		label_list = get_labels(os.path.join(task_name, "labels.tsv"), special_tokens)
	else:
		label_list = get_labels(os.path.join(task_name, "labels.tsv"))

	#generate sequence data for fine-tuning pre-trained teacher
	x_train_teacher, y_train_teacher, x_transfer_teacher, _, _ = generate_sequence_data(max_seq_length, os.path.join(task_name, "train.tsv"), label_list, pt_tokenizer, special_tokens, train=True, teacher_file=os.path.join(task_name, transfer_file), do_NER=do_NER)
	x_dev_teacher, y_dev_teacher, _, _, _ = generate_sequence_data(max_seq_length, os.path.join(task_name, "dev.tsv"), label_list, pt_tokenizer, special_tokens, do_NER=do_NER)
	x_test_teacher, y_test_teacher, _, x_test_lang_teacher, y_test_lang_teacher = generate_sequence_data(max_seq_length, os.path.join(task_name, "test.tsv"), label_list, pt_tokenizer, special_tokens, do_NER=do_NER)

	#logging teacher data shapes
	logger.info("X Teacher Train Shape {} {}".format(x_train_teacher.shape, y_train_teacher.shape))
	logger.info("X Teacher Dev Shape {} {}".format(x_dev_teacher.shape, y_dev_teacher.shape))
	logger.info("X Teacher Test Shape {} {}".format(x_test_teacher.shape, y_test_teacher.shape))
	logger.info("X Teacher Transfer Shape {}".format(x_transfer_teacher.shape))
	for i in range(3):
		logger.info ("Example {}".format(i))
		logger.info (x_train_teacher[i])
		logger.info (pt_tokenizer.convert_ids_to_tokens(x_train_teacher[i]))
		logger.info (' '.join([label_list[v] for v in y_train_teacher[i]]))

	#generate sequence data for distilling student
	x_train_student, y_train_student, x_transfer_student, distil_tokenizer = generate_sequence_data(max_seq_length, os.path.join(task_name, "train.tsv"), label_list, pt_tokenizer, special_tokens, train=True, teacher_file=os.path.join(task_name, transfer_file), distil=True, do_NER=do_NER)
	x_dev_student, y_dev_student, _, _ = generate_sequence_data(max_seq_length, os.path.join(task_name, "dev.tsv"), label_list, pt_tokenizer, special_tokens, distil=True, distil_tokenizer=distil_tokenizer, do_NER=do_NER)
	x_test_student, y_test_student, x_test_lang_student, y_test_lang_student = generate_sequence_data(max_seq_length, os.path.join(task_name, "test.tsv"), label_list, pt_tokenizer, special_tokens, distil=True, distil_tokenizer=distil_tokenizer, do_NER=do_NER)

	#logging student data
	logger.info("X Student Train Shape {} {}".format(x_train_student.shape, y_train_student.shape))
	logger.info("X Student Dev Shape {} {}".format(x_dev_student.shape, y_dev_student.shape))
	logger.info("X Student Test Shape {} {}".format(x_test_student.shape, y_test_student.shape))
	logger.info("X Student Transfer Shape {}".format(x_transfer_student.shape))

	# inv_vocab = {v: k for k, v in distil_tokenizer.word_index.items()}
	# logger.info ("*************** Train Lang Example ***************")
	# for i in range(3):
	# 	logger.info ("Train Lang Example {}".format(i))
	# 	logger.info (x_train_student[i])
	# 	logger.info (' '.join([inv_vocab[w] for w in x_train_student[i]]))
	# 	logger.info (' '.join([label_list[v] for v in y_train_student[i]]))

	#fine-tune pre-trained teacher
	strategy = tf.distribute.MirroredStrategy()
	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))
	with strategy.scope():
		config = Config.from_pretrained(pt_teacher_checkpoint, output_hidden_states=True)
		encoder = TFModel.from_pretrained(pt_teacher_checkpoint, config=config)
		input_ids = Input(shape=(max_seq_length,), dtype=tf.int32)
		if do_NER:
			embedding = encoder(input_ids)[0]
			intermediate_embeddding = encoder(input_ids)[2][teacher_layer]
		else:
			embedding = encoder(input_ids)[0][:,0]
			intermediate_embeddding = encoder(input_ids)[2][teacher_layer][:,0]
		outputs = Dense(len(label_list), activation='linear', name="final_logits")(embedding)

		teacher_model = tf.keras.Model(inputs=input_ids, outputs=outputs)
		teacher_intermediate_layer = tf.keras.Model(inputs=teacher_model.input, outputs=intermediate_embeddding)
		logger.info (teacher_model.summary())
		teacher_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
	model_file = os.path.join(model_dir, "model-ft.h5")
	if os.path.exists(model_file):
		logger.info ("Loadings weights for fine-tuned model from {}".format(model_file))
		teacher_model.load_weights(model_file)
	else:
		teacher_model.fit(x=x_train_teacher, y=y_train_teacher, batch_size=ft_batch_size*gpus, shuffle=True, epochs=ft_epochs, callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5, end_learn_rate=1e-7, warmup_epoch_count=10, total_epoch_count=ft_epochs), tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)], validation_data=(x_dev_teacher, y_dev_teacher))
		teacher_model.save_weights(model_file)

	#evaluate fine-tuned teacher
	if do_NER:
		ner_evaluate(teacher_model, x_test_lang_teacher, y_test_lang_teacher, label_list, special_tokens, max_seq_length, batch_size=ft_batch_size*gpus)
	else:
		classify_evaluate(teacher_model, x_test_lang_teacher, y_test_lang_teacher, batch_size=ft_batch_size*gpus)

	#get word embedding matrix with dimensionality reduction
	word_emb = get_word_embedding(word_emb_file, encoder, pt_tokenizer, word_emb_dim)

	#construct student models for different stages
	with strategy.scope():
		model_1 = models.construct_bilstm_student_model(distil_tokenizer.word_index, word_emb, max_seq_length, bilstm_hidden_size, config.hidden_size, dropout_rate, dense_act_func, len(label_list), stage=1, do_NER=do_NER)
		model_1 = models.compile_model(model_1, strategy, stage=1)

		model_2 = models.construct_bilstm_student_model(distil_tokenizer.word_index, word_emb, max_seq_length, bilstm_hidden_size, config.hidden_size, dropout_rate, dense_act_func, len(label_list), stage=2, do_NER=do_NER)
		model_2 = models.compile_model(model_2, strategy, stage=2)

	#get shared layers for student models
	shared_layers = []
	for layer in model_1.layers:
		if len(layer.trainable_weights) > 0:
			shared_layers.append(layer.name)
	#update parameters top down from the shared layers
	shared_layers.reverse()
	logger.info ("Shared layers {}".format(shared_layers))

	best_model = None
	best_eval = 0

	#start stage-wise distillation
	#number of loops is given by 3-stage distillation for each of shared layers
	for stage in range(1, 2*len(shared_layers)+4):

		logger.info ("*** Starting stage {}".format(stage))
		#stage = 1, optimize reprsentation loss (transfer set) with end-to-end training
		#stage = 2, copy model from stage = 1, and optimize logit loss (transfer set) with all but last layer frozen
		#stage = [3, 4, .., 2+num_shared_layers], optimize logit loss (transfer set) with gradual unfreezing
		#stage == 3+num_shared_layers, optimize CE loss (labeled data) with all but last layer frozen
		#stage = [4+num_shared_layers, ...], optimize CE loss (labeled data)with gradual unfreezing

		if stage == 2:
			#copy weights from model_stage_1
			logger.info ("Copying weights from model stage 1")
			for layer in shared_layers:
				model_2.get_layer(layer).set_weights(model_1.get_layer(layer).get_weights())
				model_2.get_layer(layer).trainable = False
			model_2 = models.compile_model(model_2, strategy, stage=2)
		elif stage > 2 and stage < 3+len(shared_layers):
			logger.info ("Unfreezing layer {}".format(shared_layers[stage-3]))
			model_2.get_layer(shared_layers[stage-3]).trainable = True
			model_2 = models.compile_model(model_2, strategy, stage=2)
		elif stage == 3+len(shared_layers):
			for layer in shared_layers:
				model_2.get_layer(layer).trainable = False
			model_2 = models.compile_model(model_2, strategy, stage=3)
		elif stage > 3+len(shared_layers):
			logger.info ("Unfreezing layer {}".format(shared_layers[stage-4-len(shared_layers)]))
			model_2.get_layer(shared_layers[stage-4-len(shared_layers)]).trainable = True
			model_2 = models.compile_model(model_2, strategy, stage=3)

		start_teacher = 0

		assert len(x_transfer_teacher) == len(x_transfer_student)

		while start_teacher < len(x_transfer_teacher) and stage < 3+len(shared_layers):

			end_teacher = min(start_teacher + distil_chunk_size, len(x_transfer_teacher))
			logger.info ("Teacher indices from {} to {}".format(start_teacher, end_teacher))

			#get teacher logits
			y_teacher = teacher_model.predict(np.array(x_transfer_teacher[start_teacher:end_teacher]), batch_size=distil_batch_size*gpus)
			#get representation from intermediate teacher layer
			y_layer_teacher = teacher_intermediate_layer.predict(np.array(x_transfer_teacher[start_teacher:end_teacher]), batch_size=distil_batch_size*gpus)
			logger.info("X Teacher Transfer Shape ".format(np.array(x_transfer_teacher[start_teacher:end_teacher]).shape, np.array(y_layer_teacher).shape))

			if stage == 1:
				model_file = os.path.join(model_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
				if os.path.exists(model_file):
					logger.info ("Loadings weights for stage 1 from {}".format(model_file))
					model_1.load_weights(model_file)
				else:
					logger.info (model_1.summary())
					model_1.fit(x_transfer_student[start_teacher:end_teacher], y_layer_teacher, shuffle=True, batch_size=distil_batch_size*gpus, verbose=2, epochs=distil_epochs, callbacks=[tf.keras.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], validation_split=0.1)
					model_1.save_weights(model_file)
			elif stage > 1 and stage < 3+len(shared_layers):
				model_file = os.path.join(model_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
				if os.path.exists(model_file):
					logger.info ("Loadings weights for stage 2 from {}".format(model_file))
					model_2.load_weights(model_file)

				else:
					logger.info (model_2.summary())
					model_2.fit(x_transfer_student[start_teacher:end_teacher], y_teacher, shuffle=True, batch_size=distil_batch_size*gpus, verbose=2, epochs=distil_epochs, callbacks=[tf.keras.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], validation_split=0.1)
					model_2.save_weights(model_file)

			start_teacher = end_teacher

		if stage > 1 and stage < 3+len(shared_layers):
			if do_NER:
				cur_eval = ner_evaluate(model_2, x_test_lang_student, y_test_lang_student, label_list, special_tokens, max_seq_length, batch_size=distil_batch_size*gpus)
			else:
				cur_eval = classify_evaluate(model_2, x_test_lang_student, y_test_lang_student, batch_size=distil_batch_size*gpus)
			if cur_eval > best_eval:
				best_eval = cur_eval
				best_model = "model-stage-{}-indx-{}.h5".format(stage, start_teacher)

		if stage >= 3+len(shared_layers):
			model_file = os.path.join(model_dir, "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
			if os.path.exists(model_file):
				logger.info ("Loadings weights for stage 3 from {}".format(model_file))
				model_2.load_weights(model_file)
			else:
				logger.info (model_2.summary())
				model_2.fit(x_train_student, y_train_student, batch_size=ft_batch_size*gpus, verbose=2, shuffle=True, epochs=ft_epochs, callbacks=[tf.keras.callbacks.LearningRateScheduler(CosineLRSchedule(lr_high=0.001, lr_low=1e-8),verbose=1), tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)], validation_data=(x_dev_student, y_dev_student))
				model_2.save_weights(model_file)
			if do_NER:
				cur_eval = ner_evaluate(model_2, x_test_lang_student, y_test_lang_student, label_list, special_tokens, max_seq_length, batch_size=distil_batch_size*gpus)
			else:
				cur_eval = classify_evaluate(model_2, x_test_lang_student, y_test_lang_student, batch_size=distil_batch_size*gpus)
			if cur_eval > best_eval:
				best_eval = cur_eval
				best_model = "model-stage-{}-indx-{}.h5".format(stage, start_teacher)

	logger.info ("Best eval score {}".format(best_eval))
	logger.info ("Best model checkpoint {}".format(best_model))


