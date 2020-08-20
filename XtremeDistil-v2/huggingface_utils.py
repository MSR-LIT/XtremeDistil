"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from sklearn.decomposition import PCA
from transformers import *

import logging
import models
import numpy as np

logger = logging.getLogger('xtremedistil')

# HuggingFace Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained model config
MODELS = [(TFBertModel, BertTokenizer, BertConfig),
          (TFOpenAIGPTModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
          (TFGPT2Model, GPT2Tokenizer, GPT2Config),
          (TFCTRLModel, CTRLTokenizer, CTRLConfig),
          (TFTransfoXLModel,  TransfoXLTokenizer, TransfoXLConfig),
          (TFXLNetModel, XLNetTokenizer, XLNetConfig),
          (TFXLMModel, XLMTokenizer),
          (TFDistilBertModel, DistilBertTokenizer, DistilBertConfig),
          (TFRobertaModel, RobertaTokenizer, RobertaConfig),
          (TFXLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig),
         ]

def get_special_tokens_from_teacher(Tokenizer, pt_tokenizer):

	if hasattr(pt_tokenizer, 'pad_token'):
		pad_token = pt_tokenizer.pad_token
	else:
		pad_token = "<pad>"
	if Tokenizer == BertTokenizer:
		bos_token = pt_tokenizer.cls_token
		eos_token = pt_tokenizer.sep_token
	else:
		if hasattr(pt_tokenizer, 'bos_token'):
			bos_token = pt_tokenizer.bos_token
		else: 
			bos_token = "<s>"
		if hasattr(pt_tokenizer, 'eos_token'):
			eos_token = pt_tokenizer.eos_token
		else:
			eos_token = "</s>"
	return {"eos_token": eos_token, "bos_token": bos_token, "pad_token": pad_token}


def get_word_embedding(word_emb_file, encoder, pt_tokenizer, word_emb_dim):

	if word_emb_file is None:
		#get word embedding matrix from teacher
		if encoder.base_model_prefix == 'bert':
			word_embedding_matrix = encoder.bert.embeddings.word_embeddings.numpy()
		elif encoder.base_model_prefix == 'transformer':
			word_embedding_matrix = encoder.transformer.embeddings.word_embeddings.numpy()
		elif encoder.base_model_prefix == 'distilbert':
			word_embedding_matrix = encoder.distilbert.embeddings.word_embeddings.numpy()
		elif encoder.base_model_prefix == 'roberta':
			word_embedding_matrix = encoder.roberta.embeddings.word_embeddings.numpy()
		else:
			logger.info("Base model not supported. Initializing word embedding with random matrix")
			word_embedding_matrix = np.random.uniform(size=(len(pt_tokenizer.get_vocab()), word_emb_dim))
		logger.info (word_embedding_matrix.shape)
		#embedding factorization to reduce embedding dimension
		if word_embedding_matrix.shape[1] > word_emb_dim:
			pca =  PCA(n_components = word_emb_dim)
			word_embedding_matrix = pca.fit_transform(word_embedding_matrix)
			word_emb = dict(zip(pt_tokenizer.get_vocab(), word_embedding_matrix))
	else:
		pretrained_word_embedding_file = path+'/pre-trained-data/'+word_emb_file
		logger.info ("Reading word embeddings from {}".format(pretrained_word_embedding_file))
		word_emb = models.read_word_embedding(pretrained_word_embedding_file, pt_tokenizer.vocab)

	return word_emb
