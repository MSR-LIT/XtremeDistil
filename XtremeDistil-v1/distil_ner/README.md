# Microsoft Research
# Code for https://aka.ms/XtremeDistil

***Update 7/4/2020*** 
Releasing v1 of XtremeDistil based on the original BERT implementation from https://github.com/google-research/bert and retaining much of the original configurations, parameter settings and nomenclatures.

***Upcoming release***
We are porting XtremeDistil to Tensorflow 2.1 with HuggingFace Transformers for easy extensibility and distilling all supported pre-trained language models. Please check back by end of July for XtremeDistil-v2.


************************************************
Instructions to run XtremeDistil
************************************************

Step 1: Continue pre-training BERT language model on task-specific **unlabeled data** starting from pre-trained checkpoint at https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip. Refer to https://github.com/google-research/bert for details.

```
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5 \
  --do_whole_word_mask=True
```

```
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

Step 2: Fine-tune BERT on task-specific **labeled data**. Generate fine-tuned model checkpoints with `run_classifier.py` with standard arguments (see below); set `do_train=True`, `do_eval=True` and `do_distil=False`.

Step 3: Run the distillation-code with the following arugments (change only the corresponding directories). 

```
python3.6 BERT_NER.py --task_name=NER --do_lower_case=False --crf=False --do_train=False --do_eval=False --do_distil=True --data_dir=../datasets/NER --vocab_file=../models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../models/multi_cased_L-12_H-768_A-12/bert_config.json --max_seq_length=32 --train_batch_size=32 --eval_batch_size=32 --predict_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=/tmp --init_checkpoint=../models/ner_ft_lm/model.ckpt-100000 --pred_file=../datasets/NER/transfer_set.tsv --model_dir=../models/multi_ner_unify --s1_loss=kld --s1_opt=adam --s2_opt=adam  --word_embedding_file=../datasets/mbase.txt --teacher_layer=-7

```
```
Arguments:
-- refer to "BERT_NER.py" for description of the above parameters
-- init_checkpoint contains the pre-trained language model checkpoint from Step 1
-- model_dir contains the model checkpoint for fine-tuned classifier from Step 2
-- pred_file is the unlabeled transfer set for used for distillation
-- word_embedding_file (could be Glove or SVD on MBERT word embedding)
-- teacher_layer to distil from
```
