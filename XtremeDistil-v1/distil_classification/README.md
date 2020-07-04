# Microsoft Research
# Code for [XtremeDistil](https://www.microsoft.com/en-us/research/publication/xtremedistil/)

***Update 7/4/2020*** 
Releasing v1 of XtremeDistil based on the original BERT implementation from https://github.com/google-research/bert and retaining much of the original configurations, parameter settings and nomenclatures.

***Upcoming release***
We are porting XtremeDistil to Tensorflow 2.1 with HuggingFace Transformers for easy extensibility and distilling all supported pre-trained language models. Please check back by end of July for XtremeDistil-v2.


************************************************
Instructions to run XtremeDistil
************************************************

Step 1: Continue pre-training BERT language model on task-specific **unlabeled data** starting from pre-trained checkpoints. Refer to https://github.com/google-research/bert for details.

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
python3.6 run_classifier.py --task_name=SST --do_train=false --do_eval=false --do_distil=true --data_dir=../datasets/SST-2 --vocab_file=../pre-trained-data/wwm_uncased_L-24_H-1024_A-16/vocab.txt --bert_config_file=../pre-trained-data/wwm_uncased_L-24_H-1024_A-16/bert_config.json --init_checkpoint=../bert_large_output/aclImdb/model.ckpt-100000 --max_seq_length=40 --train_batch_size=128 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=../bert_large_output/SST-2 --s1_loss=kld --s_opt=adam --predict_batch_size=128 --pred_file=../datasets/aclImdb/acl_unlabeled_transfer.txt --word_emb_file=../pre-trained-data/glove.840B.300d.txt
```

```
Arguments:

Refer to `run_classifier.py` for description of the arguments. Some of the important ones are highlighted here:

-- s1_loss (cosine / mse / kld) for stage 1 optimization
-- s_opt (adam / adadelta) for optimization algo

-- init_checkpoint for BERT **language model** checkpoint 
----Option 1. Use the pre-trained checkpoints available from https://github.com/google-research/bert
----Option 2. Fine-tune BERT language model on domain-specific unlabeled data from Step 1.

-- output_dir for fine-tuned BERT **classifier** checkpoint on downstream task from Step 2.

-- pred_file is the unlabeled transfer set for distillation

-- data_dir is the path of the dataset containing the following files.
---- train.tsv with first column as the text and second column as the label
---- dev.tsv with the corresponding test file
```
