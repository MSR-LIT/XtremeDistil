README
--------------------------

1. pip install -r requirements.txt

2. Sample usages for distilling different pre-trained language models (tested with Python 3.6 and CUDA 10.1)

(a) python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/SST-2 --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 32 --distil_batch_size 1024 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-large-uncased-whole-word-masking --transfer_file unlabeled_sentences.txt

(b) python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/NER --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 256 --distil_batch_size 1024 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-multilingual-cased --do_NER --transfer_file unlabeled_sentences.txt 

(c) python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/NER --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 128 --distil_batch_size 512 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFXLMRobertaModel --pt_teacher_checkpoint jplu/tf-xlm-roberta-large --do_NER --transfer_file unlabeled_sentences.txt 

Arguments

- refer to code for detailed arguments

- task folder contains

	-- train/dev/test '.tsv' files with text and classification labels / token-wise tags (space-separated)
	--- Example 1: feel good about themselves <tab> 1
	--- Example 2: '' Atelocentra '' Meyrick , 1884 <tab> O B-LOC O O O O
	--- (Optional) for computing per-language metrics for multilingual tasks, prefix the first tag of each instance with language-id
	--- Example 3: feel good about themselves <tab> en:1
	--- Example 4: Admiral T feat . <tab> fr:B-PER I-PER O O

	-- label files containing class labels
	-- transfer file containing unlabeled data

- model_dir to store/restore model checkpoints
- seq_len (sequence length)
- do_NER for sequence tagging
- ft_epochs and ft_batch_size for fine-tuning teacher model
- distil_epochs and distil_batch_size for training student model
- teacher_layer for intermediate teacher representation to distill from
- distil_chunk_size for partitioning transfer data (reduce if OOM)
- transfer_file containing unlabeled data

- pt_teacher and pt_teacher_checkpoint for HuggingFace pre-trained transformer models (https://huggingface.co/transformers/)
	-- pre-declared set of MODELS in huggingface_utils (extend for new models)
	-- for checkpoints refer to https://huggingface.co/transformers/pretrained_models.html


