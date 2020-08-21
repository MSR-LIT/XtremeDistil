[**XtremeDistil-v2**](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v2) with Tensorflow 2.1 and [HuggingFace Transformers](https://huggingface.co/transformers) for distilling all supported [pre-trained language models](https://huggingface.co/transformers/pretrained_models.html) with an unified API for multilingual text classification and sequence tagging tasks.

```pip install -r requirements.txt```

Sample usages for distilling different pre-trained language models (tested with Python 3.6 and CUDA 10.1)

```PYTHONHASHSEED=42 python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/SST-2 --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 32 --distil_batch_size 1024 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-large-uncased-whole-word-masking --transfer_file unlabeled_sentences.txt```

```PYTHONHASHSEED=42 python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/NER --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 256 --distil_batch_size 1024 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFBertModel --pt_teacher_checkpoint bert-base-multilingual-cased --do_NER --transfer_file unlabeled_sentences.txt```

```PYTHONHASHSEED=42 python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/NER --model_dir $$PT_OUTPUT_DIR --seq_len 32 --ft_epochs 80 --distil_epochs 80 --ft_batch_size 128 --distil_batch_size 512 --teacher_layer 6 --distil_chunk_size 700000 --pt_teacher TFXLMRobertaModel --pt_teacher_checkpoint jplu/tf-xlm-roberta-large --do_NER --transfer_file unlabeled_sentences.txt```

Arguments

```- refer to code for detailed arguments
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
- PYTHONHASHSEED to seed random number generators for reproducibility
- pt_teacher and pt_teacher_checkpoint for HuggingFace pre-trained transformer models (https://huggingface.co/transformers/)
	-- pre-declared set of MODELS in huggingface_utils (extend for new models)
	-- for checkpoints refer to https://huggingface.co/transformers/pretrained_models.html
```

If you use this code, please cite:
```
@inproceedings{mukherjee-hassan-awadallah-2020-xtremedistil,
    title = "{X}treme{D}istil: Multi-stage Distillation for Massive Multilingual Models",
    author = "Mukherjee, Subhabrata  and
      Hassan Awadallah, Ahmed",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.202",
    pages = "2221--2234",
    abstract = "Deep and large pre-trained language models are the state-of-the-art for various natural language processing tasks. However, the huge size of these models could be a deterrent to using them in practice. Some recent works use knowledge distillation to compress these huge models into shallow ones. In this work we study knowledge distillation with a focus on multilingual Named Entity Recognition (NER). In particular, we study several distillation strategies and propose a stage-wise optimization scheme leveraging teacher internal representations, that is agnostic of teacher architecture, and show that it outperforms strategies employed in prior works. Additionally, we investigate the role of several factors like the amount of unlabeled data, annotation resources, model architecture and inference latency to name a few. We show that our approach leads to massive compression of teacher models like mBERT by upto 35x in terms of parameters and 51x in terms of latency for batch inference while retaining 95{\%} of its F1-score for NER over 41 languages.",
}
```

