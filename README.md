# ACL 2020 Microsoft Research, Subhabrata Mukherjee
## Code for XtremeDistil [[Paper]](https://www.microsoft.com/en-us/research/publication/xtremedistil/) [[Video]](https://slideslive.com/38929189/xtremedistil-multistage-distillation-for-massive-multilingual-models)

***New Update 8/19/2020***
Releasing [**XtremeDistil-v2**](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v2) with Tensorflow 2.1 and [HuggingFace Transformers](https://huggingface.co/transformers) for distilling all supported [pre-trained language models](https://huggingface.co/transformers/pretrained_models.html) with an unified API for multilingual text classification and sequence tagging tasks. Refer to [*README*](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v2) for sample usages.


***(Deprecated)*** 
[XtremeDistil-v1](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v1) contains codes for distilling pre-trained language models for multi-lingual Named Entity Recognition (NER) in [distil_ner](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v1/distil_ner) and text classification in [distil_classification](https://github.com/MSR-LIT/XtremeDistil/tree/master/XtremeDistil-v1/distil_classification) with *README* in the corresponding directories. Code based on the original BERT implementation from https://github.com/google-research/bert retaining much of the original configurations, parameter settings and nomenclature.

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

Code is released under [MIT](https://github.com/MSR-LIT/XtremeDistil/blob/master/LICENSE) license.
