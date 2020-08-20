"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

import conlleval
import logging
import numpy as np

logger = logging.getLogger('xtremedistil')

def ner_evaluate(model, x_test, y_test, labels, special_tokens, MAX_SEQUENCE_LENGTH, batch_size=32):

    total = []
    for lang in x_test.keys():
        y_pred = model.predict(x_test[lang], batch_size=batch_size)
        pred_tags_all = []
        true_tags_all = []
        for i, seq in enumerate(y_pred):
            for j in range(MAX_SEQUENCE_LENGTH):
                indx = y_test[lang][i][j]
                true_label = labels[indx]
                if special_tokens["pad_token"] in true_label or special_tokens["bos_token"] in true_label or special_tokens["eos_token"] in true_label:
                    continue
                true_tags_all.append(true_label)
                indx = np.argmax(seq[j])
                pred_label = labels[indx]
                pred_tags_all.append(pred_label)
        prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, special_tokens, verbose=True)
        logger.info ("Lang {} scores {} {} {}".format(lang, prec, rec, f1))
        total.append(f1)
    logger.info ("All f-scores {}".format(total))
    logger.info ("Overall average f-score mean {} and variance {}".format(np.mean(total), np.var(total)))
    return np.mean(total)


def classify_evaluate(model, x_test, y_test, batch_size=32):

    total = []
    for lang in x_test.keys():
        y_pred = np.argmax(model.predict(x_test[lang], batch_size=batch_size), axis=-1)
        acc = (y_pred.flatten() == y_test[lang].flatten()).sum()/len(y_test[lang])
        logger.info ("Lang {} accuracy {}".format(lang, acc))
        total.append(acc)
    logger.info ("All accuracies {}".format(total))
    logger.info ("Overall average accuracy mean {} and variance {}".format(np.mean(total), np.var(total)))
    return np.mean(total)
