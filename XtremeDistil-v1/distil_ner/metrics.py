#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# Copyright 2016 Google
# Copyright 2019 The BioNLP-HZAU
# Time:2019/04/08
"""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
import numpy as np

def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.
    If running in a `DistributionStrategy` context, the variable will be
    "tower local". This means:
    *   The returned object will be a container with separate variables
        per replica/tower of the model.
    *   When writing to the variable, e.g. using `assign_add` in a metric
        update, the update will be applied to the variable local to the
        replica/tower.
    *   To get a metric's result value, we need to sum the variable values
        across the replicas/towers before computing the final answer.
        Furthermore, the final answer should be computed once instead of
        in every replica/tower. Both of these are accomplished by
        running the computation of the final result value inside
        `tf.contrib.distribution_strategy_context.get_tower_context(
        ).merge_call(fn)`.
        Inside the `merge_call()`, ops are only added to the graph once
        and access to a tower-local variable in a computation returns
        the sum across all replicas/towers.
    Args:
        shape: Shape of the created variable.
        dtype: Type of the created variable.
        validate_shape: (Optional) Whether shape validation is enabled for
        the created variable.
        name: (Optional) String name of the created variable.
    Returns:
        A (non-trainable) variable initialized to zero, or if inside a
        `DistributionStrategy` scope a tower-local variable container.
    """
    # Note that synchronization "ON_READ" implies trainable=False.
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
        ],
        validate_shape=validate_shape,
        synchronization=variable_scope.VariableSynchronization.ON_READ,
        aggregation=variable_scope.VariableAggregation.SUM,
        name=name)

def streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.
    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.
    Args:
        labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
        predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
        num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
    Returns:
        total_cm: A `Tensor` representing the confusion matrix.
        update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return (total_cm, update_op)


def calculate(total_cm, label_list):
    num_class = len(label_list)
    label2id = {}
    id2label = {}
    for (i,label) in enumerate(label_list):
        label2id[label] = i
        id2label[i] = label
    print (label2id)
    print (id2label)
    lang_list = set([label.split(":")[0] for label in label_list])
    print (lang_list)

    for lang in lang_list:
        labels = [label for label in label_list if lang+":" in label and lang+":O" not in label]
        print (lang, labels)
        ids = [label2id[label] for label in labels]
        print (ids)
        precisions = []
        recalls = []
        fs = []

        for i in ids:
            rowsum, colsum = np.sum(total_cm[i]), np.sum(total_cm[r][i] for r in range(num_class))
            precision = total_cm[i][i] / float(colsum+1e-12)
            recall = total_cm[i][i] / float(rowsum+1e-12)
            f = 2 * precision * recall / (precision + recall+1e-12)
            precisions.append(precision)
            recalls.append(recall)
            fs.append(f)

        print ("***Lang***", lang, str(np.mean(precisions)), str(np.mean(recalls)), str(np.mean(fs)))


    precisions = []
    recalls = []
    fs = []

    labels = [id2label[_id] for _id in range(num_class-4)]
    print (labels)

    for i in range(num_class-4):
        rowsum, colsum = np.sum(total_cm[i]), np.sum(total_cm[r][i] for r in range(num_class))
        precision = total_cm[i][i] / float(colsum+1e-12)
        recall = total_cm[i][i] / float(rowsum+1e-12)
        f = 2 * precision * recall / (precision + recall+1e-12)
        precisions.append(precision)
        recalls.append(recall)
        fs.append(f)
        print ("***Class***", id2label[i], str(precision), str(recall), str(f))

    return (np.mean(precisions), np.mean(recalls), np.mean(fs))
        
        
        
        
        
        

