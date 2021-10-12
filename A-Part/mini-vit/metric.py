"""
Online evalution metric module.
reference from  https://github.com/apache/incubator-mxnet/blob/v1.4.x/python/mxnet/metric.py
Modify mxnet api to pytorch api.
"""
import math
from collections import OrderedDict

import numpy
import torch

def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `Tensor`
        The labels of the data.

    preds : list of `Tensor`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single Tensor

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, torch.Tensor):
            labels = [labels]
        if isinstance(preds, torch.Tensor):
            preds = [preds]

    return labels, preds


class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(``**config``)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> Tensor
            name to array mapping for labels.

        preds : OrderedDict of str -> Tensor
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `Tensor`
            The labels of the data.

        preds : list of `Tensor`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):
        """Resets the local portion of the internal evaluation results to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_global(self):
        """Gets the current global evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self._has_global_stats:
            if self.global_num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.global_sum_metric / self.global_num_inst)
        else:
            return self.get()

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):
        """Returns zipped name and value pairs for global results.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()

class Accuracy(EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [torch.Tensor([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [torch.Tensor([0, 1, 1])]
    >>> acc = taivision.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `Tensor`
            The labels of the data with class indices as values, one per sample.

        preds : list of `Tensor`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.detach().numpy().astype('int32')
            label = label.detach().numpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)

