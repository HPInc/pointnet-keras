__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."
# SPDX-License-Identifier: MIT

import keras.backend as K
from keras.regularizers import Regularizer

import numpy as np


class OrthogonalRegularizer(Regularizer):
    """
    Considering that input is flattened square matrix X, regularizer tries to ensure that matrix X
    is orthogonal, i.e. ||X*X^T - I|| = 0. L1 and L2 penalties can be applied to it
    """
    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        size = int(np.sqrt(x.shape[1].value))
        assert(size * size == x.shape[1].value)
        x = K.reshape(x, (-1, size, size))
        xxt = K.batch_dot(x, x, axes=(2, 2))
        regularization = 0.0
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(xxt - K.eye(size)))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(xxt - K.eye(size)))

        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}


def orthogonal(l1=0.0, l2=0.0):
    """
    Functional wrapper for OrthogonalRegularizer.
    :param l1: l1 penalty
    :param l2: l2 penalty
    :return: Orthogonal regularizer to append to a loss function
    """
    return OrthogonalRegularizer(l1=l1, l2=l2)
