__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."
# SPDX-License-Identifier: MIT

import keras.backend as K
from keras.layers import MaxPooling2D, Flatten, Lambda, Dense, Reshape
from keras.initializers import Constant

import numpy as np

from utils import conv1d_bn, dense_bn
from regularizers import orthogonal


def transform_net(inputs, scope=None, regularize=False):
    """
    Generates an orthogonal transformation tensor for the input data
    :param inputs: tensor with input image (either BxNxK or BxNx1xK)
    :param scope: name of the grouping scope
    :param regularize: enforce orthogonality constraint
    :return: BxKxK tensor of the transformation
    """
    with K.name_scope(scope):

        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]
        num_points = input_shape[-2]

        net = conv1d_bn(inputs, num_filters=64, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid',
                        use_bias=True, scope='tconv3')

        #  Done in 2D since 1D is painfully slow
        net = MaxPooling2D(pool_size=(num_points, 1), padding='valid')(Lambda(K.expand_dims)(net))
        net = Flatten()(net)

        net = dense_bn(net, units=512, scope='tfc1', activation='relu')
        net = dense_bn(net, units=256, scope='tfc2', activation='relu')

        transform = Dense(units=k * k,
                          kernel_initializer='zeros', bias_initializer=Constant(np.eye(k).flatten()),
                          activity_regularizer=orthogonal(l2=0.001) if regularize else None)(net)
        transform = Reshape((k, k))(transform)

    return transform
