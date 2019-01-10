#
# Copyright (C) 2017 HP Development Company, L.P.
#
# NOTICE:  All information contained herein is, and remains the property of HP
# and its affiliates. The intellectual and technical concepts contained herein
# are proprietary to HP and its affiliates and may be covered by U.S. and
# Foreign Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of this
# material is strictly forbidden unless prior written permission is obtained
# from HP.
#

from keras import backend as K
from keras.initializers import Constant
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Activation, \
    Dot, Lambda, Reshape, BatchNormalization, Conv1D
from keras.models import Model
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


def dense_bn(x, units, use_bias=True, scope=None, activation=None):
    """
    Utility function to apply Dense + Batch Normalization.
    """
    with K.name_scope(scope):
        x = Dense(units=units, use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


def conv1d_bn(x, num_filters, kernel_size, padding='same', strides=1,
              use_bias=False, scope=None, activation='relu'):
    """
    Utility function to apply Convolution + Batch Normalization.
    """
    with K.name_scope(scope):
        input_shape = x.get_shape().as_list()[-2:]
        x = Conv1D(num_filters, kernel_size, strides=strides, padding=padding,
                   use_bias=use_bias, input_shape=input_shape)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


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


def pointnet_base(inputs):
    """
    Convolutional portion of pointnet, common across different tasks (classification, segmentation, etc)
    :param inputs: Input tensor with the point cloud shape (BxNxK)
    :return: tensor layer for CONV5 activations
    """

    # Obtain spatial point transform from inputs and convert inputs
    ptransform = transform_net(inputs, scope='transform_net1', regularize=False)
    point_cloud_transformed = Dot(axes=(2, 1))([inputs, ptransform])

    # First block of convolutions
    net = conv1d_bn(point_cloud_transformed, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv2')

    # Obtain feature transform and apply it to the network
    ftransform = transform_net(net, scope='transform_net2', regularize=True)
    net_transformed = Dot(axes=(2, 1))([net, ftransform])

    # Second block of convolutions
    net = conv1d_bn(net_transformed, num_filters=64, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv4')
    net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid',
                    use_bias=True, scope='conv5')

    return net


def pointnet(input_shape, classes):
    """
    PointNet model for object classification
    :param input_shape: shape of the input point clouds (NxK)
    :param classes: number of classes in the classification problem
    :return: Keras model of the classification network
    """

    assert K.image_data_format() == 'channels_last'
    num_point = input_shape[0]

    # Generate input tensor and get base network
    inputs = Input(input_shape, name='Input_cloud')
    net = pointnet_base(inputs)

    # Symmetric function: max pooling
    # Done in 2D since 1D is painfully slow
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid', name='maxpool')(Lambda(K.expand_dims)(net))
    net = Flatten()(net)

    # Fully connected layers
    net = dense_bn(net, units=512, scope='fc1', activation='relu')
    net = Dropout(0.5, name='dp1')(net)
    net = dense_bn(net, units=256, scope='fc2', activation='relu')
    net = Dropout(0.5, name='dp2')(net)
    net = Dense(units=classes, name='fc3', activation='softmax')(net)

    model = Model(inputs, net, name='pointnet_cls')

    return model

