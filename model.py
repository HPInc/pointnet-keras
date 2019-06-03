__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."
# SPDX-License-Identifier: MIT

from keras import backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Dot, Lambda
from keras.models import Model

from .utils import conv1d_bn, dense_bn
from .transform_net import transform_net


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


def pointnet_cls(input_shape, classes):
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
    net = Dropout(0.3, name='dp1')(net)
    net = dense_bn(net, units=256, scope='fc2', activation='relu')
    net = Dropout(0.3, name='dp2')(net)
    net = Dense(units=classes, name='fc3', activation='softmax')(net)

    model = Model(inputs, net, name='pointnet_cls')

    return model
