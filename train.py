__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."
# SPDX-License-Identifier: MIT

import datetime
import logging
import os

import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from modelnet_provider import ModelNetProvider
from model import pointnet_cls


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    train_list = 'modelnet40_ply_hdf5_2048/train_files.txt'
    test_list = 'modelnet40_ply_hdf5_2048/test_files.txt'
    weights_path = os.path.expanduser('~/.keras/models/pointnet_modelnet.h5')

    log_dir = os.path.expanduser('~/.keras/logs/modelnet')
    log_dir = os.path.expanduser(log_dir)
    folder_name = 'pointnet_modelnet_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(log_dir, folder_name)
    os.makedirs(log_dir)

    epochs = 100
    input_size = 2048
    batch_size = 32
    num_classes = 40

    model = pointnet_cls(input_shape=(input_size, 3), classes=num_classes, activation='softmax')
    loss = 'sparse_categorical_crossentropy'
    metric = ['sparse_categorical_accuracy']
    monitor = 'val_loss'

    # tensorboard and weights saving callbacks
    callbacks = list()
    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True))
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5, verbose=1, min_lr=1e-10))
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=10))
    callbacks.append(keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='auto', period=1))

    # prepare dataset
    train_dataset = ModelNetProvider(train_list, input_size=input_size)
    train_generator = train_dataset.generate_samples(batch_size=batch_size, augmentation=True, shuffle=True)
    train_steps_per_epoch = (train_dataset.x.shape[0] // batch_size) + 1

    val_dataset = ModelNetProvider(test_list, input_size=input_size)
    val_generator = val_dataset.generate_samples(batch_size=batch_size, augmentation=False)
    val_steps_per_epoch = (val_dataset.x.shape[0] // batch_size) + 1

    optimizer = Adam(lr=3e-4)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    # train
    # model.summary()
    history = model.fit_generator(train_generator, train_steps_per_epoch,
                                  validation_data=val_generator, validation_steps=val_steps_per_epoch,
                                  epochs=epochs, callbacks=callbacks)
