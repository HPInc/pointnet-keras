__copyright__ = "Copyright (C) 2019 HP Development Company, L.P."

import h5py
import numpy as np
import os


class ModelNetProvider(object):
    def __init__(self, h5_path, input_size=None):
        """
        Class initializer for PointNet data
        :param h5_path: path to database
        :param input_size: size of the point clouds
        """
        self.input_size = input_size
        self.h5_path = h5_path
        self.x, self.y = self.load_list()

    def load_data(self, path):
        # set x and y
        h5 = h5py.File(path, 'r')
        if self.input_size:
            x = h5['data'][:, 0:self.input_size, :]
        else:
            x = h5['data'][()]
        y = h5['label'][()]
        h5.close()

        return x, y

    def load_list(self):
        folder = os.path.dirname(self.h5_path)
        file = open(self.h5_path, 'r')
        x = []
        y = []
        for line in file.readlines():
            path = os.path.join(folder, os.path.basename(line.rstrip('\r\n')))
            x_i, y_i = self.load_data(path)
            if x == [] and y == []:
                x = x_i
                y = y_i
            else:
                x = np.vstack([x, x_i])
                y = np.vstack([y, y_i])
        file.close()

        return x, y

    def generate_samples(self, batch_size, augmentation=False, shuffle=False):
        """
        Sample generator for training
        :param batch_size: size of the batch to yield
        :param augmentation: perform data augmentation
        :param shuffle: whether to shuffle the data before feeding it to the network
        :return:
        """
        num_batches = self.x.shape[0] // batch_size
        while True:
            epoch_indices = np.arange(self.x.shape[0])
            if shuffle:
                np.random.shuffle(epoch_indices)
            for i in range(num_batches):
                batch_indices = epoch_indices[0:batch_size]
                epoch_indices = epoch_indices[batch_size:]
                yield self.get_batch(batch_indices, augmentation)

            if epoch_indices.size:
                yield self.get_batch(epoch_indices, augmentation)

    def rotate_point_cloud_by_angle(self, batch_data):
        """ Rotate the point cloud along up direction with certain angle.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, batch_data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data

    def augment_batch(self, x_batch):
        x_batch = self.rotate_point_cloud_by_angle(x_batch)
        x_batch = self.jitter_point_cloud(x_batch)
        return x_batch

    def get_batch(self, indices, augmentation):
        """
        Grab batch from dataset
        :param indices: indices from data to use as batch
        :param augmentation: perform augmentation of the batch data
        :return: x_batch, y_batch
        """
        x_batch = np.copy(self.x[indices])
        y_batch = np.copy(self.y[indices])

        # augment
        if augmentation:
            x_batch = self.augment_batch(x_batch)

        return x_batch, y_batch
