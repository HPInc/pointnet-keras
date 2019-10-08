# Pointnet Keras

Keras implementation of the PointNet 3D classification network: https://github.com/charlesq34/pointnet

- Includes orthogonal regulatization of features
- Optimized for fast training (1D convolutions and fast 2D max-pooling)
- Achieves similar rates of original paper (~88%-89% accuracy)

## Data Preparation

Download and unzip the modelnet dataset:

```
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
```

## Training

Tested on Tensorflow 1.14 with tf.keras. Originally designed using Keras 2.2.4.

Simply run the training script to proceed. The best model will be saved with respect to the validation loss.

```
python train.py
```

