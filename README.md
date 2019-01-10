# Pointnet Keras

Keras implementation of the PointNet 3D classification network: https://github.com/charlesq34/pointnet

## Data Preparation

Download and unzip the modelnet dataset

```
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
```

## Training

Tested in Keras 2.2.4 and Tensorflow 1.9.0

```
python train.py
```
