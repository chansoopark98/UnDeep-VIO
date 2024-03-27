import torch 
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
import tensorflow_graphics as tfg
from utils.grid_sample import grid_sampler
from tensorflow_graphics.image.transformer import sample
from utils.inverse_warp import *

device = torch.device('cpu')

# TODO
# cam2pixel 구현

if __name__ == '__main__':
    # raw_vec = [[0.1, 0.2, 0.3, 0.9, 0.9, 0.9], [0.1, 0.2, 0.3, 0.9, 0.9, 0.9]]
    # raw_vec = np.array(raw_vec)
    # print(raw_vec.shape)

    raw_depth = np.load('./npy_depth.npy')
    raw_intrinsic_inv = np.load('./npy_intrinsics_inv.npy')
    
    torch_depth = torch.from_numpy(raw_depth.copy())
    torch_intrinsic_inv = torch.from_numpy(raw_intrinsic_inv.copy())

    tf_depth = tf.convert_to_tensor(raw_depth.copy())
    tf_intrinsic_inv = tf.convert_to_tensor(raw_intrinsic_inv.copy())

    torch_result = pixel2cam(torch_depth, torch_intrinsic_inv)
    tf_result = pixel2cam_tf(tf_depth, tf_intrinsic_inv)

    # print(torch_result)
    print(torch_result.shape) # 4, 3, 256, 832


    # print(tf_result)
    print(tf_result.shape) # 4, 3, 256, 832

    tf_result = tf.transpose(tf_result, perm=[0, 3, 1, 2])

    print(tf.reduce_mean(torch_result - tf_result))

    