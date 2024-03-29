import torch 
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
import tensorflow_graphics as tfg
from utils.grid_sample import grid_sampler
from tensorflow_graphics.image.transformer import sample
from utils.inverse_warp import *

device = torch.device('cpu')

def inverse_warp_3d(img, depth, pose, intrinsics, rotation_mode='euler', padding_mode='zeros', ref_depth=None):
    batch_size, h, w, _ = img.shape

    cam_coords = pixel2cam_tf(depth, b_inv_tf(intrinsics)) # (B, h, w, 3)

    pose_mat = pose

    proj_cam_to_src_pixel = intrinsics @ pose_mat # (b, 3, 4)

    src_pixel_coords = cam2pixel_tf(cam_coords,
                                    proj_cam_to_src_pixel[:, :, :3],
                                    proj_cam_to_src_pixel[:, :, -1:],
                                    padding_mode) # (4, 256, 832, 2)
    
    projected_img = grid_sampler(img, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

    if ref_depth is not None:
        cam_coords_flat = tf.reshape(cam_coords, shape=[batch_size, h*w, 3]) # (4, 212992, 3)
        pcoords = tf.linalg.matmul(pose_mat[:, :, :3], cam_coords_flat, transpose_b=True)

        pcoords = pcoords + pose_mat[:, :, -1:]
        pcoords = tf.reshape(pcoords, shape=[batch_size, h, w, 3]) # (4, 256, 832, 3)

        ref_coords_3d = pixel2cam_tf(ref_depth, b_inv_tf(intrinsics)) # (4, 256, 832, 3)
        projected_3d_points = grid_sampler(ref_coords_3d, src_pixel_coords, padding_mode=padding_mode, align_corners=True)

        return projected_img, [pcoords, projected_3d_points]
        
    return projected_img, None
        

if __name__ == '__main__':
    # Load Inputs
    img = np.load('./img_npy.npy') # (4, 3, 256 ,832)
    depth = np.load('./depth_npy.npy') # (4, 256, 832)
    pose = np.load('./pose_npy.npy') # (4, 3, 4)
    intrinsics = np.load('./intrinsics_npy.npy') # (4, 3, 3)
    ref_depth = np.load('./ref_depth_npy.npy') # (4, 256, 832)

    print(f'img shape {img.shape}')
    print(f'depth shape {depth.shape}')
    print(f'pose shape {pose.shape}')
    print(f'intrinsics shape {intrinsics.shape}')
    print(f'ref_depth shape {ref_depth.shape}')

    # Load gt
    projected_img = np.load('./result_projected_img.npy')
    pcoords = np.load('./result_pcoords.npy')
    projected_3d_points = np.load('./result_projected_3d_points.npy')

    print(f'projected_img shape {projected_img.shape}') # (4, 3, 256, 832)
    print(f'pcoords shape {pcoords.shape}') # (4, 3, 256, 832)
    print(f'projected_3d_points shape {projected_3d_points.shape}') # (4, 3, 256, 832)


    tf_img = tf.convert_to_tensor(img)
    tf_depth = tf.convert_to_tensor(depth)
    tf_pose = tf.convert_to_tensor(pose)
    tf_intrinsics = tf.convert_to_tensor(intrinsics)
    tf_ref_depth = tf.convert_to_tensor(ref_depth)

    # transpose image (b, 3, h, w) -> (b, h, w, 3)
    tf_img = tf.transpose(tf_img, perm=[0, 2, 3, 1])

    tf_projected_img, tf_points = inverse_warp_3d(tf_img, depth, pose, intrinsics, ref_depth=ref_depth)
    tf_pcoords, tf_projected_3d_points = tf_points

    # result shape
    print(f'tf result : projected_img shape {tf_projected_img.shape}') # (4, 256, 832, 3)
    print(f'tf result : pcoords shape {tf_pcoords.shape}') # (4, 256, 832, 3)
    print(f'tf result : projected_3d_points shape {tf_projected_3d_points.shape}') # (4, 256, 832, 3)

    tf_projected_img = tf.transpose(tf_projected_img, perm=[0, 3, 1, 2])
    tf_pcoords = tf.transpose(tf_pcoords, perm=[0, 3, 1, 2])
    tf_projected_3d_points = tf.transpose(tf_projected_3d_points, perm=[0, 3, 1, 2])

    print(tf.reduce_mean(projected_img - tf_projected_img.numpy()))
    print(tf.reduce_mean(tf_pcoords - tf_pcoords.numpy()))
    print(tf.reduce_mean(tf_projected_3d_points - tf_projected_3d_points.numpy()))



