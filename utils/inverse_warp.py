import torch 
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
import tensorflow_graphics as tfg
from utils.grid_sample import grid_sampler
from tensorflow_graphics.image.transformer import sample

device = torch.device('cpu')

pixel_coords = None

# 검증완료
def euler2mat_tf(angle):
    """Convert euler angles to rotation matrix in TensorFlow.

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = tf.shape(angle)[0]
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = tf.cos(z)
    sinz = tf.sin(z)

    zeros = tf.zeros_like(z)
    ones = tf.ones_like(z)
    zmat = tf.stack([cosz, -sinz, zeros,
                     sinz,  cosz, zeros,
                     zeros, zeros, ones], axis=1)
    zmat = tf.reshape(zmat, [B, 3, 3])

    cosy = tf.cos(y)
    siny = tf.sin(y)

    ymat = tf.stack([cosy, zeros, siny,
                     zeros, ones, zeros,
                     -siny, zeros, cosy], axis=1)
    ymat = tf.reshape(ymat, [B, 3, 3])

    cosx = tf.cos(x)
    sinx = tf.sin(x)

    xmat = tf.stack([ones, zeros, zeros,
                     zeros, cosx, -sinx,
                     zeros, sinx, cosx], axis=1)
    xmat = tf.reshape(xmat, [B, 3, 3])

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat

# 검증완료
def pose_vec2mat4_tf(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """

    vec = tf.convert_to_tensor(vec)
    vec = tf.cast(vec, tf.float32)

    translation = vec[:, 3:]
    translation = tf.expand_dims(translation, axis=-1)
    rot = vec[:, :3]

    rot_mat = euler2mat_tf(rot)

    # Assuming rot_mat is [B, 3, 3] and translation is [B, 3, 1]
    transform_mat = tf.concat([rot_mat, translation], axis=2)  # [B, 3, 4]

    # Creating the additional row to be added to make it a homogeneous transformation matrix
    add = tf.constant([[0., 0., 0., 1.]], dtype=tf.float32)
    add = tf.reshape(add, [1, 1, 4])
    add = tf.tile(add, [tf.shape(transform_mat)[0], 1, 1])  # Repeat it for each item in the batch

    print(add, add.shape)
    # Concatenating the additional row to the transformation matrix
    transform_mat = tf.concat([transform_mat, add], axis=1)  # [B, 4, 4]
    return transform_mat

# 검증 완료
def b_inv_tf(b_mat):
    b_inv = tf.linalg.inv(b_mat)
    return b_inv

# 검증 완료
def out2posew_tf(out):
    seq_len = out.shape[1]
    pose = [pose_vec2mat4_tf(out[:, i]) for i in range(seq_len)]
    pose = pose[:seq_len//2] + [b_inv_tf(p) for p in pose[seq_len//2:]]
    pose = tf.stack(pose, axis=1)
    return pose[:, :, :3, :]

# 검증 완료
def set_id_grid_tf(depth, channel_last=True):
    global pixel_coords
    b, h, w = depth.shape
    # tf.meshgrid로 i, j 범위 생성
    i_range, j_range = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    
    # depth와 동일한 dtype으로 변환
    i_range = tf.cast(i_range, depth.dtype)
    j_range = tf.cast(j_range, depth.dtype)
    
    ones = tf.ones_like(i_range, dtype=depth.dtype)
    
    # [H, W, 3] 형태로 스택
    pixel_coords = tf.stack([j_range, i_range, ones], axis=-1)  # [H, W, 3]
    
    # [1, H, W, 3]로 reshape하여 배치 차원 추가
    pixel_coords = tf.reshape(pixel_coords, [1, h, w, 3])
    
    if not channel_last:
        # [1, 3, H, W]로 transpose하여 channel_first 형태로 변경
        pixel_coords = tf.transpose(pixel_coords, [0, 3, 1, 2])
    
    return pixel_coords

# 검증 완료
def pixel2cam_tf(depth, intrinsics_inv):
    global pixel_coords
    b, h, w = depth.shape
    if pixel_coords is None or pixel_coords.shape[1] < h:
        set_id_grid_tf(depth)

    current_pixel_coords = tf.reshape(pixel_coords[:, :h, :w, :], [b, -1, 3])  # [B, H*W, 3]
    cam_coords = tf.linalg.matmul(current_pixel_coords, intrinsics_inv, transpose_b=True)
    cam_coords = tf.reshape(cam_coords, [-1, h, w, 3]) # 결과를 [B, H, W, 3] 형태로 재조정

    return cam_coords * tf.expand_dims(depth, axis=-1)

if __name__ == '__main__':
    # raw_vec = [[0.1, 0.2, 0.3, 0.9, 0.9, 0.9], [0.1, 0.2, 0.3, 0.9, 0.9, 0.9]]
    # raw_vec = np.array(raw_vec)
    # print(raw_vec.shape)

    vec = np.load('./test.npy')
    print(vec.shape)

    img = np.load('./grid_img.npy')
    src = np.load('./grid_src.npy')

    torch_img = torch.from_numpy(img.copy())
    torch_src = torch.from_numpy(src.copy())

    print(torch_img.shape)
    print(torch_src.shape)

    projected_torch = F.grid_sample(torch_img, torch_src, padding_mode='zeros', align_corners=True).numpy()

    
    tf_img = tf.convert_to_tensor(img.copy())
    print(tf_img.shape)
    tf_img = tf.transpose(tf_img,  perm=[0, 2, 3, 1])
    print('tf_img', tf_img.shape)
    tf_src = tf.convert_to_tensor(src.copy())
    print('tf_src', tf_src.shape)

    
    
    # projected_tf = sample(tf_img, tf_src, border_type=tfg.image.transformer.BorderType.ZERO,
    #                       pixel_type=tfg.image.transformer.PixelType.HALF_INTEGER)
    
    projected_tf = grid_sampler(tf_img, tf_src, align_corners=True)
    projected_tf = tf.transpose(projected_tf,  perm=[0, 3, 1, 2])
    print(projected_tf.shape)

    print(projected_torch - projected_tf.numpy())