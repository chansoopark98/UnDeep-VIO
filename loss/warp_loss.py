import torch
import tensorflow as tf
from utils.inverse_warp import inverse_warp_3d
import numpy as np


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))

def disp_smooth_loss(disparities, img):
    loss = 0
    weight = 1.
    for disp in disparities:
        b, h, w, c = disp.shape
        img_scaled = tf.image.resize(img, (h, w), method=tf.image.ResizeMethod.BILINEAR)
        loss += depth_smoothness(disp, img_scaled) * weight
        weight /= 2.3
    return loss

def voloss(pose1, pose2, k=10):
    if pose1.shape[1] == 6:
        rot1, tra1 = pose1[:, :3], pose1[:, 3:6]
        rot2, tra2 = pose2[:, :3], pose2[:, 3:6]
    elif pose1.shape[2] == 6:
        rot1, tra1 = pose1[:, :, :3], pose1[:, :, 3:6]
        rot2, tra2 = pose2[:, :, :3], pose2[:, :, 3:6]

    abs_rot = tf.reduce_mean(tf.abs(rot1 - rot2)) * k
    abs_tra = tf.reduce_mean(tf.abs(tra1 - tra2))

    loss = abs_rot + abs_tra
    return loss

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, pose, 
                                    rotation_mode='euler', padding_mode='zeros', explainability_mask=None, ref_depth=None):
    def one_scale(depth, explainability_mask = None, ref_depth=None):

        reconstruction_loss, loss_3d = 0, 0
        b, h, w, _ = depth.shape

        downscale = tgt_img.shape[1] / h

        tgt_img_scaled = tf.image.resize(tgt_img, (h, w), method=tf.image.ResizeMethod.BILINEAR)
        ref_imgs_scaled = [tf.image.resize(ref_img, (h, w), method=tf.image.ResizeMethod.BILINEAR) for ref_img in ref_imgs]
        intrinsics_scaled = tf.concat([intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]], axis=1)

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i] # pose (4, 2, 3, 4)
            if (ref_depth is not None) and (ref_depth[i] is not None):
                ref_img_warped, coords_3d = inverse_warp_3d(ref_img, depth[:, :, :, 0], current_pose,
                                          intrinsics_scaled, rotation_mode, padding_mode, ref_depth=ref_depth[i][:, :, :, 0])
            else:
                # 기존 파이토치에서 depth는 (b, 1, h, w) 이므로 depth[:,0]를 depth[:, :, :, 0]으로 바꿔줘야 함
                ref_img_warped, _ = inverse_warp_3d(ref_img, depth[:, :, :, 0], current_pose,
                                          intrinsics_scaled, rotation_mode, padding_mode, ref_depth=None)

            out_of_bound = 1 - tf.reduce_prod(tf.cast(ref_img_warped == 0, tf.float32), axis=-1, keepdims=True)
            # print(out_of_bound)
            # print(tf.reduce_mean(out_of_bound))
            diff = appearence_loss(tgt_img_scaled, ref_img_warped, out_of_bound)

            if (ref_depth is not None) and (ref_depth[i] is not None):
                # 수정
                # abs_3d = out_of_bound * tf.abs(coords_3d[0][:, :, :, 2:] - coords_3d[1][:, :, :, 2:])
                # mean_3d = coords_3d[0][:, :, :, 2:] - coords_3d[1][:, :, :, 2:]
                # loss_3d = tf.reduce_mean(abs_3d / mean_3d)

                z_coords_0 = coords_3d[0][..., 2]  # Z 차원만 선택 [B, H, W]
                z_coords_1 = coords_3d[1][..., 2]  # Z 차원만 선택 [B, H, W]

                # 절대값, 덧셈, 나눗셈 연산 수행
                numerator = tf.abs(z_coords_0 - z_coords_1)
                denominator = z_coords_0 + z_coords_1
                
                # out_of_bound를 적용. out_of_bound가 [B, H, W, 1] 형태라고 가정
                out_of_bound = tf.squeeze(out_of_bound, axis=-1)  # 차원 축소 [B, H, W]
                
                # 조건부 연산을 적용하기 전에 out_of_bound와 numerator / denominator를 같은 형태로 만들어야 함
                loss_3d_component = out_of_bound * (numerator / denominator)
                
                # 전체 텐서에 대한 평균을 구합니다.
                loss_3d += tf.reduce_mean(loss_3d_component)

            reconstruction_loss += tf.reduce_mean(diff)
            # assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss/len(ref_imgs), loss_3d/len(ref_imgs)
    
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss_photo, loss_3d = 0, 0

    for i, d in enumerate(depth):
        r_depth = ref_depth[i]
        tmp1, tmp2 = one_scale(d, None, ref_depth=r_depth)
        loss_photo += tmp1
        loss_3d += tmp2
    return loss_photo / len(depth), loss_3d/len(depth)



def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    # batch, height, width, channels = tf.unstack(tf.shape(x))
    # normalization = tf.cast(batch * height * width * channels, tf.float32)

    error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

    if mask is not None:
        error = tf.multiply(mask, error)

    if truncate is not None:
        error = tf.minimum(error, truncate)

    return error # tf.reduce_sum(error) / normalization

def s_SSIM_tf(x, y, window_size):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # TensorFlow에서 평균 풀링을 적용합니다.
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=window_size, strides=1, padding='same')
    mu_x = avg_pool(x)
    mu_y = avg_pool(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = tf.math.pow(mu_x, 2)
    mu_y_sq = tf.math.pow(mu_y, 2)
    
    sigma_x = avg_pool(x * x) - mu_x_sq
    sigma_y = avg_pool(y * y) - mu_y_sq
    sigma_xy = avg_pool(x * y) - mu_x_mu_y
    
    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    
    # 결과를 [0, 1] 범위로 클리핑합니다.
    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def appearence_loss(img1, img2, valid_pixel, ternary=False, weights=[0.15, 0.85, 0.08]):

    diff = img1 - img2 
    diff = charbonnier_loss(diff, valid_pixel, alpha=0.5)
    ssim_loss = s_SSIM_tf(img1*valid_pixel, img2*valid_pixel, window_size=3)

    diff = tf.reduce_mean(diff, axis=3, keepdims=True)
    ssim_loss = tf.reduce_mean(ssim_loss, axis=3, keepdims=True)
    
    return weights[0] * diff + weights[1] * ssim_loss

if __name__ == '__main__':
    # image_1 = tf.ones((4, 256, 832, 3))
    # image_2 = tf.ones((4, 256, 832, 3)) - 0.5
    # mask = tf.ones((4, 256, 832, 1)) - 0.5

    # ssim_loss = s_SSIM_tf(image_1, image_2, 11)
    # print(ssim_loss)
    # print(ssim_loss.shape) # 4, 256, 832, 3

    # char_loss = charbonnier_loss(image_1, mask, None)
    # print(char_loss) 
    # print(char_loss.shape) # () 4, 256, 832, 3

    # appearence = appearence_loss(image_1, image_2, mask)
    # print(appearence)
    # print(appearence.shape)

    # tgt_img = np.load('./npy_tgt_img.npy') # 
    # npy_ref_img_1 = np.load('./npy_ref_img_1.npy')
    # npy_ref_img_2 = np.load('./npy_ref_img_2.npy')
    # npy_intrinsics = np.load('./npy_intrinsics.npy')
    # npy_depth = np.load('./npy_depth.npy')
    # npy_pose = np.load('./npy_pose.npy')

    # print(f'tgt_img shape {tgt_img.shape}')
    # print(f'npy_ref_img_1 shape {npy_ref_img_1.shape}')
    # print(f'npy_ref_img_2 shape {npy_ref_img_2.shape}')
    # print(f'npy_intrinsics shape {npy_intrinsics.shape}')
    # print(f'npy_depth shape {npy_depth.shape}')
    # print(f'npy_pose shape {npy_pose.shape}')

    """
        tgt_img shape (4, 3, 256, 832)
        npy_ref_img_1 shape (4, 3, 256, 832)
        npy_ref_img_2 shape (4, 3, 256, 832)
        npy_intrinsics shape (4, 3, 3)
        npy_depth shape (4, 1, 256, 832)
        npy_pose shape (4, 2, 3, 4)
    """

    # tf_tgt_img = tf.convert_to_tensor(tgt_img)
    # tf_npy_ref_img_1 = tf.convert_to_tensor(npy_ref_img_1)
    # tf_npy_ref_img_2 = tf.convert_to_tensor(npy_ref_img_2)
    # tf_npy_intrinsics = tf.convert_to_tensor(npy_intrinsics)
    # tf_npy_depth = tf.convert_to_tensor(npy_depth)
    # tf_npy_pose = tf.convert_to_tensor(npy_pose)

    # tf_tgt_img = tf.transpose(tf_tgt_img, perm=[0, 2, 3, 1])
    # tf_npy_ref_img_1 = tf.transpose(tf_npy_ref_img_1, perm=[0, 2, 3, 1])
    # tf_npy_ref_img_2 = tf.transpose(tf_npy_ref_img_2, perm=[0, 2, 3, 1])
    # tf_npy_ref_img = [tf_npy_ref_img_1, tf_npy_ref_img_2]
    # tf_npy_depth = tf.transpose(tf_npy_depth, perm=[0, 2, 3, 1])

    # loss_photo, loss_3d = photometric_reconstruction_loss(tgt_img=tf_tgt_img,
    #                                 ref_imgs=tf_npy_ref_img,
    #                                 intrinsics=tf_npy_intrinsics,
    #                                 depth=tf_npy_depth,
    #                                 pose=tf_npy_pose,
    #                                 ref_depth=[None]*4)
    
    # print(loss_photo, loss_3d)
    


