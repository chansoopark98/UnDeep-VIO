import tensorflow as tf
import numpy as np
import pandas as pd
import random
from imageio import imread
from path import Path
import random
import pandas as pd
import time
from pose_tranfer import *

train_set = {'2011_10_03_drive_0042_sync_02':'01',
            '2011_10_03_drive_0034_sync_02':'02',
            '2011_10_03_drive_0027_sync_02':'00',
            '2011_09_30_drive_0028_sync_02':'08',
            '2011_09_30_drive_0027_sync_02':'07',
            '2011_09_30_drive_0020_sync_02':'06',
            '2011_09_30_drive_0018_sync_02':'05',
            '2011_09_30_drive_0016_sync_02':'04'
            }
test_set = {'2011_09_30_drive_0033_sync_02':'09',
            '2011_09_30_drive_0034_sync_02':'10'}

def load_as_float(path):
    return imread(path).astype(np.float32)

def pose12to16(mat):
    if mat.ndim == 1:
        mat = mat.reshape(3, -1)
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat
    else:
        mat = np.vstack([mat, [0, 0, 0, 1]])
        return mat

def mat_to_6dof(mat):
    if mat.shape[0] == 3:
        mat = pose12to16(mat)
    else:
        translation = list(mat[:3,3])
        rotation = list(euler_from_matrix(mat))
        pose = rotation + translation
    return pose

def absolute2Relative(seqGT):
    sequence_length = len(seqGT)
    seqGT_mat = [pose12to16(item) for item in seqGT]
    seqGT_Rela_mat = []
    seqGT_Rela_mat.append(seqGT_mat[0])
    seqGT_Rela_Eul = []
    seqGT_Rela_Eul.append(mat_to_6dof(seqGT_mat[0]))
    for i in range(1, sequence_length):
        seqGT_Rela_mat.append(np.linalg.inv(seqGT_mat[i-1]) @ seqGT_mat[i])
    seqGT_Rela_mat = np.array(seqGT_Rela_mat)
    return seqGT_Rela_mat

class DataSequence(tf.data.Dataset):
    def __new__(cls, root, seed=None, train=True, sequence_length=3,
                imu_range=[0, 0], transform=None, shuffle=True,
                scene='default', image_width=640, image_height=480):
        
        # 데이터셋 초기화 및 샘플 로딩
        self = tf.data.Dataset.__new__(cls)
        np.random.seed(seed)
        random.seed(seed)
        root = Path(root)
        settype = scene
        scene_list = train_set if train else test_set if settype == 'default' else scene
        scenes = [root/folder for folder in scene_list]
        samples = cls.crawl_folders(scenes, sequence_length, imu_range)
        
        # TensorFlow Dataset 생성
        dataset = tf.data.Dataset.from_tensor_slices(samples)
        
        def _load_data(sample):
            # 샘플 로딩 및 전처리 로직 구현
            # TensorFlow 함수와 메서드를 사용하여 데이터 로드 및 전처리 수행
            pass
        
        dataset = dataset.map(_load_data)
        return dataset
    
    @staticmethod
    def crawl_folders(scenes, sequence_length, imu_range):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length+1))
        
        for scene in scenes:
            imgs = sorted((scene).files('*.jpg'))
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))   
            try:
                imus = np.load(scene/'sampled_imu_{}_{}.npy'.format(imu_range[0], imu_range[1]), allow_pickle=True).astype(np.float32)
            except EOFError as e:
                print("No npy files 'sampled_imu_{}_{}.npy' as commmand specified".format(imu_range[0], imu_range[1]))
            GT = np.array(pd.read_csv(scene/'poses.txt', sep=' ', header=None))

            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'imgs':[], 'imus':[], 'intrinsics': intrinsics, 'gts': []}
                seq_GT = []
                # put the target in the middle
                for j in shifts:
                    sample['imgs'].append(imgs[i+j])
                    sample['imus'].append(imus[i+j])
                    seq_GT.append(GT[i+j])
                seq_GT = absolute2Relative(seq_GT)
                sample['gts'].append(seq_GT)
                sequence_set.append(sample)
                
        samples = sequence_set
        return samples

if __name__ == '__main__':
    # 사용 예
    root_path = '/media/park-ubuntu/park_file/dataset/KITTI_rec_256/'
    dataset = DataSequence(root=root_path, train=True, sequence_length=5)
