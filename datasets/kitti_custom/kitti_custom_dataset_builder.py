"""kitti_custom dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from imageio import imread
from path import Path
import pandas as pd
from data.pose_tranfer import *

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

def load_as_uint(path):
    return imread(path).astype(np.uint8)

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


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kitti_custom dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    self.imu_range = [-10, 0]
    self.seq_len = 5
    self.img_shape = (256, 832)

    """
        'train': <SplitInfo num_examples=20773, num_shards=256>,
        'validation': <SplitInfo num_examples=2810, num_shards=32>,
    """
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Sequence(tfds.features.Image(shape=(*self.img_shape, 3), dtype=tf.uint8), length=self.seq_len),
            'imus': tfds.features.Tensor(shape=(self.seq_len, 11, 6), dtype=tf.float32),
            'gts': tfds.features.Tensor(shape=(self.seq_len, 4, 4), dtype=tf.float32),
            'intrinsics': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
            'inv_intrinsics': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        }),
        disable_shuffling=True,
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/chansoopark98/UnDeep-VIO/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    root_path = '/media/park-ubuntu/park_file/dataset/KITTI_rec_256'
    self.root = Path(root_path)
    train_scenes = [self.root/folder for folder in train_set]
    test_sceness = [self.root/folder for folder in test_set]

    return {
        'train': self._generate_examples(train_scenes, 'train'),
        'validation': self._generate_examples(test_sceness, 'validation'),
    }

  def _generate_examples(self, scenes, mode):
    sequence_set = []
    demi_length = (self.seq_len-1)//2
    shifts = list(range(-demi_length, demi_length+1))
    
    for scene in scenes:
        imgs = sorted((scene).files('*.jpg'))
        intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))   
        try:
            imus = np.load(scene/'sampled_imu_{}_{}.npy'.format(self.imu_range[0], self.imu_range[1]), allow_pickle=True).astype(np.float32)
        except EOFError as e:
            print("No npy files 'sampled_imu_{}_{}.npy' as commmand specified".format(self.imu_range[0], self.imu_range[1]))
        GT = np.array(pd.read_csv(scene/'poses.txt', sep=' ', header=None))

        for i in range(demi_length, len(imgs)-demi_length):
            sample = {'imgs':[], 'imus':[], 'intrinsics': intrinsics, 'gts': []}
            seq_GT  = []
            # put the target in the middle
            for j in shifts:
                sample['imgs'].append(imgs[i+j])
                sample['imus'].append(imus[i+j])
                seq_GT.append(GT[i+j])
            seq_GT = absolute2Relative(seq_GT)
            sample['gts'].append(seq_GT)
            sequence_set.append(sample)
    
    print(f'Current model : {mode}, num_samples : {len(sequence_set)}')

    for idx in range(len(sequence_set)):
        sample = sequence_set[idx]
        imgs = [load_as_uint(img) for img in sample['imgs']]
        imus = np.copy(sample['imus'])
        gts = np.squeeze(np.array(sample['gts'])).astype(np.float32) # (5, 4, 4)
        intrinsics = np.copy(sample['intrinsics'])
        inv_intrinsics = np.linalg.inv(intrinsics)

        batch = {
            'image': imgs,
            'imus': imus,
            'gts': gts, 
            'intrinsics': intrinsics,
            'inv_intrinsics': inv_intrinsics
        }
        yield idx, batch
        