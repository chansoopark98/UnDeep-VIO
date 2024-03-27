import tensorflow_datasets as tfds
import tensorflow as tf
import math
from typing import Union
import os
import sys
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler(object):
    def __init__(self, data):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            dataset_name (str)   : Tensorflow dataset name (e.g: 'citiscapes')
        
        """
        self.dataset_name = 'kitti_custom'
        self.norm_mode = 'tf'
        self.data_dir = './data/'

        self.seq_len = 5
        self.img_shape = (256, 832) # (H, W)
        self.__select_dataset()

    def __select_dataset(self):
        self.train_data, self.valid_data = self.__load_custom_dataset()

    def __load_custom_dataset(self):
        train_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train')
        valid_data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='validation')
        return train_data, valid_data

class TspxrTFDSGenerator(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int):
        """
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
            image_size   (tuple) : Model input image resolution 
            batch_size   (int)   : Batch size
            dataset_name (str)   : Tensorflow dataset name (e.g: 'cityscapes')
            norm_type    (str)   : Set input image normalization type (e.g: 'torch')
        """
        # Configuration
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.number_train = math.ceil(46873 / self.batch_size)
        self.number_test = math.ceil(6017 / self.batch_size)
        super().__init__(data=self.data_dir)

    @tf.function(jit_compile=True)
    def prepare_data(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Load RGB images and segmentation labels from the dataset.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().

            Returns:
                (img, labels) (dict) : Returns the image and label extracted from sample as a key value.
        """
        image = tf.cast(sample['image'], dtype=tf.float32)
        imus = tf.cast(sample['imus'], dtype=tf.float32)
        gts = tf.cast(sample['gts'], dtype=tf.float32)
        intrinsics = tf.cast(sample['intrinsics'], dtype=tf.float32)
        inv_intrinsics = tf.cast(sample['inv_intrinsics'], dtype=tf.float32)
        image = tf.image.resize(image, self.image_size,
                                    method=tf.image.ResizeMethod.BILINEAR)
    

        return (image, imus, gts, intrinsics, inv_intrinsics)

    @tf.function(jit_compile=True)
    def preprocess(self, sample: dict) -> Union[tf.Tensor, tf.Tensor]:
        """
            Dataset mapping function to apply to the train dataset.
            Various methods can be applied here, such as image resizing, random cropping, etc.
            Args:
                sample    (dict)  : Dataset loaded through tfds.load().
            
            Returns:
                (img, labels) (dict) : tf.Tensor
        """
        image, imus, gts, intrinsics, inv_intrinsics = self.prepare_data(sample)
         
        return (image, imus, gts, intrinsics, inv_intrinsics)
    
    @tf.function(jit_compile=True)
    def normalize_images(self, image, imus, gts, intrinsics, inv_intrinsics):
        image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode=self.norm_mode)
        return (image, imus, gts, intrinsics, inv_intrinsics)

   
    @tf.function(jit_compile=True)
    def decode_image(self, image):
        # torch 모드에서 사용된 mean과 std 값
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
    
        # 채널별 역정규화
        image *= std
        image += mean
        
        # 픽셀 값의 역스케일링
        image *= 255.0
        image = tf.cast(image, dtype=tf.uint8)
        return image
    
    def get_trainData(self, train_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                train_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                train_data    (tf.data.Dataset)  : Apply data augmentation, batch, and shuffling
        """    
        train_data = train_data.shuffle(1024, reshuffle_each_iteration=True)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.normalize_images, num_parallel_calls=AUTO)
        
        # train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.batch(self.batch_size, drop_remainder=True)
        train_data = train_data.prefetch(AUTO)
        
        # train_data = train_data.repeat()
        return train_data

    def get_testData(self, valid_data: tf.data.Dataset) -> tf.data.Dataset:
        """
            Prepare the Tensorflow dataset (tf.data.Dataset)
            Args:
                valid_data    (tf.data.Dataset)  : Dataset loaded through tfds.load().

            Returns:
                valid_data    (tf.data.Dataset)  : Apply data resize, batch, and shuffling
        """    
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        # valid_data = valid_data.map(self.image_load, num_parallel_calls=AUTO)
        valid_data = valid_data.map(self.normalize_images, num_parallel_calls=AUTO)
        valid_data = valid_data.batch(self.batch_size, drop_remainder=True)
        # valid_data = valid_data.prefetch(AUTO)
        return valid_data
    
if __name__ == '__main__':
    # tf.executing_eagerly()
    # tf.config.run_functions_eagerly(True)
    # tf.config.optimizer.set_jit(False)
    # tf.data.experimental.enable_debug_mode()
    dataset = TspxrTFDSGenerator(data_dir='../data',
                       image_size=(192, 256),
                       batch_size=1)
    
    train_data = dataset.get_trainData(dataset.train_data)
    
    for sample in train_data.take(dataset.number_train):
        img_seq, gt = sample
        gt = gt[0]
        img_seq = img_seq[0]
        img_seq = dataset.decode_image(img_seq)

        # 그림 및 gridspec 생성
        fig = plt.figure(figsize=(10, 15))
        
        gs = gridspec.GridSpec(11, 3)

        # 11장의 이미지를 좌측에 표시
        for i in range(10):
            ax = fig.add_subplot(gs[i, 0], )
            print(gt[i])
            ax.imshow(img_seq[i])
            ax.axis('off')

        plt.show()

        

        # rel_pose = np.vstack(rel_pose)
        # rel_to_global_poses = path_accu(rel_pose)
        # original_poses = origin_pose
        # print('Global pose items : ', len(original_poses))

        # # 시각화 초기 설정
        # fig = plt.figure(figsize=(10, 5))

        # # 이미지를 위한 축
        # ax_img = fig.add_subplot(1, 2, 1)

        # # 3D 궤적을 위한 축
        # ax_traj = fig.add_subplot(1, 2, 2, projection='3d')
        # ax_traj.set_xlabel('X Translation (m)')
        # ax_traj.set_ylabel('Y Translation (m)')
        # ax_traj.set_zlabel('Z Translation (m)')

        # # 3D 축의 초기 범위 설정
        # # ax_traj.set_xlim([np.min(global_poses[0][0, 3]), np.max(global_poses[0][0, 3])])
        # # ax_traj.set_ylim([np.min(global_poses[0][1, 3]), np.max(global_poses[0][1, 3])])
        # # ax_traj.set_zlim([np.min(global_poses[0][2, 3]), np.max(global_poses[0][2, 3])])

        # def update(frame_idx):
        #     # 이미지 축 업데이트
        #     ax_img.clear()  # 이전 이미지를 지웁니다.
        #     ax_img.imshow(frames[frame_idx], cmap='gray')  # frames 리스트에서 이미지를 가져옵니다.
        #     ax_img.set_title(f'Frame {frame_idx}')
        #     ax_img.axis('off')  # 축 레이블을 숨깁니다.

        #     # 3D 궤적 축 업데이트
        #     ax_traj.scatter(original_poses[frame_idx][0, 3], original_poses[frame_idx][1, 3], original_poses[frame_idx][2, 3], color='blue')
        #     ax_traj.scatter(rel_to_global_poses[frame_idx][0, 3], rel_to_global_poses[frame_idx][1, 3], rel_to_global_poses[frame_idx][2, 3], color='green')
        #     if frame_idx > 0:
        #         prev_liens = original_poses[:frame_idx+1]
        #         x_prev = np.asarray([prev_line[0, 3] for prev_line in prev_liens])
        #         y_prev = np.asarray([prev_line[1, 3] for prev_line in prev_liens])
        #         z_prev = np.asarray([prev_line[2, 3] for prev_line in prev_liens])

        #         prev_liens = rel_to_global_poses[:frame_idx+1]
        #         x_rel_prev = np.asarray([prev_line[0, 3] for prev_line in prev_liens])
        #         y_rel_prev = np.asarray([prev_line[1, 3] for prev_line in prev_liens])
        #         z_rel_prev = np.asarray([prev_line[2, 3] for prev_line in prev_liens])

        #         ax_traj.plot(x_prev, y_prev, z_prev, color='red')
        #         ax_traj.plot(x_rel_prev, y_rel_prev, z_rel_prev, color='green')
        #     ax_traj.set_title('Trajectory')

        # ani = FuncAnimation(fig, update, frames=len(frames), interval=500)

        # plt.tight_layout()
        # plt.show()
            
