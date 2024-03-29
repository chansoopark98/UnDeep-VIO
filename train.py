import tensorflow as tf
from data.kitti_loader import TspxrTFDSGenerator
from models.model import *
from models.monodepth2.monodepth2 import DispNet
from tqdm import tqdm
import math
import numpy as np
import gc
from datetime import datetime
import yaml
import os
import tensorflow_graphics.geometry.transformation as tfg
from loss.warp_loss import *
from utils.inverse_warp import *

"""
 This usually means you are trying to call the optimizer to update different parts of the model separately.
 Please call `optimizer.build(variables)`
 with the full list of trainable variables before the training loop or use legacy optimizer `tf.keras.optimizers.legacy.Adam.'
"""
# tensorflow.python.framework.errors_impl.ResourceExhaustedError: Out of memory while trying to allocate
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# RANDOM SEED 세팅
# os.environ["TF_DETERMINISTIC_OPS"] = "True"
# os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "True"

class Trainer(object):
    def __init__(self, config) -> None:
        self.config = config
        self._clear_session()
        self.configure_train_ops()
        print('initialize')

    def _clear_session(self):
        """
            Tensorflow 계산 그래프의 이전 session 및 메모리를 초기화
        """
        tf.keras.backend.clear_session()
        _ = gc.collect()
   
    def configure_train_ops(self) -> None:
        """
            학습 관련 설정
            1. Model
            2. Dataset
            3. Optimizer
            4. Loss
            5. Metric
            6. Logger
        """
        # Params 
        disp_alpha, disp_beta = 10, 0.01
        self.batch_size = self.config['Train']['batch_size']

        # 1. Model
        self.disp_net = DispNet(input_shape=(self.config['Train']['img_h'],
                                             self.config['Train']['img_w'],
                                             3)).model
        self.visual_net = VisualNet()
        self.imu_net = ImuNet()
        self.pose_net = PoseNet(input_size=1024)


        # 2. Dataset
        self.dataset = TspxrTFDSGenerator(data_dir=self.config['Directory']['data_dir'],
                                        image_size=(self.config['Train']['img_h'], self.config['Train']['img_w']),
                                        batch_size=self.config['Train']['batch_size'])
        self.train_dataset = self.dataset.get_trainData(self.dataset.train_data)
        self.test_dataset = self.dataset.get_testData(self.dataset.valid_data)
        
        # 3. Optimizer
        self.warmup_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(self.config['Train']['init_lr'],
                                                                        self.config['Train']['epoch'],
                                                                         self.config['Train']['init_lr'] * 0.1,
                                                                         power=0.9)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['Train']['init_lr'],
                                                  weight_decay=self.config['Train']['weight_decay']
                                                  )# weight_decay=self.config['Train']['weight_decay']

        # 4. Loss
        # self.angle_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
        # self.translation_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)
        
        # 5. Metric
        # self.train_t_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_t_rmse')
        # self.train_r_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_r_rmse')
        # self.valid_t_rmse = tf.keras.metrics.RootMeanSquaredError(name='valid_t_rmse')
        # self.valid_r_rmse = tf.keras.metrics.RootMeanSquaredError(name='valid_r_rmse')

        # 6. Logger
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_path = self.config['Directory']['log_dir'] + '/' + current_time + '_'
        self.train_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/train')
        # self.valid_summary_writer = tf.summary.create_file_writer(tensorboard_path + self.config['Directory']['exp_name'] + '/valid')

        os.makedirs(self.config['Directory']['weights'], exist_ok=True)
        os.makedirs('{0}/{1}'.format(self.config['Directory']['weights'],
                                self.config['Directory']['exp_name']),
                                exist_ok=True)
    
    @tf.function(jit_compile=True)
    def train_step(self, imgs, imus, gts, intrinsics, inv_intrinsics) -> tf.Tensor:
        with tf.GradientTape() as tape:
            alpha1, alpha2, alpha3, alpha4 = 1, 0.1, 0.1, 0.1
            scales = 4
            imgs = tf.unstack(imgs, axis=1)
            
            input_images = tf.concat(imgs[1:4], axis=0)

            disp = self.disp_net(input_images, training=True)
            
            depth = [1/dis for dis in disp]
            depth1 = [d[self.batch_size*0: self.batch_size*1] for d in depth]
            depth2 = [d[self.batch_size*1: self.batch_size*2] for d in depth]
            depth3 = [d[self.batch_size*2: self.batch_size*3] for d in depth]

            visual_feature = self.visual_net(imgs, training=True)
            imu_feature = self.imu_net(imus[:, 1:]) # B, 4, 512
            out = self.pose_net([visual_feature, imu_feature], training=True) # B, 4, 6
            out_w_pose = []
            for j in range(3):
                tmp_out = self.pose_net([visual_feature[:, j:j+2], imu_feature[:, j:j+2]], training=True) # B, 2, 6
                out_w_pose.append(tmp_out)
            out_w_pose_avg = [out_w_pose[0][:, 0], (out_w_pose[0][:, 1]+out_w_pose[1][:, 0])/2,
                                 (out_w_pose[1][:, 1]+out_w_pose[2][:, 0])/2, out_w_pose[2][:, 1]]    # T12 (T23) (T34) T45
            out_w_pose_avg = tf.stack(out_w_pose_avg, axis=1)  # B, 4, 6

            loss_photo, loss_3d = 0, 0

            for j in range(3):
                # test_npy = out_w_pose[j].cpu().detach().numpy()
                # np.save('./test.npy', test_npy)

                pose_j = out2posew_tf(out_w_pose[j]) # (4,2,6) -> (4,2,3,4)
                depth_j = [d[self.batch_size*j: self.batch_size*(j+1)] for d in depth]
                if j != 1:
                    tmp1, tmp2 = photometric_reconstruction_loss(imgs[j+1], imgs[j:j+1]+imgs[j+2:j+3], intrinsics,
                                                        depth_j[:scales], pose_j, 'euler', 'zeros', ref_depth=[None]*scales)
                else:
                    tmp1, tmp2 = photometric_reconstruction_loss(imgs[j+1], imgs[j:j+1]+imgs[j+2:j+3], intrinsics,
                                                        depth_j[:scales], pose_j, 'euler', 'zeros',
                                                        ref_depth=[[depth1[s], depth3[s]] for s in range(scales)])

                loss_photo += tmp1
                loss_3d += tmp2
            pose = out2pose(out, 5)
            loss_vo1 = voloss(out_w_pose[0][:, 1], out_w_pose[1][:, 0]) + voloss(out_w_pose[1][:, 1], out_w_pose[2][:, 0])
            loss_vo2 = voloss(out, out_w_pose_avg)

            loss_smooth = disp_smooth_loss(depth[:scales], input_images)

            total_loss = alpha1*loss_photo + alpha2*loss_vo1 + alpha3*loss_vo2 + alpha4*loss_smooth + alpha2*loss_3d
        # print(f'loss_photo  {loss_photo}, loss_vo1  {loss_vo1}, loss_vo2  {loss_vo2}, loss_smooth  {loss_smooth}, loss_3d  {loss_3d}')
            
            
        # loss update
        model_variables = self.disp_net.trainable_variables + \
                        self.visual_net.trainable_variables + \
                        self.imu_net.trainable_variables + \
                        self.pose_net.trainable_variables
        gradients = tape.gradient(total_loss, model_variables)
        self.optimizer.apply_gradients(zip(gradients, model_variables))

        return total_loss

    # @tf.function(jit_compile=True)
    # def validation_step(self, imgs, gts) -> tf.Tensor:
    #     poses = self.model([imgs], hc=None, training=False)

    #     self.valid_r_rmse.update_state(gts[:, :, :3], poses[:, :, :3])
    #     self.valid_t_rmse.update_state(gts[:, :, 3:], poses[:, :, 3:]) 
        
    #     return poses
    
    # def decode_items(self, imgs, preds, gts, decisions=None):
    #     img = self.dataset.decode_image(imgs[0, :, :, :, :3]).numpy()
    #     final_img = self.dataset.decode_image(imgs[0, :, :, :, 3:]).numpy()
    #     img = tf.concat([img, final_img], axis=0)
    #     pred = preds[0].numpy()
    #     gt = gts[0].numpy()

    #     # plot_buffer = plot_line_tensorboard(imgs=img, pred=pred, gt=gt, decision=decision)
    #     plot_buffer = plot_3d_tensorboard(imgs=img, pred=pred, gt=gt, decision=decisions)

    #     image = tf.image.decode_png(plot_buffer.getvalue(), channels=4)
    #     return tf.expand_dims(image, 0)

    def train(self) -> None:        
        for epoch in range(self.config['Train']['epoch']):    
            lr = self.warmup_scheduler(epoch)

            # Set learning rate
            self.optimizer.learning_rate = lr
            
            train_tqdm = tqdm(self.train_dataset, total=self.dataset.number_train)
            print(' LR : {0}'.format(self.optimizer.learning_rate))
            train_tqdm.set_description('Training   || Epoch : {0} ||'.format(epoch,
                                                                             round(float(self.optimizer.learning_rate.numpy()), 8)))
            for _, (image, imus, gts, intrinsics, inv_intrinsics) in enumerate(train_tqdm):
                epoch_loss = self.train_step(image, imus, gts, intrinsics, inv_intrinsics)
                print(f'Train loss {epoch_loss}')
            with self.train_summary_writer.as_default():
            #     tf.summary.scalar(self.train_t_rmse.name, self.train_t_rmse.result(), step=epoch)
            #     tf.summary.scalar(self.train_r_rmse.name, self.train_r_rmse.result(), step=epoch)
                tf.summary.scalar('epoch_loss', tf.reduce_mean(epoch_loss).numpy(), step=epoch)
            
            # Validation
            # valid_tqdm = tqdm(self.test_dataset, total=self.dataset.number_test)
            # valid_tqdm.set_description('Validation || ')
            # for _, (image, imus, gts, intrinsics, inv_intrinsics) in enumerate(valid_tqdm):
            #     poses = self.validation_step(image, imus, gts, intrinsics, inv_intrinsics)
            
            # plot_image = self.decode_items(imgs=image_seq, preds=poses, gts=rel_pose, decisions=None)

            # with self.valid_summary_writer.as_default():
            #     tf.summary.image('Validation_plot', plot_image, step=epoch+1)
            #     tf.summary.scalar(self.valid_t_rmse.name, self.valid_t_rmse.result(), step=epoch)
            #     tf.summary.scalar(self.valid_r_rmse.name, self.valid_r_rmse.result(), step=epoch)

            if epoch % 5 == 0:
                # self.model.save_weights('./{0}/epoch_{1}_model.h5'.format(self.config['Directory']['weights'], epoch))

                self.model.save_weights('{0}/{1}/epoch_{2}_model.h5'.format(self.config['Directory']['weights'],
                                                                            self.config['Directory']['exp_name'],
                                                                            epoch))

            # Log epoch loss
            
            # print(f'\n \
            #         train_T_RMSE : {self.train_t_rmse.result()}, train_R_RMSE : {self.train_r_rmse.result()}, \n \
            #         valid_T_RMSE : {self.valid_t_rmse.result()}, valid_R_RMSE : {self.valid_r_rmse.result()} \n')

            # clear_session()
            # self.train_t_rmse.reset_states()
            # self.train_r_rmse.reset_states()
            # self.valid_t_rmse.reset_states()
            # self.valid_r_rmse.reset_states()

            self._clear_session()

if __name__ == '__main__':
    # LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.9" python trainer.py
    debug = False

    if debug:
        tf.executing_eagerly()
        tf.config.run_functions_eagerly(not debug)
        tf.config.optimizer.set_jit(False)
    else:
        tf.config.optimizer.set_jit(True)
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    with tf.device('/device:GPU:1'):
        # args = parser.parse_args()

        # Set random seed
        # SEED = 42
        # os.environ['PYTHONHASHSEED'] = str(SEED)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # tf.random.set_seed(SEED)
        # np.random.seed(SEED)

        trainer = Trainer(config=config)

        trainer.train()