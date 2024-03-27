import tensorflow as tf

KERNEL_INIT = 'glorot_uniform'

def conv(in_planes, out_planes, kernel_size=3):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(out_planes, kernel_size=kernel_size, padding='same', strides=2, 
                      kernel_initializer=KERNEL_INIT, use_bias=True),
        tf.keras.layers.ReLU()
    ])

class VisualNet(tf.keras.Model):
    '''
    Encode imgs into visual features
    '''
    def __init__(self):
        super(VisualNet, self).__init__()
        
        conv_planes = [16, 32, 64, 128, 256, 512, 512]
        self.conv1 = conv(6, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        self.conv8 = tf.keras.layers.Conv2D(conv_planes[6], kernel_size=1, padding='same', 
                                   kernel_initializer=KERNEL_INIT, use_bias=True)

    def call(self, imgs):
        vis_feat = []
        for i in range(len(imgs) - 1):
            input = tf.concat(imgs[i:i + 2], -1)
            out_conv1 = self.conv1(input)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)
            out_conv5 = self.conv5(out_conv4)
            out_conv6 = self.conv6(out_conv5)
            out_conv7 = self.conv7(out_conv6)
            out_conv8 = self.conv8(out_conv7)

            # mean 연산에서 차원 변경 필요 (PyTorch와 TensorFlow 차원 다름)
            vis_feat.append(tf.reduce_mean(out_conv8, axis=[1, 2]))
        return tf.stack(vis_feat, axis=1)
    
class ImuNet(tf.keras.Model):
    def __init__(self, input_size=6, hidden_size=512):
        super(ImuNet, self).__init__()
        # Define LSTM cells
        self.units = hidden_size
        self.rnn_dropout = 0.
        
        cells = [tf.keras.layers.LSTMCell(self.units,
                                          dropout=self.rnn_dropout,
                                          name='ImuNet_lstm_1'),
                tf.keras.layers.LSTMCell(self.units,
                                          dropout=self.rnn_dropout,
                                          name='ImuNet_lstm_2')
                                          ]
        stacked_lstm_cells = tf.keras.layers.StackedRNNCells(cells, name='ImuNet_stacked_lstm')
        
        self.rnn = tf.keras.layers.RNN(stacked_lstm_cells, return_sequences=True, return_state=True,name='ImuNet_lstm')

    # Multi LSTM
    def call(self, imus):
        x = imus
        B, t, N, _ = x.shape  # B, T, N, 6
        x = tf.reshape(x, (B * t, N, -1))  # Reshape to (B*T, N, 6)
        out, _, _ = self.rnn(x)
        out = out[:, -1, :]
        return tf.reshape(out, (B, t, -1))

class FuseModule(tf.keras.Model):
    def __init__(self, channels, reduction):
        super(FuseModule, self).__init__()
        self.fc1 = tf.keras.layers.Dense(channels // reduction)
        self.relu = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Dense(channels)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class PoseNet(tf.keras.Model):
    def __init__(self, input_size=1024):
        super(PoseNet, self).__init__()

        self.se = FuseModule(input_size, 16)

        cells = [tf.keras.layers.LSTMCell(input_size,
                                          dropout=0,
                                          name='PoseNet_lstm_1')]
        stacked_lstm_cells = tf.keras.layers.StackedRNNCells(cells, name='PoseNet_stacked_lstm')
        
        self.rnn = tf.keras.layers.RNN(stacked_lstm_cells, return_sequences=True, return_state=True,name='PoseNet_lstm')

        self.fc1 = tf.keras.layers.Dense(6)

    def call(self, inputs):
        visual_fea, imu_fea = inputs
        if imu_fea is not None:
            B, t, _ = imu_fea.shape
            imu_input = tf.reshape(imu_fea, (B, t, -1))
            visual_input = tf.reshape(visual_fea, (B, t, -1))
            inpt = tf.concat([visual_input, imu_input], axis=-1)
        else:
            inpt = visual_fea
        
        inpt = self.se(inpt)
        out, _ = self.rnn(inpt)
        out = 0.01 * self.fc1(out)
        return out
    
if __name__ == '__main__':
    image_model = VisualNet()
    imgs = [tf.ones((4, 256, 832, 3))] * 5
    img_fea = image_model(imgs)
    print(img_fea.shape)

    imu_model = ImuNet()
    imus = tf.ones((4, 4, 11, 6))
    imu_fea = imu_model(imus)
    print(imu_fea.shape)

    pose_modle = PoseNet()
    pose = pose_modle(img_fea, imu_fea)
    print(pose.shape)
