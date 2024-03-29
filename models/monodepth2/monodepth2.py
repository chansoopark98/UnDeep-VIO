import tensorflow as tf
import torch
from .layers import *

class DispNet():
    def __init__(self, input_shape) -> None:
        self.input_shape = input_shape
        self.model = self.build_model()
        self.model = self.load_weight_torch(model=self.model)

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        encoder = []
        outputs = []

        # Encoder part:
        # x = (inputs - 0.45) / 0.225
        x = inputs
        x = tf.keras.layers.ZeroPadding2D(3)(x)
        x = tf.keras.layers.Conv2D(64, 7, strides=2, activation='linear', use_bias=False, name='conv1')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
        x = tf.keras.layers.ReLU()(x)
        encoder.append(x)
        x = tf.keras.layers.ZeroPadding2D(1)(x)
        # x = tf.keras.layers.MaxPooling2D(3, 2)(x)
        x = tf.nn.max_pool2d(x, 3, 2, padding='VALID')
        

        for i in range(1, 5):
            x = res_block(x, (i, 0), i > 1)
            x = res_block(x, (i, 1))
            encoder.append(x)

        # Decoder part:
        x = up_conv(256, encoder[4], encoder[3])
        x = up_conv(128, x, encoder[2])
        outputs.append(conv_block(128, x, disp=True))

        x = up_conv(64, x, encoder[1])
        outputs.append(conv_block(64, x, disp=True))

        x = up_conv(32, x, encoder[0])
        outputs.append(conv_block(32, x, disp=True))

        x = up_conv(16, x)
        outputs.append(conv_block(16, x, disp=True))

        outputs = outputs[::-1]
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='depth')
        return model

    def load_weight_torch(self, model: tf.keras.Model):
        # Loading the weights from the PyTorch files.
        # The weights for the mono_640x192 model are can be obtained from the original PyTorch repo at
        # https://github.com/nianticlabs/monodepth2

        encoder_path = './models/monodepth2/encoder.pth'
        decoder_path = './models/monodepth2/depth.pth'
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        loaded_dict = torch.load(decoder_path, map_location='cpu')

        model.get_layer('conv1').set_weights(
            [loaded_dict_enc['encoder.conv1.weight'].numpy().transpose(2, 3, 1, 0)])

        model.get_layer('bn1').set_weights(
            [loaded_dict_enc['encoder.bn1.weight'].numpy(),
            loaded_dict_enc['encoder.bn1.bias'].numpy(),
            loaded_dict_enc['encoder.bn1.running_mean'].numpy(),
            loaded_dict_enc['encoder.bn1.running_var'].numpy()])

        for layer in model.layers:
            name = layer.name.split('.')
            if name[0] == 'en':
                name = '.'.join(name[1:])
                num_weights = len(layer.get_weights())
                if num_weights == 1:
                    layer.set_weights([loaded_dict_enc['encoder.' + name + '.weight']
                                    .numpy().transpose(2, 3, 1, 0)])
                else:
                    layer.set_weights(
                        [loaded_dict_enc['encoder.' + name + '.weight'].numpy(),
                        loaded_dict_enc['encoder.' + name + '.bias'].numpy(),
                        loaded_dict_enc['encoder.' + name + '.running_mean'].numpy(),
                        loaded_dict_enc['encoder.' + name + '.running_var'].numpy()])

            if name[0] == 'de':
                if name[1] == 'upconv':
                    num = str(2 * (4 - int(name[2])) + int(name[3]))
                    layer.set_weights([loaded_dict['decoder.' + num + '.conv.conv.weight']
                                    .numpy().transpose(2, 3, 1, 0),
                                    loaded_dict['decoder.' + num + '.conv.conv.bias'].numpy()])
                else:
                    num = str(int(name[2]) + 10)
                    layer.set_weights([loaded_dict['decoder.' + num + '.conv.weight']
                                    .numpy().transpose(2, 3, 1, 0),
                                    loaded_dict['decoder.' + num + '.conv.bias'].numpy()])
        
        return model


# Define the model using Keras Functional API:



if __name__ == '__main__':
    # set input size
    input_shape = (256, 512, 3)
    # load model
    disp = DispNet(input_shape=input_shape)

    # prepare data
    data = tf.ones((16, *input_shape))

    # execute model
    results = disp.model(data)

    # get results
    for result in results:
        print(result.shape)