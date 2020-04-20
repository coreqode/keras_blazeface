import tensorflow as tf
import tensorflow.keras.backend as K
import logging
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, GlobalAveragePooling2D

def channel_padding(x):
    """
    zero padding in an axis of channel 
    """

    return tf.keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)


def singleBlazeBlock(x, filters=24, kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = tf.keras.layers.BatchNormalization()(x_0) #tf.keras.layers.BatchNormalization

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]

        x_ = tf.keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = tf.keras.layers.Lambda(channel_padding)(x_)

        out = tf.keras.layers.Add()([x_1, x_])
        return tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Add()([x_1, x])
    return tf.keras.layers.Activation("relu")(out)


def doubleBlazeBlock(x, filters_1=24, filters_2=96,
                     kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution, project
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = tf.keras.layers.BatchNormalization()(x_0)

    x_2 = tf.keras.layers.Activation("relu")(x_1)

    # depth-wise separable convolution, expand
    x_3 = tf.keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)

    x_4 = tf.keras.layers.BatchNormalization()(x_3)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]

        x_ = tf.keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = tf.keras.layers.Lambda(channel_padding)(x_)

        out = tf.keras.layers.Add()([x_4, x_])
        return tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Add()([x_4, x])
    return tf.keras.layers.Activation("relu")(out)


def network(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)
    print(inputs.shape)

    x_0 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = tf.keras.layers.BatchNormalization()(x_0)
    x_0 = tf.keras.layers.Activation("relu")(x_0)

    # single BlazeBlock phase
    x_1 = singleBlazeBlock(x_0)
    x_2 = singleBlazeBlock(x_1)
    x_3 = singleBlazeBlock(x_2, strides=2, filters=48)
    x_4 = singleBlazeBlock(x_3, filters=48)
    x_5 = singleBlazeBlock(x_4, filters=48)

    # double BlazeBlock phase

    x_6 = doubleBlazeBlock(x_5, strides=2)
    x_7 = doubleBlazeBlock(x_6)
    x_8 = doubleBlazeBlock(x_7)
    x_9 = doubleBlazeBlock(x_8, strides=2)
    x10 = doubleBlazeBlock(x_9)
    x11 = doubleBlazeBlock(x10)
    x_12 = doubleBlazeBlock(x11, strides=2)
    x13 = doubleBlazeBlock(x_12)
    x14 = doubleBlazeBlock(x13)

    model = tf.keras.models.Model(inputs=inputs, outputs =[x11, x14])
    return model

class BlazeFace():

    def __init__(self, config):
        self.channels = 3
        self.input_shape = (config.input_shape,
                            config.input_shape, self.channels)
        self.feature_extractor = network(self.input_shape)
        print(self.feature_extractor.output)
        self.n_boxes = [2,6]


    def build_model(self):
        conf_layer_1  = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 2,
                                            kernel_size=3,
                                            padding='same', 
                                            activation = 'sigmoid')(self.feature_extractor.output[0])
        reg_layer_1 = tf.keras.layers.Conv2D(filters = self.n_boxes[0] * 4, 
                                            kernel_size = 3,
                                            padding = 'same')(self.feature_extractor.output[0])    
    
        conf_layer_2 = tf.keras.layers.Conv2D(filters = self.n_boxes[1] * 2,
                                            kernel_size = 3,
                                            padding = 'same',
                                            activation = 'sigmoid')(self.feature_extractor.output[1])
                                        
        reg_layer_2 = tf.keras.layers.Conv2D(filters = self.n_boxes[1] * 4,
                                            kernel_size = 3,
                                            padding = 'same')(self.feature_extractor.output[1])

        conf_layer_reshaped_1 = tf.keras.layers.Reshape((14*14*2,2))(conf_layer_1)
        reg_layer_reshaped_1 = tf.keras.layers.Reshape((14*14*2, 4))(reg_layer_1)
        conf_layer_reshaped_2 = tf.keras.layers.Reshape((7*7*6,2))(conf_layer_2)
        reg_layer_reshaped_2 = tf.keras.layers.Reshape((7*7*6, 4))(reg_layer_2)

        conf_layer = tf.keras.layers.concatenate([conf_layer_reshaped_2, conf_layer_reshaped_1], axis = 1)
        reg_layer = tf.keras.layers.concatenate([reg_layer_reshaped_2, reg_layer_reshaped_1], axis = 1)

        output = tf.keras.layers.concatenate([conf_layer, reg_layer], axis = -1)

        model = tf.keras.models.Model(self.feature_extractor.input, output)
        return model

if __name__ == "__main__":
    pass