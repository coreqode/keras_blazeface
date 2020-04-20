import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten, GlobalAveragePooling2D

class MobileNetV2():
    def __init__(self, config):
        self.input_shape = (config.input_shape, config.input_shape, 3)

    def build_model(self):
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights=None)
        output = GlobalAveragePooling2D()(base_model.output)
#         output = Conv2D(64, 3, padding = 'same')(base_model.output)
#         output = BatchNormalization()(output)
#         output = Activation("relu")(output)
        # output = Conv2D(256, 3, padding = 'same')(output)
        # output = BatchNormalization()(output)
        # output = Activation("relu")(output)
        box = Dense(10,name="box")(output)
        box_prob = box[:,:2]
        box_coords = box[:,2:]
        box_coords = tf.keras.layers.Lambda(lambda box_coords: tf.keras.activations.relu(box_coords,max_value=224))(box_coords)
        box_prob = tf.keras.layers.Lambda(lambda box_prob: tf.keras.activations.sigmoid(box_prob))(box_prob)
        


#         loc_layer_dense = Dense(10, activation = 'relu')(output)
#         conf_layer_dense = Dense(2, activation = 'sigmoid')(output)
        out= tf.keras.layers.concatenate(inputs = [box_prob,box_coords],axis = 1)
#         output = tf.keras.layers.concatenate([conf_layer_dense, loc_layer_dense], axis = -1)
        model = tf.keras.models.Model(base_model.input, out)
        return model



