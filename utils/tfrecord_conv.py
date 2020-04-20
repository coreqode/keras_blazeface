import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from pandas import read_csv
import cv2
import glob
import os
import numpy as np
import logging
import pandas as pd 
from PIL import Image
import random

class Uplara():
    def __init__(self, config):
        self.dataset = pd.read_csv(config.dataset_path)
        self.foot_id = self.dataset['foot_id']
        self.image_size = config.input_shape
        self.config = config
#        print(len(self.foot_id))

    def __getitem__(self, idx):
        image_path = self.config.image_dir + str(self.foot_id[idx]) +"_" + str(self.dataset['angle'][idx]) +".jpg" 
        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        image = image.convert('RGB')
        image = np.array(image)

        ###################For Left Foot#######################
        l_xmin = np.min(self.dataset.loc[idx][::2][1:26])
        l_ymin = np.min(self.dataset.loc[idx][1:][::2][1:26])
        l_xmax = np.max(self.dataset.loc[idx][::2][1:26])
        l_ymax = np.max(self.dataset.loc[idx][1:][::2][1:26])
        #transformation of coordinates
        l_prob = self.dataset.loc[idx][-2]
        ###################For Right Foot #######################
        r_xmin = np.min(self.dataset.loc[idx][52:][::2][0:25])
        r_ymin = np.min(self.dataset.loc[idx][51:][::2][1:26])
        r_xmax = np.max(self.dataset.loc[idx][52:][::2][0:25])
        r_ymax = np.max(self.dataset.loc[idx][51:][::2][1:26])
        # cv2.circle(image, (int(r_xmin), int(r_ymin)),2,  (0,255,0), 2)
        # cv2.circle(image, (int(r_xmax), int(r_ymax)),2,  (0,255,0), 2)
        # plt.imshow(image)
        # plt.show()
        #transformation of coordinates
        r_prob =self.dataset.loc[idx][-1]

        l_box = [l_xmin, l_ymin, l_xmax, l_ymax]
        r_box = [r_xmin, r_ymin, r_xmax, r_ymax]
        boxes = [l_box, r_box]
        labels = [l_prob, r_prob]
        ###################### Visualize the bounding boxes ##############################
        # for visualizing the bounding boxes use, augmentation.py file
        target = self.encoder(boxes, labels)  # To get the encoding of the target
        return image, target

    def encoder(self, boxes, labels):
        target = np.zeros(( 10))
        # for left box
        if not (labels[0] == 0):
            xmin, ymin, xmax, ymax = boxes[0]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            label = labels[0]
            target[0] = label
            target[2:6] = center_x, center_y, width, height
        # for right box
        if not (labels[1] == 0):
            xmin, ymin, xmax, ymax = boxes[1]
            width = xmax - xmin
            height = ymax - ymin
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            label = labels[1]
            target[1] = label
            target[6:10] = center_x, center_y, width, height
        return target

    def __len__(self):
        return len(self.foot_id)


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # hyperparameters
    args.add_argument('--input_shape', type=int, default=224)
    args.add_argument('--train_batch_size', type=int, default=128)
    args.add_argument('--val_batch_size', type=int, default=128)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./checkpoint/best_model.ckpt")
    args.add_argument('--dataset_path', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_dataset.csv")
    args.add_argument('--image_dir', type=str, default="/home/noldsoul/Desktop/Uplara/MobileNet_ObjectDetection/src/phase1/utils/augmented_images/")

    config = args.parse_args()
    dataset = Uplara(config)


    TRAIN_FILEPATH =  "./trainset_300k.record"
    VAL_FILEPATH =  "./valset_300k.record"
    trainwriter = tf.io.TFRecordWriter(TRAIN_FILEPATH)
    valwriter = tf.io.TFRecordWriter(VAL_FILEPATH)
    # Define the features of your tfrecord
    indices = np.arange(len(dataset))
    random.shuffle(indices)
    train_indices = indices[:int(0.95 * len(dataset))]
    val_indices = indices[int(0.95 * len(dataset)) : ]

    # for trainset
    for i in train_indices:
        print("processing :", i)
        image, label = dataset[i]
        feature = {'image':  _bytes_feature(tf.compat.as_bytes(image.tostring())),
                'l_prob':  _float32_feature(float(label[0])),
                'r_prob':  _float32_feature(float(label[1])),
                'l_center_x':  _float32_feature(float(label[2])),
                'l_center_y':  _float32_feature(float(label[3])),
                'l_width':  _float32_feature(float(label[4])),
                'l_height':  _float32_feature(float(label[5])),
                'r_center_x':  _float32_feature(float(label[6])),
                'r_center_y':  _float32_feature(float(label[7])),
                'r_width':  _float32_feature(float(label[8])),
                'r_height':  _float32_feature(float(label[9])),
                }


        # Serialize to string and write to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        trainwriter.write(example.SerializeToString())
    print("trainset completed")
    
    #for valset
    for i in val_indices:
        print("processing :", i)
        image, label = dataset[i]
        feature = {'image':  _bytes_feature(tf.compat.as_bytes(image.tostring())),
                'l_prob':  _float32_feature(float(label[0])),
                'r_prob':  _float32_feature(float(label[1])),
                'l_center_x':  _float32_feature(float(label[2])),
                'l_center_y':  _float32_feature(float(label[3])),
                'l_width':  _float32_feature(float(label[4])),
                'l_height':  _float32_feature(float(label[5])),
                'r_center_x':  _float32_feature(float(label[6])),
                'r_center_y':  _float32_feature(float(label[7])),
                'r_width':  _float32_feature(float(label[8])),
                'r_height':  _float32_feature(float(label[9])),
                }


        # Serialize to string and write to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        valwriter.write(example.SerializeToString())
    print("valset completed")
