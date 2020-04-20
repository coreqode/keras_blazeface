from pandas import read_csv
import cv2
import glob
import os
import numpy as np
import logging
import tensorflow as tf
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

def append_zero(arr):
    return np.append([0], arr)

class Uplara():
    def __init__(self, config):
        self.dataset = pd.read_csv(config.dataset_path)
        self.foot_id = self.dataset['foot_id']
        self.image_size = config.input_shape
        self.config = config
        print(len(self.foot_id))

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

def dataloader(config, train_generator = True):
    dataset = Uplara(config)
    indices = np.arange(len(dataset))
    random.shuffle(indices)
    train_indices = indices[:int(0.95 * len(dataset))]
    val_indices = indices[int(0.95 * len(dataset)) : ]
    while True:
        if train_generator:
            batch_idx = np.random.choice(train_indices, size=config.train_batch_size, replace=False)
            batch_img = []
            batch_label = []
            for i in batch_idx:
                image, target = dataset[i]
                batch_img.append(image)
                batch_label.append(target)
            yield np.array(batch_img, dtype=np.float32), [np.array(batch_label, dtype=np.float32)]
        
        else:
            batch_idx = np.random.choice(val_indices, size=config.val_batch_size, replace=False)
            batch_img = []
            batch_label = []
            for i in batch_idx:
                image, target = dataset[i]
                batch_img.append(image)
                batch_label.append(target)
            yield np.array(batch_img, dtype=np.float32), [np.array(batch_label, dtype=np.float32)]
        


if __name__ == "__main__":
    dataloader()
