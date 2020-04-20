import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'image': tf.io.FixedLenFeature([], tf.string),
                        'l_prob':  tf.io.FixedLenFeature([], tf.float32),
                        'r_prob':  tf.io.FixedLenFeature([], tf.float32),
                        'l_center_x':  tf.io.FixedLenFeature([], tf.float32),
                        'l_center_y':  tf.io.FixedLenFeature([], tf.float32),
                        'l_width':  tf.io.FixedLenFeature([], tf.float32),
                        'l_height':  tf.io.FixedLenFeature([], tf.float32),
                        'r_center_x':  tf.io.FixedLenFeature([], tf.float32),
                        'r_center_y':  tf.io.FixedLenFeature([], tf.float32),
                        'r_width':  tf.io.FixedLenFeature([], tf.float32),
                        'r_height':  tf.io.FixedLenFeature([], tf.float32)}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    all_features = [parsed_features['l_prob'],parsed_features['r_prob'],parsed_features['l_center_x'],parsed_features['l_center_y']
                    ,parsed_features['l_width'],parsed_features['l_height'],parsed_features['r_center_x'],parsed_features['r_center_y'],
                    parsed_features['r_width'],parsed_features['r_height']]
    # Turn your saved image string into an array
    image = tf.io.decode_raw(
        parsed_features['image'], tf.uint8)

    image = tf.cast(tf.reshape(image, (224, 224, 3)), tf.float32)
    image = (image / 127.5) - 1 
    
    return image, all_features

def create_dataset(batch_size, filepath, shuffle_buffer):
    dataset = tf.data.TFRecordDataset(filepath)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # iterator = iter(dataset)
    # image, labels= iterator.get_next()
    return dataset

if __name__ == "__main__":
    dataset = create_dataset("/home/noldsoul/Desktop/Uplara/keras_object_detection/src/object_detection/utils/trainset.record")
    iterator = iter(dataset)
    for i in range(50):
        image, labels= iterator.get_next()
        image = tf.reshape(image, [-1, 224, 224, 3])
        print(image.shape)
        image = np.squeeze(image, axis = 0)
        print(labels.shape)
        plt.imshow(image)
        plt.show()

    # for image, label in dataset.take(1):
    #     image = image[0]
    #     label = label[0]
    #     image = np.array(image)
    #     lxmin = int(label[2] - label[4] / 2)
    #     lxmax = int(label[2] + label[4] / 2)
    #     lymin = int(label[3] - label[5] / 2)
    #     lymax = int(label[3] + label[5] / 2)
    #     cv2.rectangle(image, (lxmin, lymin), (lxmax, lymax), (0,255,0), 2)
    #     rxmin = int(label[6] - label[8] / 2)
    #     rxmax = int(label[6] + label[8] / 2)
    #     rymin = int(label[7] - label[9] / 2)
    #     rymax = int(label[7] + label[9] / 2)
    #     cv2.rectangle(image, (rxmin, rymin), (rxmax, rymax), (0,255,0), 2)
    #     cv2.imshow('image', image)
    #     if cv2.waitKey() & 0xFF == ord('q'):
    #             break
    #             # plt.show()