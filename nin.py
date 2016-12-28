import numpy as np
import tensorflow as tf
from utils import conv_op, mlpconv


class NIN(object):

  def __init__(self):

    # define placeholder in train or eval usage
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.target = tf.placeholder(dtype=tf.int64, shape=[None])
    self.keep_probs = tf.placeholder(dtype=tf.float32, shape=[])
    self.is_training=tf.placeholder(dtype=tf.bool, shape=[])

    # define main architecture
    y = mlpconv('mlpconv_layer1', self.image, [192, 160 ,92], 5, 5, 2, 2, is_training=self.is_training)
    y = tf.nn.dropout(y, self.keep_probs)
    y = mlpconv('mlpconv_layer2', y, [192, 156, 128], 5, 5, 2, 2, is_training=self.is_training)
    y = tf.nn.dropout(y, self.keep_probs)
    y = mlpconv('mlpconv_output', y, [192, 156, 10], 3, 3, is_training=self.is_training)
    y = tf.nn.dropout(y, self.keep_probs)
    self.y = y

    y_shape = y.get_shape().as_list()
    
    # global averaging
    avg_pool = tf.nn.avg_pool(y,
                              ksize=[1, y_shape[1], y_shape[2], 1],
                              strides=[1]*4, 
                              padding='VALID')
    # Version 1
    self.logits = tf.squeeze(avg_pool)
