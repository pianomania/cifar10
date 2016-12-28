import numpy as np
import tensorflow as tf
from utils import conv_op, dense_op, transition_op

class DenseNet(object):

  def __init__(self, k):
  
    self.k = k
    
    self.is_training=tf.placeholder(dtype=tf.bool, shape=[])
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.target = tf.placeholder(dtype=tf.int64, shape=[None])
    
    x = conv_op('conv1', self.image, 12, 3, 3)
    x, n_map = dense_op('dense1', x, 12, k, 3, 3, self.is_training)
    x = transition_op('transition1', x, n_map, self.is_training)
    x, n_map = dense_op('dense2', x, 12, k, 3, 3, self.is_training)
    x = transition_op('transition2', x, n_map, self.is_training)
    x, n_map = dense_op('dense3', x, 12, k, 3, 3, self.is_training)

    with tf.variable_scope('avg_pool'):
      #x = bnrelu('last_bnrelu', x, is_training=self.is_training)

      x_shape = x.get_shape().as_list()
      avg_pool = tf.nn.avg_pool(x, 
                                ksize=[1, x_shape[1], x_shape[2], 1], 
                                strides=[1]*4, 
                                padding='VALID')

    logits = conv_op('logits', avg_pool, 10, 1, 1, padding='VALID')

    self.logits = tf.squeeze(logits)
