import numpy as np
import tensorflow as tf
from utils import res_op, conv_op, bnrelu

class ResNet(object):

  def __init__(self, nstack=3, is_training=True):
  
    self.nstack = nstack
    
    self.is_training=tf.placeholder(dtype=tf.bool, shape=[])
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.target = tf.placeholder(dtype=tf.int64, shape=[None])
    
    res = conv_op('conv1', self.image, 16, 3, 3)
    res = bnrelu('bnrelu', res, self.is_training)

    res = res_op('res1_0', res, 16, 3, 3, 1, pre_activation=False, is_training=self.is_training)
    for i in xrange(1, self.nstack):
      res = res_op('res1_%d' % i, res, 16, 3, 3, 1, is_training=self.is_training)

    for i in xrange(0, self.nstack):
      res = res_op('res2_%d' % i, res, 32, 3, 3, i==0 and 2 or 1, is_training=self.is_training)

    for i in xrange(0, self.nstack):
      res = res_op('res3_%d' % i, res, 64, 3, 3, 1, i==0 and 2 or 1, is_training=self.is_training)

    with tf.variable_scope('avg_pool'):
      res = bnrelu('last_bnrelu', res, is_training=self.is_training)

      res_shape = res.get_shape().as_list()
      avg_pool = tf.nn.avg_pool(res, 
                                ksize=[1, res_shape[1], res_shape[2], 1], 
                                strides=[1]*4, 
                                padding='VALID')

    logits = conv_op('logits', avg_pool, 10, 1, 1, padding='VALID')

    self.logits = tf.squeeze(logits)
