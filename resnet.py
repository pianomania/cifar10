import numpy as np
import tensorflow as tf
from utils import res_op, conv_op, bnrelu

class ResNet(object):

  def __init__(self, sess, nstack=3, is_training=True):
  
    self.nstack = nstack

    # define placeholder in train or eval usage
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.target = tf.placeholder(dtype=tf.int64, shape=[None])
    
    res = conv_op('conv1', self.image, [3, 3, 3, 16], [1]*4,
                  wd=0.00001)
    res = bnrelu('bnrelu', res, is_training)

    res = res_op('res1_0', res, [3, 3, 16 ,16], 1, pre_activation=False)
    for i in xrange(1, self.nstack):
      res = res_op('res1_%d' % i, res, [3, 3, 16, 16], 1)

    for i in xrange(0, self.nstack):
      res = res_op('res2_%d' % i, res, [3, 3, i==0 and 16 or 32, 32], i==0 and 2 or 1)

    for i in xrange(0, self.nstack):
      res = res_op('res3_%d' % i, res, [3, 3, i==0 and 32 or 64, 64], i==0 and 2 or 1)

    with tf.variable_scope('avg_pool'):
      res = bnrelu('last_bnrelu', res, is_training=is_training)

      res_shape = res.get_shape().as_list()
      avg_pool = tf.nn.avg_pool(res, 
                                ksize=[1, res_shape[1], res_shape[2], 1], 
                                strides=[1]*4, 
                                padding='VALID')

    class_logits = conv_op('output', avg_pool, 
                  [1, 1, 64, 10],
                  stride = [1, 1, 1, 1],
                  wd = 0.00001,
                  padding='VALID')

    self.class_logits = tf.squeeze(class_logits)
