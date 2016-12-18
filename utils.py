import numpy as np
import tensorflow as tf

const_init = tf.constant_initializer
w_init = tf.contrib.layers.variance_scaling_initializer
batch_norm = tf.contrib.layers.batch_norm

def conv_op(name, x, kernel_shape, stride,
            padding='SAME',
            wd=0.0001,
            w_initializer=w_init(mode='FAN_OUT'),
            b_initializer=const_init(0.0)):

  with tf.variable_scope(name):
    kernel = tf.get_variable(name='W', 
                             shape=kernel_shape, # e.x. [8, 8, 4, 16], 
                             initializer=w_initializer)

    conv = tf.nn.conv2d(x, kernel, stride, padding=padding)

    if b_initializer==None:
      out = conv
    else:
      bias = tf.get_variable(name='b', shape=[kernel_shape[-1]], initializer=b_initializer)
      out = tf.nn.bias_add(conv, bias)

    if wd is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
      tf.add_to_collection('reg_losses', weight_decay)    

    return out

def mlpconv(name, x, kernel_shape, stride, layers, is_training):

  with tf.variable_scope(name):
    conv = conv_op('layer1', x, kernel_shape, stride, 
                   b_initializer=None)

    conv = bnrelu('bnrelu_1', conv, is_training)

    conv = conv_op('layer2', conv,
                   [1 ,1, kernel_shape[-1], layers[0]],
                   [1, 1, 1, 1],
                   b_initializer=None)

    conv = bnrelu('bnrelu_1',conv, is_training)

    conv = conv_op('layer3', conv, 
                   [1, 1, layers[0],
                   layers[1]], [1, 1, 1, 1],
                   b_initializer=None)

    conv = bnrelu('bnrelu', conv, is_training)

  return conv

def res_op(name, x, kernel_shape, stride, 
           pre_activation=True,
           is_training=True):

  origin_x = x

  with tf.variable_scope(name):
    if pre_activation is True:
      x = bnrelu('bnrelu_1', x, is_training)

  with tf.variable_scope(name+'/residual'):
    kw, kh, ki, ko = tuple(kernel_shape)

    x = conv_op('map1', x, kernel_shape, [1, stride, stride, 1], 
                wd=0.00001,
                b_initializer=None)
    x = bnrelu('bnrelu_2', x, is_training)
    x = conv_op('map2', x, [kw, kh, ko, ko], [1]*4,
                wd=0.00001,
                b_initializer=None)

  with tf.variable_scope(name+'/identity'):
    origin_x_shape = origin_x.get_shape().as_list()
    if stride != 1:
      identity = tf.nn.avg_pool(origin_x, 
                                [1, 1, 1, 1],
                                [1, stride, stride, 1],
                                padding='VALID') 
    else:
      identity = origin_x

    if kernel_shape[-1] != origin_x_shape[-1]:
      identity_depth = identity.get_shape().as_list()[-1]
      x_depth = x.get_shape().as_list()[-1]

      dp = (x_depth - identity_depth)/2.0
      dp0 = np.ceil(dp).astype('int32')
      dp1 = np.floor(dp).astype('int32')

      identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [dp0, dp1]])

    out = tf.add(x, identity)

    return out
 
def prelu(x):

  with tf.variable_scope('prelu'):
    a = tf.get_variable(name='P',
                        shape=[],
                        initializer=const_init(0.25))
    out = tf.maximum(x, 0.0) + tf.scalar_mul(a, tf.minimum(0.0, x))

  return out

def bnrelu(name, x, is_training=True, scale=True):

  with tf.variable_scope(name):
    x = batch_norm(x, scale=True, is_training=is_training)
    x = tf.nn.relu(x)

  return x