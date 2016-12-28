import numpy as np
import tensorflow as tf

const_init = tf.constant_initializer
w_init = tf.contrib.layers.variance_scaling_initializer
w_init2 = tf.truncated_normal_initializer
batch_norm = tf.contrib.layers.batch_norm

def conv_op(name, x, outChannel, kw, kh, dw=1, dh=1, 
            padding='SAME',
            wd=0.0001,
            w_initializer=w_init(mode='FAN_OUT'),
            b_initializer=const_init(0.0)):
  
  with tf.variable_scope(name):
    inChannel = x.get_shape().as_list()[3]
    kernel = tf.get_variable(name='W', 
                             shape=[kw, kh, inChannel, outChannel],
                             initializer=w_initializer)

    conv = tf.nn.conv2d(x, kernel, [1, dw, dh, 1], padding=padding)

    if b_initializer==None:
      out = conv
    else:
      bias = tf.get_variable(name='b',
                             shape=[outChannel],
                             initializer=b_initializer)
      out = tf.nn.bias_add(conv, bias)

    if wd is not None:
      weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
      tf.add_to_collection('reg_losses', weight_decay)    

  return out

def mlpconv(name, x, outChannel, kw, kh, dw=1, dh=1, is_training=True):

  with tf.variable_scope(name):
    conv = conv_op('layer1', x, outChannel[0], kw, kh, dw, dh, 
                   b_initializer=None)

    conv = bnrelu('bnrelu_1', conv, is_training)

    conv = conv_op('layer2', conv, outChannel[1], 1, 1, 
                    b_initializer=None)
    
    conv = bnrelu('bnrelu_2',conv, is_training)

    conv = conv_op('layer3', conv, outChannel[2], 1, 1,
                   b_initializer=None)

    conv = bnrelu('bnrelu_3', conv, is_training)

  return conv

def res_op(name, x, outChannel, kw, kh, d, 
           pre_activation=True,
           is_training=True):

  origin_x = x

  with tf.variable_scope(name+'/residual'):

    if pre_activation is True:
      x = bnrelu('bnrelu_1', x, is_training)

    x = conv_op('map1', x, outChannel, kw, kh, d, d,
                b_initializer=None)

    x = bnrelu('bnrelu_2', x, is_training)

    x = conv_op('map2', x, outChannel, kw, kh,
                b_initializer=None)

  with tf.variable_scope(name+'/identity'):
    inChannel = origin_x.get_shape().as_list()[-1]
    if d != 1:
      identity = tf.nn.avg_pool(origin_x, 
                                [1, 1, 1, 1],
                                [1, d, d, 1],
                                padding='VALID') 
    else:
      identity = origin_x

    if outChannel != inChannel:
      identityChannel = identity.get_shape().as_list()[-1]
      xChannel = x.get_shape().as_list()[-1]

      dp = (xChannel - identityChannel)
      #dp0 = np.ceil(dp).astype('int32')
      #dp1 = np.floor(dp).astype('int32')

      identity = tf.pad(identity, [[0, 0], [0, 0], [0, 0], [dp, 0]])

    out = tf.add(x, identity)

  return out
 
def prelu(x):

  with tf.variable_scope('prelu'):
    a = tf.get_variable(name='P',
                        shape=[],
                        initializer=const_init(0.25))
    out = tf.maximum(x, 0.0) + tf.scalar_mul(a, tf.minimum(0.0, x))

  return out

def bnrelu(name, x, is_training=True, center=True, scale=True):

  with tf.variable_scope(name):
    x = batch_norm(x,
                   decay=0.9,
                   center=True,
                   scale=True,
                   is_training=is_training)

    x = tf.nn.relu(x)

  return x

def dense_op(name, x, n_unit, k, kw, kd, is_training):
  
  with tf.variable_scope(name):
    for i in range(0, n_unit):
      _name = 'unit_%i' % i
      out = _dense_basic_unit(_name, x, k, kw, kd, is_training)
      x = tf.concat(3, [out, x])

  n_map = x.get_shape().as_list()[-1]

  return out, n_map

def transition_op(name, x, k, is_training):
  '''
  x: input
  k: numbers of feature-map
  '''
  with tf.variable_scope(name):
    x = bnrelu('BN-Relu', x, is_training)
    x = conv_op('conv', x, k, 1, 1, b_initializer=None, padding='VALID')
    x = tf.nn.avg_pool(x, 
                       ksize=[1, 2, 2, 1], 
                       strides=[1, 2, 2, 1], 
                       padding='VALID')
  return x

def _dense_basic_unit(name, x, k, kw, kd, is_training):

  with tf.variable_scope(name):
    out = bnrelu('BN-Relu_0', x, is_training)
    out = conv_op('dim_reduction', out, 4*k, 1, 1, b_initializer=None)
    out = bnrelu('BN-Relu_1', out, is_training)
    out = conv_op('conv', out, k, kw, kd, b_initializer=None)

  return out