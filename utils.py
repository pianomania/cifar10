import numpy as np
import tensorflow as tf


const_init = tf.constant_initializer(0.001)
xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
trunc_gau_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.05)


def conv_op(x, kernel_shape, stride, 
	 		use_relu = True,
			use_batch_norm = True,
			padding = 'SAME',
			wd = 0.01,
			is_training=True):

	''' convolution layer and ReLU activation'''

	kernel = tf.get_variable(name = 'W', 
							 shape = kernel_shape, # e.x. [8, 8, 4, 16], 
							 initializer = trunc_gau_init
							)

	conv = tf.nn.conv2d(x, kernel, stride, padding=padding)
	bias = tf.get_variable(name='b', shape=[kernel_shape[-1]], initializer=const_init)
	out = tf.nn.bias_add(conv, bias)

	if use_batch_norm is True:
		out = tf.contrib.layers.batch_norm(out, is_training=is_training)

	if use_relu is True:
		out = tf.nn.relu(out)

	if wd is not None:
	    weight_decay = tf.mul(tf.nn.l2_loss(kernel), wd, name='weight_loss')
	    tf.add_to_collection('reg_losses', weight_decay)		

	return out

def mlpconv(name, x, filter_shape, stride, layers, is_training):

	with tf.variable_scope(name+'/layer1'):
		conv = conv_op(x, filter_shape, stride, is_training=is_training)

	with tf.variable_scope(name+'/layer2'):
		conv = conv_op(conv, [1 ,1, filter_shape[-1], layers[0]], [1, 1, 1, 1], is_training=is_training)

	with tf.variable_scope(name+'layer3'):
		conv = conv_op(conv, [1, 1, layers[0], layers[1]], [1, 1, 1, 1], is_training=is_training)

	return conv

def image_preprocessing(x):

	x -= np.mean(x, axis=0)
	x /= np.std(x, axis=0)
	
	return x