import numpy as np
import tensorflow as tf

const_init = tf.constant_initializer(0.0)
xavier_init_conv2d = tf.contrib.layers.xavier_initializer_conv2d()

def conv_op(x, kernel_shape, stride, use_relu=True):
	''' convolution layer and ReLU activation'''

	kernel = tf.get_variable(name='W', 
							 shape=kernel_shape, # e.x. [8, 8, 4, 16], 
							 initializer=xavier_init_conv2d)

	conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
	bias = tf.get_variable(name='b', shape=[kernel_shape[-1]], initializer=const_init)
	out = tf.nn.bias_add(conv, bias)

	if use_relu is True:
		out = tf.nn.relu(out)

	return out


def mlpconv(name, x, filter_shape, stride, layers):

	with tf.variable_scope(name+'/layer1'):
		conv = conv_op(x, filter_shape, stride)

	with tf.variable_scope(name+'/layer2'):
		conv = conv_op(conv, [1 ,1, filter_shape[-1], layers[0]], [1, 1, 1, 1])

	with tf.variable_scope(name+'layer3'):
		conv = conv_op(conv, [1, 1, layers[0], layers[1]], [1, 1, 1, 1])

	return conv


class NIN(object):

	def __init__(self, sess, dataset, max_train_step, batch_size):

		self.sess = sess
		self.data = dataset['data'][0:49000].astype('float32').reshape(-1, 32, 32, 3)
		self.labels = dataset['labels'][0:49000]
		self.val_data = dataset['data'][49000:].astype('float32').reshape(-1, 32, 32, 3)
		self.val_labels = dataset['labels'][49000:]

		self.max_train_step = max_train_step
		self.batch_size = batch_size


		self.global_step = tf.Variable(0, trainable=False, name='global_step')
		self.lr = tf.train.exponential_decay(1e-3, self.global_step, 1*10**3, 0.97, staircase=True)

		# define nin
		self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
		self.targets = tf.placeholder(dtype=tf.int64, shape=[None])

		y = mlpconv('mlpconv_layer1', self.image, [7, 7, 3, 64], [1, 2, 2, 1], [48, 32])
		y = mlpconv('mlpconv_layer2', y, [3, 3, 32, 96], [1, 2, 2, 1], [64, 48])
		y = mlpconv('mlpconv_layer3', y, [3, 3, 48, 128], [1, 1, 1, 1], [64, 48]) #testing
		y = mlpconv('mlpconv_layer4', y, [3, 3, 48, 128], [1, 1, 1, 1], [64, 10])

		self.y = y

		y_shape = y.get_shape().as_list()
		avg_pool = tf.nn.avg_pool(y, ksize=[1, y_shape[1], y_shape[2], 1], strides=[1]*4, padding='VALID')

		self.avg_pool = tf.squeeze(avg_pool)

		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.avg_pool, self.targets)

		self.loss = tf.reduce_mean(cross_entropy) 

		self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)


		# define accuracy
		correct_prediction = tf.equal(tf.argmax(self.avg_pool, 1), self.targets)
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	def _whitening(self, x):

		x -= np.mean(x, axis=0)
		x /= np.std(x, axis=0)

		return x

	def _fancy_pca(self, x):
		
		r, b , g = x[...,0].flatten(), x[...,1].flatten(), x[...,2].flatten()
		xx = self._whitening(np.vstack([r, b, g]).T)

		covar = xx.T.dot(xx)
		u, s, v = np.linalg.svd(covar)

		scale = s.reshape(-1, 1)*np.random.randn(3, x.shape[0])*0.1
		noise = u.dot(scale).T

		x += noise[:, np.newaxis, np.newaxis, :]

		return x

	def do_preprocess(self):

		self.data = self._whitening(self.data)
		self.val_data = self._whitening(self.val_data)

	def summary(self):

		tf.scalar_summary('loss', self.loss)
		tf.scalar_summary('accuracy', self.accuracy)


	def _batch(self):

		idx = np.random.randint(0, self.data.shape[0], self.batch_size)
		x = self.data[idx]
		y = self.labels[idx]

		# random intensity
		#x = self._fancy_pca(x)
		
		# random fliping image
		flip_idx = np.random.permutation(x.shape[0])[:x.shape[0]/2]
		x[flip_idx] = x[flip_idx,:,:,::-1]

		# contrastness and brightness
		gain = 0.2 + np.random.rand(x.shape[0], 1, 1, 1)*1.6
		#bias = np.random.randint(-20, 21, (bc_idx.shape[0], 1, 1, 1))
		x = x*gain #+ bias
		x = np.clip(x, 0, 255)
		
		# whitening
		x = self._whitening(x)

		return x, y

	def train(self):

		self.do_preprocess()
		self.summary()

		#saver = tf.train.Saver()
		summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter('/home/yao/cifar10/tmp/train_summary', self.sess.graph)

		for i in range(self.max_train_step):

			x, y = self._batch()
			self.sess.run(self.optimize, feed_dict={self.image: x, self.targets: y})

			if i % 20 == 0 :

				#loss = self.sess.run(self.loss, feed_dict={self.image: x, self.targets: y})
				loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, summary_op], 
										        		 feed_dict={self.image: self.val_data, 
										                   			self.targets: self.val_labels})

				summary_writer.add_summary(summary, i)
				print "step:{0}, loss:{1}, accuracy:{2}".format(self.global_step.eval(), loss, accuracy)


