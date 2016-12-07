import numpy as np
import tensorflow as tf
from utils import conv_op, mlpconv


class NIN(object):

	def __init__(self, sess, train_dataset, val_dataset, eval_dataset, max_train_step, batch_size, is_training=True):

		self.sess = sess

		if is_training is True:
			self.data = train_dataset['data'].astype('float32').reshape(-1, 32, 32, 3)
			self.labels = train_dataset['labels']
			self.val_data = val_dataset['data'].astype('float32').reshape(-1, 32, 32, 3)
			self.val_labels = val_dataset['labels']

			self.max_train_step = max_train_step
			self.batch_size = batch_size


			self.global_step = tf.Variable(0, trainable=False, name='global_step')
			self.lr = tf.train.exponential_decay(1e-2, self.global_step, 2000, 0.9, staircase=True)

		else:

			self.eval_data = eval_dataset['data'].astype('float32').reshape(-1, 32, 32, 3)
			self.eval_labels = eval_dataset['labels']


		# define placeholder in train or eval usage
		self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
		self.targets = tf.placeholder(dtype=tf.int64, shape=[None])

		# define main architecture
		y = mlpconv('mlpconv_layer1', self.image, [5, 5, 3, 192], [1, 2, 2, 1], [160, 128], is_training=is_training)
		y = mlpconv('mlpconv_layer2', y, [5, 5, 128, 192], [1, 2, 2, 1], [192, 192], is_training=is_training)
		y = mlpconv('mlpconv_layer3', y, [3, 3, 192, 192], [1, 2, 2, 1], [192, 192], is_training=is_training)
		#y = mlpconv('mlpconv_layer4', y, [3, 3, 192, 192], [1, 1, 1, 1], [192, 192])
		#y = mlpconv('mlpconv_output', y, [3, 3, 192, 192], [1, 1, 1, 1], [192, 192])
		self.y = y

		y_shape = y.get_shape().as_list()
		'''
		# global averaging
		avg_pool = tf.nn.avg_pool(y, ksize=[1, y_shape[1], y_shape[2], 1], strides=[1]*4, padding='VALID')
		self.avg_pool = tf.squeeze(avg_pool)
		'''
		# fc out
		avg_pool = conv_op(y, [y_shape[1], y_shape[2], 192, 10], [1, 1, 1, 1],
			  			   use_relu = False,
			  			   use_batch_norm = False,
			  			   padding = 'VALID',
			  			   wd = 0.01
			  			   )
		
		self.avg_pool = tf.squeeze(avg_pool)
	
		# define loss
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.avg_pool, self.targets)
		total_reg_loss = tf.add_n(tf.get_collection('reg_losses'), name='total_reg_loss')

		self.loss = tf.reduce_mean(cross_entropy) + total_reg_loss

		if is_training is True:
			self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss, global_step=self.global_step)

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

	def summary_setting(self):

		tf.scalar_summary('loss', self.loss)
		tf.scalar_summary('accuracy', self.accuracy)
		self.summary_op = tf.merge_all_summaries()
		self.summary_writer = tf.train.SummaryWriter('/home/yao/cifar10/tmp/train_summary', self.sess.graph)

	def _batch(self, idx):

		#idx = np.random.randint(0, self.data.shape[0], self.batch_size)
		x = self.data[idx]
		y = self.labels[idx]

		# random intensity
		#x = self._fancy_pca(x)
		
		# random fliping image
		flip_idx = np.random.permutation(x.shape[0])[:x.shape[0]/2]
		x[flip_idx] = x[flip_idx,:,:,::-1]

		'''
		# contrastness and brightness
		gain = 0.2 + np.random.rand(x.shape[0], 1, 1, 1)*1.6
		#bias = np.random.randint(-20, 21, (bc_idx.shape[0], 1, 1, 1))
		x = x * gain #+ bias
		x = np.clip(x, 0.0, 255.0)
		'''

		return x, y

	def train(self):

		self.summary_setting()
		saver = tf.train.Saver()

		epoch = 1
		step = 1

		for epoch in range(900):
			
			num_data = self.data.shape[0]
			group_idx = np.array_split(np.random.permutation(num_data), np.ceil(num_data/128))
			for idx in group_idx:

				x, y = self._batch(idx)
				self.sess.run(self.optimize, feed_dict={self.image: x, self.targets: y})	
				if step % 20 == 0 :

					self.validate()
					
				step +=1

			saver.save(self.sess, "./tmp/model")
			print "---Model have been saved---"

	def validate(self):

		loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, self.summary_op], 
								        		 feed_dict={self.image: self.val_data, 
								                   			self.targets: self.val_labels})

		self.summary_writer.add_summary(summary, self.global_step.eval())
		print "step:{0}, loss:{1}, accuracy:{2}".format(self.global_step.eval(), loss, accuracy)

	def eval(self):
		''' function for evaluating test data '''

		accuracy = self.sess.run(self.accuracy, 
								  feed_dict={self.image: self.eval_data, 
								             self.targets: self.eval_labels})

		print "accuracy %f" % accuracy
