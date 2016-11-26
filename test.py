from reader import read_cifar10
import os
import tensorflow as tf
import numpy as np
from nin import NIN 

r = read_cifar10(os.getcwd()+'/cifar10_dataset')
d, l = r.load_data()

with tf.Session() as sess:

	nin = NIN(sess=sess, dataset={'data':d, 'labels': l}, max_train_step=10**6, batch_size=128)

	sess.run(tf.initialize_all_variables())

	nin.train()
	'''
	y, avg_pool, loss = sess.run([nin.y, nin.avg_pool, nin.loss], feed_dict={nin.image: np.random.random((32,32,32,3)),
																		     nin.targets: np.random.randint(0,10,32)})
	
	print y.shape
	print avg_pool.shape
	print loss


	sess.run(nin.optimize, feed_dict={nin.image: np.random.random((32,32,32,3)),nin.targets: np.random.randint(0,10,32)})
	'''

	