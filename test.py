from reader import read_cifar10
import os
import tensorflow as tf
import numpy as np
from utils import image_preprocessing
from nin import NIN 

is_training = True

r = read_cifar10(os.getcwd()+'/cifar10_dataset', is_training=is_training)
d, l = r.load_data()

#d = image_preprocessing(d.astype('float32'))

train_dataset = {}
val_dataset = {}
eval_dataset = {}

if is_training is True:
	train_dataset['data'] = d[0:49000]
	train_dataset['labels'] = l[0:49000]
	val_dataset['data'] = image_preprocessing(d[49000:].astype('float32'))
	val_dataset['labels'] = l[49000:]

else:
	eval_dataset['data'] = d
	eval_dataset['labels'] = l

'''
train_dataset['data'] = d[0:50]
train_dataset['labels'] = l[0:50]
val_dataset['data'] = d[0:50]
val_dataset['labels'] = l[0:50]
'''

with tf.Session() as sess:
	if is_training is True:
		nin = NIN(sess=sess
				 ,train_dataset=train_dataset
				 ,val_dataset=val_dataset
				 ,eval_dataset=None
				 ,max_train_step=10**6
				 ,batch_size=128
				 ,is_training=True)

		sess.run(tf.global_variables_initializer())
		nin.train()

	else:
		nin = NIN(sess=sess
				 ,train_dataset = train_dataset
				 ,val_dataset = val_dataset
				 ,eval_dataset = eval_dataset
				 ,max_train_step = 10**6
				 ,batch_size = 128
				 ,is_training = False)
		saver = tf.train.Saver()
		path = tf.train.latest_checkpoint("./tmp")
		saver.restore(sess, path)
		nin.eval()
	
	#d = sess.run(nin.avg_pool, feed_dict={nin.image: train_dataset['data'].reshape(-1,32,32,3), nin.targets: train_dataset['labels']})
	#y = sess.run(nin.y, feed_dict={nin.image: train_dataset['data'].reshape(-1,32,32,3), nin.targets: train_dataset['labels']})
	#print d.shape, y.shape