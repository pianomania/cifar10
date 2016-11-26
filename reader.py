import numpy as np
import os
import re
import cPickle

class read_cifar10(object):

	def __init__(self, data_path=None):
		self.data_path = data_path

	def load_data(self):

		files = os.listdir(self.data_path)
		pattern = re.compile('(data_batch_).')

		to_read = [m.group(0) for i in files for m in [pattern.search(i)] if m] 

		data = []
		labels = []

		for t in to_read:
			with open(self.data_path+'/'+t, 'rb') as f:
				d = cPickle.load(f)
				data.append(d['data'])
				labels.append(d['labels'])

		data = np.vstack(data)
		labels = np.hstack(labels)
		
		return data, labels

