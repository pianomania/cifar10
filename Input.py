import numpy as np
import os
from reader import read_cifar10

class Input(object):

  def __init__(self, is_training, batch_num=128):

    self.is_training = is_training
    self.batch_num = batch_num

    r = read_cifar10(os.getcwd()+'/cifar10_dataset', is_training=is_training)
    d, l = r.load_data()

    if self.is_training  is True:

      self.train_data = d[0:49000].astype('float32').reshape(-1, 32, 32, 3)
      self.train_labels = l[0:49000]
      self.val_data = d[49000:].astype('float32').reshape(-1, 32, 32, 3)
      self.val_data = self._whitening(self.val_data)
      self.val_labels = l[49000:]

      self.num_data = self.train_data.shape[0]
      self.epoch_size = np.ceil(self.num_data/float(batch_num)).astype('int32')

    else:

      self.eval_data = d.astype('float32').reshape(-1, 32, 32, 3)
      self.eval_data = self._whitening(self.eval_data)
      self.eval_labels = l

  def _whitening(self, x):

    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    return x

  def _augmenting(self, x, idx):
    x = self.train_data[idx]
    # random cropping
    '''
    xx = np.random.randint(5)
    yy = np.random.randint(5)
    x = np.pad(x, ((0,),(2,),(2,),(0,)), 'constant')[:, xx:xx+32, yy:yy+32, :]
    '''
    for i in xrange(x.shape[0]):
      x[i] = self._random_crop(x[i])
    # random fliping image
    flip_idx = np.random.permutation(x.shape[0])[:x.shape[0]/2]
    x[flip_idx] = x[flip_idx,:,::-1,:]
    # contrastness and brightness
    gain = 0.5 + np.random.rand(x.shape[0], 1, 1, 1)
    bias = np.random.randint(-50, 50, (x.shape[0], 1, 1, 1))
    x = x * gain + bias
    x = np.clip(x, 0.0, 255.0)
    # whitening
    x = self._whitening(x)

    return x

  def batch(self):
    
    try:
      idx = self.batch_idx_iter.next()

    except (AttributeError, StopIteration):
      print "Start a new epoch!"

      self.batch_idx = np.array_split(np.random.permutation(self.num_data),
                            self.epoch_size)
      self.batch_idx_iter = iter(self.batch_idx)
      idx = self.batch_idx_iter.next()

    x = self.train_data[idx]
    y = self.train_labels[idx]
    x = self._augmenting(x, idx)

    return x, y

  def _random_crop(self, x):

    xx = np.random.randint(9)
    yy = np.random.randint(9)
    x = np.pad(x, ((4,),(4,),(0,)), 'constant')[xx:xx+32, yy:yy+32, :]

    return x
