import tensorflow as tf
from Input import Input
from nin import NIN
from trainer import trainer, hps
import os

hps = {'max_epoch': 178, 'keep_probs':0.75}

with tf.Session() as sess:
  
  is_training = True
  Input = Input(is_training=is_training)

  path = '/home/yao/cifar10/tmp/nin'

  nin = NIN()
  nin_trainer = trainer(sess, nin, Input, hps, path)

  if is_training is True:

    if os.path.exists(path):
      for fname in os.listdir(path):
        os.remove(path+'/'+fname)
    else:
      os.mkdir(path)

    sess.run(tf.global_variables_initializer())
    nin_trainer.train()
  else:
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint(path)
    saver.restore(sess, path)
    nin_trainer.eval()
