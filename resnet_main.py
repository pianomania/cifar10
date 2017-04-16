import tensorflow as tf
from Input import Input
from resnet import ResNet
from trainer import trainer
import os
import time

hps = {'max_epoch': 178, 'nstack':8}

with tf.Session() as sess:
  
  is_training = True
  Input = Input(is_training=is_training)
  path = '/home/yao/cifar10/tmp/resnet%d' % (hps['nstack']*6+2)

  if is_training is True:
    if os.path.exists(path):
      for fname in os.listdir(path):
        os.remove(path+'/'+fname)
    else:
      os.mkdir(path)

  resnet = ResNet(hps)
  resnet_trainer = trainer(sess, resnet, Input, hps, path)

  if is_training is True:
    sess.run(tf.global_variables_initializer())
    resnet_trainer.train()

  else:
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint(path)
    saver.restore(sess, path)
    resnet_trainer.eval()