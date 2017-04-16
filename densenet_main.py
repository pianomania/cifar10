import tensorflow as tf
from Input import Input
from densenet import DenseNet
from trainer import trainer
import os
import time

hps = {'max_epoch': 178}

with tf.Session() as sess:
  
  is_training = True
  path = '/home/yao/cifar10/tmp/DenseNet'
  Input = Input(is_training=is_training, batch_num=128)

  if is_training is True:
    if os.path.exists(path):
      for fname in os.listdir(path):
        os.remove(path+'/'+fname)
    else:
      os.mkdir(path)

  model = DenseNet(k=12)
  model_trainer = trainer(sess, model, Input, hps, path)

  if is_training is True:
    sess.run(tf.global_variables_initializer())
    model_trainer.train()

  else:
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint(path)
    saver.restore(sess, path)
    model_trainer.eval()