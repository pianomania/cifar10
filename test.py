from nin import NIN
from Input import Input
import tensorflow as tf

is_training = True
with tf.Session() as sess:

  Input = Input(is_training=is_training)

  nin = NIN(sess=sess,
        max_epoch=300,
        Input=Input,
        is_training=is_training)

  if is_training == True:
    sess.run(tf.global_variables_initializer())
    nin.train()
  else:
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint("./tmp")
    saver.restore(sess, path)
    nin.eval()
  