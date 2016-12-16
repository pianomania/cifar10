from nin import NIN
from Input import Input
import tensorflow as tf

is_training = True

Input = Input(is_training=is_training)

with tf.Session() as sess:
  if is_training is True:
    nin = NIN(sess=sess,
          max_epoch=300,
          Input=Input,
          is_training=True)

    sess.run(tf.global_variables_initializer())
    nin.train()

  else:
    nin = NIN(sess=sess,
          max_epoch=300,
          Input=Input,
          is_training=False)
    saver = tf.train.Saver()
    path = tf.train.latest_checkpoint("./tmp")
    saver.restore(sess, path)
    nin.eval()
  