import numpy as np
import tensorflow as tf
from utils import res_op, conv_op

class ResNet(object):

  def __init__(self, sess, Input=None, 
         max_epoch=300, 
         batch_size=128, 
         is_training=True,
         pre_activation=True):
  
    self.sess = sess
    self.Input = Input
    self.nstack = 3
    
    self.max_epoch = max_epoch
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    #self.lr = tf.train.exponential_decay(0.1, self.global_step, 383*80, 0.1, staircase=True)
    self.lr = tf.train.piecewise_constant(self.global_step, [32000, 48000], [0.1, 0.01, 0.001])
    #self.lr = tf.train.polynomial_decay(0.1, self.global_step, 48000, 0.001)

    # define placeholder in train or eval usage
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.targets = tf.placeholder(dtype=tf.int64, shape=[None])
    
    res = conv_op('conv1', self.image, [3, 3, 3, 16], [1]*4,
                  wd=0.0001,
                  is_training=is_training)

    res = res_op('res1_0', res, [3, 3, 16 ,16], 1, pre_activation=False)

    for i in xrange(1, self.nstack):
      res = res_op('res1_%d' % i, res, [3, 3, 16, 16], 1)

    for i in xrange(0, self.nstack):
      res = res_op('res2_%d' % i, res, [3, 3, i==0 and 16 or 32, 32], i==1 and 2 or 1)

    for i in xrange(0, self.nstack):
      res = res_op('res3_%d' % i, res, [3, 3, i==0 and 32 or 64, 64], i==1 and 2 or 1)

    with tf.variable_scope('avg_pool'):
      res = tf.contrib.layers.batch_norm(res, is_training=is_training)
      res = tf.nn.relu(res)
      
      res_shape = res.get_shape().as_list()
      avg_pool = tf.nn.avg_pool(res, 
                                ksize=[1, res_shape[1], res_shape[2], 1], 
                                strides=[1]*4, 
                                padding='VALID')

    class_logits = conv_op('output', avg_pool, 
                  [1, 1, 64, 10],
                  stride = [1, 1, 1, 1],
                  use_relu = False,
                  use_batch_norm = False,
                  wd = 0.0001,
                  padding='VALID')
    self.class_logits = tf.squeeze(class_logits)

    # define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.class_logits, self.targets)
    total_reg_loss = tf.add_n(tf.get_collection('reg_losses'), name='total_reg_loss')

    self.loss = tf.reduce_mean(cross_entropy) + total_reg_loss

    self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
    self.train_op = self.optimize.minimize(self.loss, global_step=self.global_step)

    correct_prediction = tf.equal(tf.argmax(self.class_logits, 1), self.targets)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  def summary_setting(self):

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    self.summary_op = tf.merge_all_summaries()
    self.summary_writer = tf.summary.FileWriter('/home/yao/cifar10/tmp/train_summary', self.sess.graph)

  def train(self):

    self.summary_setting()
    saver = tf.train.Saver()

    step = 1

    for epoch in range(self.max_epoch):
      for i in range(self.Input.epoch_size):
        x, y = self.Input.batch()
        self.sess.run(self.train_op, 
                      feed_dict={self.image: x, 
                                 self.targets: y})
        if step % 20 == 0:
          self.validate(x, y) 

        step +=1

      saver.save(self.sess, "./tmp/model")
      print "---Model have been saved---"

  def validate(self, x, y):

    loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, self.summary_op], 
                                            feed_dict={self.image: self.Input.val_data, 
                                                       self.targets: self.Input.val_labels})

    t_loss, t_accuracy = self.sess.run([self.loss, self.accuracy],
                                       feed_dict={self.image: x,
                                                  self.targets: y})

    self.summary_writer.add_summary(summary, self.global_step.eval())
    print "step:{0}, loss: {1}, accuracy: {2}".format(self.global_step.eval(), loss, accuracy)
    print "train: loss: {0}, accuracy: {1}".format(t_loss, t_accuracy)

  def eval(self):
    ''' function for evaluating test data '''

    accuracy = self.sess.run(self.accuracy, 
                             feed_dict={self.image: self.Input.eval_data, 
                             self.targets: self.Input.eval_labels})

    print "accuracy %f" % accuracy

