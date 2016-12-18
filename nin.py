import numpy as np
import tensorflow as tf
from utils import conv_op, mlpconv


class NIN(object):

  def __init__(self, sess, Input=None, 
         max_epoch=300, 
         batch_size=128, 
         is_training=True):
  
    self.sess = sess
    self.Input = Input
    
    self.max_epoch = max_epoch
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.lr = tf.train.exponential_decay(0.1, self.global_step, 383*50, 0.5, staircase=True)

    # define placeholder in train or eval usage
    self.image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
    self.targets = tf.placeholder(dtype=tf.int64, shape=[None])
    self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

    # define main architecture
    y = mlpconv('mlpconv_layer1', self.image, [5, 5, 3, 192], [1, 2, 2, 1], [160, 96], is_training=is_training)
    y = tf.nn.dropout(y, self.keep_prob)
    y = mlpconv('mlpconv_layer2', y, [5, 5, 96, 192], [1, 2, 2, 1], [156, 128], is_training=is_training)
    y = tf.nn.dropout(y, self.keep_prob)
    y = mlpconv('mlpconv_output', y, [3, 3, 128, 192], [1, 1, 1, 1], [156, 128], is_training=is_training)
    y = tf.nn.dropout(y, self.keep_prob)
    self.y = y

    y_shape = y.get_shape().as_list()
    
    # global averaging
    avg_pool = tf.nn.avg_pool(y, ksize=[1, y_shape[1], y_shape[2], 1], strides=[1]*4, padding='VALID')
    # Version 1
    #self.class_logits = tf.squeeze(avg_pool)
    
    # Version 2
    #avg_pool_shape = avg_pool.get_shape().as_list()
    self.class_logits = conv_op('output', avg_pool, 
                  [1, 1, 128, 10], 
                  stride = [1, 1, 1, 1],
                  padding='VALID')
    self.class_logits = tf.squeeze(self.class_logits)
    
    # define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.class_logits, self.targets)
    total_reg_loss = tf.add_n(tf.get_collection('reg_losses'), name='total_reg_loss')

    self.loss = tf.reduce_mean(cross_entropy) + total_reg_loss

    self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
    self.train_op = self.optimize.minimize(self.loss, global_step=self.global_step)

    # define accuracy
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
                                 self.targets: y,
                                 self.keep_prob: 0.75})
        if step % 20 == 0:
          self.validate(x, y) 

        step +=1

      saver.save(self.sess, "./tmp/model")
      print "---Model have been saved---"

  def validate(self, x, y):

    loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, self.summary_op], 
                                            feed_dict={self.image: self.Input.val_data, 
                                                       self.targets: self.Input.val_labels,
                                                       self.keep_prob: 1.0})

    t_loss, t_accuracy = self.sess.run([self.loss, self.accuracy],
                                       feed_dict={self.image: x,
                                                  self.targets: y,
                                                  self.keep_prob: 1.0})

    self.summary_writer.add_summary(summary, self.global_step.eval())
    print "step:{0}, loss: {1}, accuracy: {2}".format(self.global_step.eval(), loss, accuracy)
    print "train: loss: {0}, accuracy: {1}".format(t_loss, t_accuracy)

  def eval(self):
    ''' function for evaluating test data '''

    accuracy = self.sess.run(self.accuracy, 
                             feed_dict={self.image: self.Input.eval_data, 
                             self.targets: self.Input.eval_labels,
                             self.keep_prob: 1.0})

    print "accuracy %f" % accuracy

