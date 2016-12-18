import tensorflow as tf
from collections import namedtuple

hps = namedtuple('hps', 'max_epoch')

class trainer(object):

  def __init__(self, sess, model, Input, hps):

    self.sess = sess
    self.model = model
    self.Input = Input
    self.hps = hps

    self.image = model.image
    self.target = model.target
    self.class_logits = model.class_logits
    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    self._train()

  def _train(self):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.class_logits, self.target)
    total_reg_loss = tf.add_n(tf.get_collection('reg_losses'), name='total_reg_loss')

    self.loss = tf.reduce_mean(cross_entropy) + total_reg_loss

    self.lr = tf.train.piecewise_constant(self.global_step, [32000, 48000], [0.1, 0.01, 0.001])

    self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
    self.train_op = self.optimize.minimize(self.loss, global_step=self.global_step)

    correct_prediction = tf.equal(tf.argmax(self.class_logits, 1), self.target)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  def summary_setting(self):

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    self.summary_op = tf.merge_all_summaries()
    self.summary_writer = tf.summary.FileWriter('/home/yao/cifar10/tmp/train_summary', self.sess.graph)

  def train(self):

    self.summary_setting()
    saver = tf.train.Saver()

    hps = self.hps

    step = 1
    for epoch in range(hps.max_epoch):
      for i in range(self.Input.epoch_size):
        x, y = self.Input.batch()
        self.sess.run(self.train_op, 
                      feed_dict={self.image: x, 
                                 self.target: y})
        if step % 20 == 0:
          self.validate(x, y) 

        step +=1

      saver.save(self.sess, "./tmp/model")
      print "---Model have been saved---"

  def validate(self, x, y):

    loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, self.summary_op], 
                                            feed_dict={self.image: self.Input.val_data, 
                                                       self.target: self.Input.val_labels})

    t_loss, t_accuracy = self.sess.run([self.loss, self.accuracy],
                                       feed_dict={self.image: x,
                                                  self.target: y})

    self.summary_writer.add_summary(summary, self.global_step.eval())
    print "step:{0}, loss: {1}, accuracy: {2}".format(self.global_step.eval(), loss, accuracy)
    print "train: loss: {0}, accuracy: {1}".format(t_loss, t_accuracy)

  def eval(self):

    accuracy = self.sess.run(self.accuracy, 
                             feed_dict={self.image: self.Input.eval_data, 
                             self.target: self.Input.eval_labels})

    print "accuracy %f" % accuracy
