import tensorflow as tf
from collections import namedtuple

hps = namedtuple('hps', 'max_epoch')

class trainer(object):

  def __init__(self, sess, model, Input, hps, path):

    self.sess = sess
    self.model = model
    self.Input = Input
    self.hps = hps
    self.path = path

    self.image = model.image
    self.target = model.target
    self.logits = model.logits
    self.is_training = model.is_training

    if 'keep_probs' in hps:
      self.keep_probs = model.keep_probs
    else:
      self.keep_probs = None


    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    self._train()
    self._train_feed()

  def _train_feed(self):

    if self.keep_probs is None:
      feed = lambda x, y, z: {self.image: x,
                                 self.target: y,
                                 self.is_training: z}

      self.train_feed = lambda x, y: feed(x, y, True)
      self.eval_feed = lambda x, y: feed(x, y, False)

    else:
      feed = lambda x, y, z, p: {self.image: x,
                                 self.target: y,
                                 self.is_training: z,
                                 self.keep_probs: p}

      self.train_feed = lambda x, y: feed(x, y, True, hp['keep_probs'])
      self.eval_feed = lambda x, y: feed(x, y, False, 1.0)

    
    

  def _train(self):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.target)
    total_reg_loss = tf.add_n(tf.get_collection('reg_losses'), name='total_reg_loss')

    self.loss = tf.reduce_mean(cross_entropy) + total_reg_loss

    self.lr = tf.train.piecewise_constant(self.global_step, [32000, 48000], [0.1, 0.01, 0.001])

    self.optimize = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
    self.train_op = self.optimize.minimize(self.loss, global_step=self.global_step)

    correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.target)
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

  def summary_setting(self):

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    self.summary_op = tf.merge_all_summaries()
    self.summary_writer = tf.summary.FileWriter(self.path, self.sess.graph)

  def train(self):

    self.summary_setting()
    saver = tf.train.Saver()

    hps = self.hps

    step = 1
    for epoch in range(hps.max_epoch):
      for i in range(self.Input.epoch_size):
        x, y = self.Input.batch()
        self.sess.run(self.train_op, 
                      feed_dict=self.train_feed(x, y))

        if step % 20 == 0:
          self.validate(x, y) 

        step +=1

      saver.save(self.sess, self.path + '/model')
      print "---Model have been saved---"

  def validate(self, x, y):

    loss, accuracy, summary = self.sess.run([self.loss, self.accuracy, self.summary_op], 
                                            feed_dict=self.eval_feed(self.Input.val_data,
                                                                     self.Input.val_labels))

    t_loss, t_accuracy = self.sess.run([self.loss, self.accuracy],
                                       feed_dict=self.eval_feed(x, y))

    self.summary_writer.add_summary(summary, self.global_step.eval())
    print "step:{0}, loss: {1}, accuracy: {2}".format(self.global_step.eval(), loss, accuracy)
    print "train: loss: {0}, accuracy: {1}".format(t_loss, t_accuracy)

  def eval(self):

    accuracy = self.sess.run(self.accuracy, 
                             feed_dict=self.eval_feed(self.Input.eval_data,
                                                      self.Input.eval_labels))

    print "accuracy %f" % accuracy
