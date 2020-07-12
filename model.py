import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
# https://github.com/princewen/tensorflow_practice/blob/master/RL/myPtrNetwork/model.py
# https://www.jianshu.com/p/2ad389e91467

from layers import *

class Model(object):
  def __init__(self, config, 
               inputs, labels, enc_seq_length, dec_seq_length, mask,
               reuse=False, is_critic=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.max_enc_length = config.max_enc_length
    self.max_dec_length = config.max_dec_length
    self.num_glimpse = config.num_glimpse

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.use_terminal_symbol = config.use_terminal_symbol

    self.lr_start = config.lr_start
    self.lr_decay_step = config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    ##############
    # inputs
    ##############

    self.is_training = tf.placeholder_with_default( # 带有默认值的placeholder
        tf.constant(False, dtype=tf.bool),
        shape=(), name='is_training'
    )

    """
    self.enc_seq = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.max_enc_length,input_dim=2],name='enc_seq')
    self.target_seq = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_dec_length],name='target_seq')
    self.enc_seq_length = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='enc_seq_length')
    self.target_seq_length = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='target_seq_length')
    """
    self.enc_inputs, self.dec_targets, self.enc_seq_length, self.dec_seq_length, self.mask = \
        smart_cond(
            self.is_training,
            lambda: (inputs['train'], labels['train'], enc_seq_length['train'],
                     dec_seq_length['train'], mask['train']),
            lambda: (inputs['test'], labels['test'], enc_seq_length['test'],
                     dec_seq_length['test'], mask['test'])
        )

    if self.use_terminal_symbol:
      self.dec_seq_length += 1 # terminal symbol

    self._build_model()
    self._build_steps()

    if not reuse:
      self._build_optim()

    self.train_summary = tf.summary.merge([
        tf.summary.scalar("train/total_loss", self.total_loss),
        tf.summary.scalar("train/lr", self.lr),
    ])

    self.test_summary = tf.summary.merge([
        tf.summary.scalar("test/total_loss", self.total_loss),
    ])

  def _build_steps(self):
    def run(sess, fetch, feed_dict, summary_writer, summary):
      fetch['step'] = self.global_step
      if summary is not None:
        fetch['summary'] = summary

      result = sess.run(fetch)
      if summary_writer is not None:
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train(sess, fetch, summary_writer):
      return run(sess, fetch, feed_dict={},
                 summary_writer=summary_writer, summary=self.train_summary)

    def test(sess, fetch, summary_writer=None):
      return run(sess, fetch, feed_dict={self.is_training: False},
                 summary_writer=summary_writer, summary=self.test_summary)

    self.train = train
    self.test = test

  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)

    """
    我们要对输入进行处理，将输入转换为embedding，embedding的长度和lstm的隐藏神经元个数相同。
    这里指的注意的就是 tf.nn.conv1d函数了，这个函数首先会对输入进行一个扩展，然后再调用tf.nn.conv2d进行二维卷积。关于该函数的过程可以看代码中的注释或者看该函数的源代码。
    
    input_dim 是 2，hidden_dim 是 lstm的隐藏层的数量
    
    # 将 输入转换成embedding,下面是根据源码的转换过程：
    # enc_seq :[batch_size,seq_length,2] -> [batch_size,1,seq_length,2]，在第一维进行维数扩展, 2是二维的x,y坐标, 看成NHWC
    # input_embed : [1,2,256] -> [1,1,2,256] # 在第0维进行维数扩展, 作为filters=[height,width, in_channel, out_channel]
    # 所以卷积后的输出为: [batch, 1, seq_length, out_channel]
    
    # tf.nn.conv1d首先将input和filter进行填充，然后进行二维卷积，因此卷积之后维度为batch * 1 * seq_length * 256
    # 最后还有一步squeeze的操作，从tensor中删除所有大小是1的维度，所以最后的维数为batch * seq_length * 256
    # 即将输入数据:[batch, seq_length, input_dim=2] -> 高维[batch, seq_length, hidden_dim=256], 其实就相当于最后一个维度全连接而己
    """
    # input_embed: [filter_width=1, input_channel=input_dim, output_channel=hidden_dim]
    input_embed = tf.get_variable(
        "input_embed", [1, self.input_dim, self.hidden_dim],
        initializer=self.initializer)
    # enc_inputs: [batch, max_enc_length, input_channel=2] => [batch, in_height=1, in_width=seq_length, in_channels=input_dim=2]
    # input_embed: [filter_width=1, input_channel=input_dim=2, output_channel=hidden_dim]
    #           => [filter_height=1, filter_width=1, in_channels=input_dim=2, out_channels=hidden_dim]
    # conv2d: out_size = (img_size+2*pad-filter_size)//stride+1 = (img_size-1)//1+1=img_size,即保持原来大小
    # embeded_enc_inputs: [batch, out_height=1, out_width=seq_length=max_enc_length, output_channel=hidden_dim=256]
    #                  => [batch, seq_length=max_enc_length, hidden_dim=256]
    with tf.variable_scope("encoder"):
      # 可以看出来,作者选用的是1*1的卷积,即基于单像素在不同通道上的卷积,两个通道分别代表[x,y]坐标
      self.embeded_enc_inputs = tf.nn.conv1d(
          values=self.enc_inputs, filters=input_embed, stride=1, padding="VALID")

    # -----------------encoder------------------
    batch_size = tf.shape(self.enc_inputs)[0]
    with tf.variable_scope("encoder"):
      # 构建一个多层的LSTM
      self.enc_cell = LSTMCell(
          self.hidden_dim, # hidden_dim:256
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.enc_cell] * self.num_layers # num_layers=1,代表有多少层lstm layer
        self.enc_cell = MultiRNNCell(cells)
      # 初始化rnn
      self.enc_init_state = trainable_initial_state(batch_size, self.enc_cell.state_size)

      # embeded_enc_inputs: [batch, seq_length=max_enc_length, hidden_dim=256]
      # self.encoder_outputs = output_all_hidden_states: [batch_size, seq_length, hidden_dim]
      # self.enc_final_states = last_cell_and_hidden_state: {c: [batch_size, hidden_size], h:[batch_size, hidden_size]}
      self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell,
          self.embeded_enc_inputs,
          self.enc_seq_length,
          self.enc_init_state)

      # 给最开头添加一个开始标记SOS，同时这个标记也将作为decoder的初始输入
      # first_decoder_input:[batch_size,1,hidden_dim]
      self.first_decoder_input = tf.expand_dims(
          trainable_initial_state(batch_size, self.hidden_dim, name="first_decoder_input"),
          axis=1)

      if self.use_terminal_symbol:
        # 0 index indicates terminal
        # first_decoder_input: [batch_size,1,hidden_dim]
        # encoder_outputs: [batch_size, seq_length, hidden_dim]
        #  => [batch_size, 1+seq_length, hidden_dim]
        self.enc_outputs = concat_v2([self.first_decoder_input, self.enc_outputs], axis=1)

    # -----------------decoder 训练--------------------
    with tf.variable_scope("decoder"):
      # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
      #                        [[0, 2], [1, 3], [2, 1]]]
      # dec_targets:[batch_size, max_dec_length]
      # idx_paris: [batch_size, max_dec_length, 2]
      self.idx_pairs = index_matrix_to_pairs(self.dec_targets)

      # encoder_outputs: [batch_size, 1+seq_length, hidden_dim]
      # idx_paris:       [batch_size, max_dec_length, 2]
      # embedd_dec_inputs: [batch_size, ]
      self.embeded_dec_inputs = tf.stop_gradient(tf.gather_nd(self.enc_outputs, self.idx_pairs)) # 此处没有梯度

      if self.use_terminal_symbol:
        # 给target最后一维增加结束标记,数据都是从1开始的，所以结束也是回到1，所以结束标记为1
        # tiled_zero_idxs:[batch, 1]
        tiled_zero_idxs = tf.tile(tf.zeros([1, 1], dtype=tf.int32),
                                  [batch_size, 1],
                                  name="tiled_zero_idxs")
        self.dec_targets = concat_v2([self.dec_targets, tiled_zero_idxs], axis=1)

      # 如果使用了结束标记的话，要给encoder的输出拼上开始状态，同时给decoder的输入拼上开始状态
      self.embeded_dec_inputs = concat_v2([self.first_decoder_input, self.embeded_dec_inputs], axis=1)

      # 建立一个多层的lstm网络
      self.dec_cell = LSTMCell( self.hidden_dim, initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_pred_logits, _, _ = decoder_rnn(
          self.dec_cell, self.embeded_dec_inputs,
          self.enc_outputs, self.enc_final_states, # encoder的最后的状态作为decoder的初始状态
          self.dec_seq_length, self.hidden_dim,
          self.num_glimpse, batch_size, is_train=True,
          initializer=self.initializer)

      self.dec_pred_prob = tf.nn.softmax(self.dec_pred_logits, 2, name="dec_pred_prob")
      self.dec_pred = tf.argmax(self.dec_pred_logits, 2, name="dec_pred")

    with tf.variable_scope("decoder", reuse=True):
      self.dec_inference_logits, _, _ = decoder_rnn(
          self.dec_cell, self.first_decoder_input,
          self.enc_outputs, self.enc_final_states,
          self.dec_seq_length, self.hidden_dim,
          self.num_glimpse, batch_size, is_train=False,
          initializer=self.initializer,
          max_length=self.max_dec_length + int(self.use_terminal_symbol))
      self.dec_inference_prob = tf.nn.softmax(self.dec_inference_logits, 2, name="dec_inference_logits")
      self.dec_inference = tf.argmax(self.dec_inference_logits, 2, name="dec_inference")

  def _build_optim(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.dec_targets, logits=self.dec_pred_logits)
    inference_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.dec_targets, logits=self.dec_inference_logits)

    def apply_mask(op):
      length = tf.cast(op[:1], tf.int32)
      loss = op[1:]
      return tf.multiply(loss, tf.ones(length, dtype=tf.float32))

    batch_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_loss")

    batch_inference_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_inference_loss")

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    self.total_loss = total_loss
    self.target_cross_entropy_losses = losses
    self.total_inference_loss = batch_inference_loss

    self.lr = tf.train.exponential_decay(self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.total_loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    else:
      self.optim = optimizer.minimize(self.total_loss, global_step=self.global_step)
