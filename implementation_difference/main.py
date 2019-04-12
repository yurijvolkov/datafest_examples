import tensorflow as tf
import numpy as np

# CudnnGRU
print('CudnnGRU')
x = tf.placeholder(tf.float32, [5, 1, 1], name='input')
init_state = tf.placeholder(tf.float32, [1, 1, 2], name='init_state')

cell = tf.contrib.cudnn_rnn.CudnnGRU(1, 2,
                                     kernel_initializer=tf.initializers.ones(),
                                     bias_initializer=tf.initializers.zeros())

out, final_state = cell(x, initial_state=tuple([init_state]))
kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print(f'Kernel tensor : {kernel}')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_v, final_state_v, kernel_v = sess.run(
                                        [out, final_state, kernel],
                                        feed_dict={x: np.ones([5, 1, 1]),
                                                   init_state: np.zeros([1, 1, 2])})
print(f'Final state : {final_state_v}\n')
print(f'Output : {out_v}\n')
print(f'Kernel({np.shape(kernel_v)}) : {kernel_v}')


# GRUCell
print('\n\nGRUCell')
tf.reset_default_graph()

x = tf.placeholder(tf.float32, [1, 5, 1], name='input')
init_state = tf.placeholder(tf.float32, [1, 2], name='init_state')

cell = tf.nn.rnn_cell.MultiRNNCell(
                                [tf.nn.rnn_cell.GRUCell(
                                  2,
                                  kernel_initializer=tf.initializers.ones(),
                                  bias_initializer=tf.initializers.constant(0.))])


out, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=tuple([init_state]))
kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

print(f'Kernel tensor : {kernel}\n')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out_v, final_state_v, kernel_v = sess.run(
                                        [out, final_state, kernel],
                                        feed_dict={x: np.ones([1, 5, 1]),
                                                   init_state: np.zeros([1, 2])})
print(f'Final state Cudnn : {final_state_v}\n')
print(f'Output Cudnn : {out_v}\n')
print(f'Kernel Cudnn : {kernel_v}\n')
