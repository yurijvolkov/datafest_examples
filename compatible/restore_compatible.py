import tensorflow as tf

from tensorflow.contrib.cudnn_rnn import \
        CudnnCompatibleGRUCell


with tf.variable_scope("cudnn_gru"):
    x = tf.placeholder(tf.float32, [1, 1, 1],
                       name='input')
    cells = [CudnnCompatibleGRUCell(2)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    kernel = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, './ckpt')
    print(sess.run(kernel))
