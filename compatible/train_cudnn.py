import tensorflow as tf


x = tf.placeholder(tf.float32, [1, 1, 1], name='input')
cudnn_gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1,
                                          num_units=2)
_ = cudnn_gru(x)
kernel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
assign_op = tf.assign(kernel, [2.7] * 30)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(assign_op)
    tf.train.Saver().save(sess, './ckpt')
