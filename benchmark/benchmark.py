import tensorflow as tf
import numpy as np

from time import perf_counter
from tqdm import tqdm


def benchmark_dynamic_rnn(cell, batch_size, time_steps, depth,
                          num_iter=1000, hot_start_after=10,
                          num_units=100, verbose=False):
    """
            Benchmarks RNN cells that are compatible with `dynamic_rnn`
        interface.

        Args:
            cell (RNNCell): Rnn cell
            batch_size (int):
            time_steps (int):
            depth (int):
            num_iter (int): Total number of iterations to run.
            hot_start_after (int): Function starts to benchmark
                calculations only after `hot_start_after` iterations.
            num_units (int): Number of units in RNN layer.
            verbose (bool):
        Returns: tuple(list, list)
            First element is benchmarks for inference and second
            for training with Adam.
    """

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, None, depth], name='input')
    y = tf.placeholder(tf.float32, [None, num_units], name='y')
    cell = tf.nn.rnn_cell.MultiRNNCell([cell])

    out, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    loss = tf.reduce_mean(tf.abs(y - final_state))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    inference_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking inference...'):
            start_ts = perf_counter()
            sess.run(out, {x: np.random.randn(batch_size, time_steps, depth)})
            end_ts = perf_counter()
            if i > hot_start_after:
                inference_time.append(end_ts - start_ts)

    train_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking train...'):
            start_ts = perf_counter()
            sess.run(train_step,
                     {x: np.random.randn(batch_size, time_steps, depth),
                      y: np.random.randn(batch_size, num_units)})
            end_ts = perf_counter()
            if i > hot_start_after:
                train_time.append(end_ts - start_ts)

    return inference_time, train_time


def benchmark_cudnn(cell_name, batch_size, time_steps, depth,
                    num_iter=1000, hot_start_after=10,
                    num_units=100, verbose=False):
    """
            Benchmarks RNN cells that use Cudnn interface.

        Args:
            cell (RNNCell): Rnn cell
            batch_size (int):
            time_steps (int):
            depth (int):
            num_iter (int): Total number of iterations to run.
            hot_start_after (int): Function starts to benchmark
                calculations only after `hot_start_after` iterations.
            num_units (int): Number of units in RNN layer.
            verbose (bool):
        Returns: tuple(list, list)
            First element is benchmarks for inference and second
            for training with Adam.
    """

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, None, depth], name='input')
    y = tf.placeholder(tf.float32, [None, num_units], name='y')

    if cell_name == 'gru':
        cell = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
        init_state = (tf.random.normal([1, batch_size, num_units]),)
    elif cell_name == 'lstm':
        cell = tf.contrib.cudnn_rnn.CudnnLSTM(1, num_units)
        init_state = (tf.random.normal([1, batch_size, num_units]),
                      tf.random.normal([1, batch_size, num_units]))

    out, final_state = cell(x, init_state)
    loss = tf.reduce_mean(tf.abs(y - final_state))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    inference_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking inference...'):
            start_ts = perf_counter()
            sess.run(out,
                     {x: np.random.randn(time_steps, batch_size, depth)})
            end_ts = perf_counter()
            if i > hot_start_after:
                inference_time.append(end_ts - start_ts)

    train_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking train...'):
            start_ts = perf_counter()
            sess.run(train_step,
                     {x: np.random.randn(time_steps, batch_size, depth),
                      y: np.random.randn(batch_size, num_units)})
            end_ts = perf_counter()
            if i > hot_start_after:
                train_time.append(end_ts - start_ts)

    return inference_time, train_time


def benchmark_fused_rnn(cell, batch_size, time_steps, depth,
                        num_iter=1000, hot_start_after=10,
                        num_units=100, verbose=False):
    """
            Benchmarks LSTMBlockFusedCell.

        Args:
            cell (RNNCell): LSTMBlockFusedCell
            batch_size (int):
            time_steps (int):
            depth (int):
            num_iter (int): Total number of iterations to run.
            hot_start_after (int): Function starts to benchmark
                calculations only after `hot_start_after` iterations.
            num_units (int): Number of units in RNN layer.
            verbose (bool):
        Returns: tuple(list, list)
            First element is benchmarks for inference and second
            for training with Adam.
    """

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, None, depth], name='input')
    y = tf.placeholder(tf.float32, [None, num_units], name='y')
    cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units)

    out, final_state = cell(x, dtype=tf.float32)
    loss = tf.reduce_mean(tf.abs(y - final_state[0]))
    train_step = tf.train.AdamOptimizer().minimize(loss)

    inference_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking inference...'):
            start_ts = perf_counter()
            sess.run(out, {x: np.random.randn(time_steps, batch_size, depth)})
            end_ts = perf_counter()
            if i > hot_start_after:
                inference_time.append(end_ts - start_ts)

    train_time = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(num_iter), disable=not verbose,
                      desc='Benchmarking train...'):
            start_ts = perf_counter()
            sess.run(train_step,
                     {x: np.random.randn(time_steps, batch_size, depth),
                      y: np.random.randn(batch_size, num_units)})
            end_ts = perf_counter()
            if i > hot_start_after:
                train_time.append(end_ts - start_ts)

    return inference_time, train_time
