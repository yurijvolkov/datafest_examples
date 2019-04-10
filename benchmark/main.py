import os
import logging
import pandas as pd
import tensorflow as tf

from itertools import product
from benchmark import benchmark_dynamic_rnn, benchmark_cudnn,\
                      benchmark_fused_rnn

logger = logging.getLogger(__name__)


def add_exp(df, time, time_steps, depth, calc_type, hardware, cell):
    return df.append(pd.DataFrame({'time': time,
                                   'time_steps': time_steps,
                                   'depth': depth,
                                   'calc_type': calc_type,
                                   'hardware': hardware,
                                   'cell': cell}))


def main(hardware):
    params = product([10, 100], [100, 500])
    for (depth, time_steps) in params:
        df_all = pd.DataFrame(columns=['time', 'time_steps', 'depth',
                                       'calc_type', 'hardware', 'cell'])

        # GRUCell
        inference, train = benchmark_dynamic_rnn(tf.nn.rnn_cell.GRUCell(100),
                             batch_size=100,
                             time_steps=time_steps,
                             depth=depth, verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'GRUCell')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'GRUCell')

        # CudnnGRU
        inference, train = benchmark_cudnn('gru', batch_size=100,
                           time_steps=time_steps, depth=depth,
                           verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'CudnnGRU')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'CudnnGRU')

        # GRUBlockCell
        inference, train = benchmark_dynamic_rnn(
                        tf.contrib.rnn.GRUBlockCell(100),
                        batch_size=100,
                        time_steps=time_steps,
                        depth=depth, verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'GRUBlockCell')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'GRUBlockCell')

        # LSTMCell
        inference, train = benchmark_dynamic_rnn(
                        tf.nn.rnn_cell.LSTMCell(100),
                        batch_size=100,
                        time_steps=time_steps,
                        depth=depth, verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'LSTMCell')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'LSTMCell')

        # LSTMBlockCell
        inference, train = benchmark_dynamic_rnn(
                        tf.contrib.rnn.LSTMBlockCell(100),
                        batch_size=100,
                        time_steps=time_steps,
                        depth=depth, verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'LSTMBlockCell')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'LSTMBlockCell')

        # LSTMBlockFusedCell
        inference, train = benchmark_fused_rnn(
                        tf.contrib.rnn.LSTMBlockFusedCell(100),
                        batch_size=100,
                        time_steps=time_steps,
                        depth=depth, verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'LSTMBlockFusedCell')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'LSTMBlockFusedCell')

        # CudnnLSTM
        inference, train = benchmark_cudnn('lstm', batch_size=100,
                           time_steps=time_steps, depth=depth,
                           verbose=True)
        df_all = add_exp(df_all, inference, time_steps, depth,
                 'inference', hardware, 'CudnnLSTM')
        df_all = add_exp(df_all, train, time_steps, depth,
                 'train', hardware, 'CudnnLSTM')

    df_all.to_csv(f'{hardware}_profile.csv')


if __name__ == "__main__":
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError('CUDA_VISIBLE_DEVICES wasn\'t set')
    if os.environ['CUDA_VISIBLE_DEVICES'] != '':
        hardware = 'gpu'
    else:
        hardware = 'cpu'
    logger.info(f'Hardware: {hardware}.')

    main(hardware)
