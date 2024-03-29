# Datafest 2019: Tensorflow RNNs performance and internals 

## Experiments setup

All experiments were done with next setup:
* Nvidia GeForce 1050Ti Mobile
* Intel Core i5 8300H
* Tensorflow 1.13
* Cuda 10.0
* Cuda driver 410.48
* Cudnn 7.3.1

## Examples :
* **compatible** : Shows how to restore and inference(on CPU and GPU) network trained with Cudnn. 
* **benchmark** : Benchmark main Tensorflow RNNs on both GPU and CPU

  To reproduce experiment just set `CUDA_VISIBLE_DEVICES` (e.g. '0' for GPU and '' for CPU) variable and run `main.py`. 
  
  `CUDA_VISIBLE_DEVICES=0 python main.py`.
  
  Benchmarks are done on multiple hyperparameters configurations so you can go for a cup of coffee while it is running.
  After that you can find `cpu_profile.csv` or `gpu_profile.csv` file on your filesystem and visualize it with `benchmark/Plots.ipynb`.
  
  ### Benchmark on GPU
  ![gpu](https://github.com/yurijvolkov/datafest_examples/blob/master/benchmark/gpu_plot.png)
  
  ### Benchmark on CPU
  ![cpu](https://github.com/yurijvolkov/datafest_examples/blob/master/benchmark/cpu_plot.png)
  
* **implementation_difference** : Shows differences of CudnnGRU from platform-independent RNN that tf provides.




## Presentation : 
TO BE UPLOADED

## Video record :
TO BE UPLOADED
