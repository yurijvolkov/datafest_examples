# Datafest 2019: Tensorflow RNNs performance and internals 

## Examples :
* **compatible** : Shows how to restore and inference(on CPU and GPU) network trained with Cudnn. 
* **benchmark** : Benchmark main Tensorflow RNNs on both GPU and CPU

  To reproduce experiment just set `CUDA_VISIBLE_DEVICES` (e.g. '0' for GPU and '' for CPU) variable and run `main.py`. 
  
  `CUDA_VISIBLE_DEVICES=0 python main.py`.
  
  Benchmarks are done on multiple hyperparameters configurations so you can go for a cup of coffee while it is running.
  After that you can find `cpu_profile.csv` or `gpu_profile.csv` file on your filesystem and visualize it with `benchmark/Plots.ipynb`. 
* **implementation_difference** : Shows differences of CudnnGRU from platform-independent RNN that tf provides.

## Presentation : 
TO BE UPLOADED

## Video record :
TO BE UPLOADED
