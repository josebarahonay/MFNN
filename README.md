# Multi-fidelity deep neural network
An implementation of the multi-fidelity deep neural network architecture proposed in ["A composite neural network that learns from multi-fidelity data: Application to function approximation and inverse PDE problems".](https://doi.org/10.1016/j.jcp.2019.109020)

Files:
- `mdfl.py`: Implementation of a single- and multi-fidelity architectures.
- `multifidelity_benchmark.ipynb`: some of the benchmark problems presented in the paper (linear and non-linear correlations), I forgot to add the plot legends but you can see that the predictions are good :)

Notes:
- It is written in PyTorch.
- I didn't include the MPINN part.
- The cost function is slightly modified, leaving a separate regularization for each network:

  $\mathcal{L} = MSE_{y_{L}} + MSE_{y_{H}} + \lambda_{L} \Sigma{\theta_L^2} + \lambda_{H1} \Sigma{\theta_{H1}^2} + \lambda_{H2} \Sigma{\theta_{H2}^2}$
  
- In addition to 'Tanh' I also included 'ReLU'.
- I tried to extend it to Bayesian network (see the notebook) but it is too expensive.
