# Code for 'Automating Involutive MCMC using Probabilistic and Differentiable Programming'

TODO: put arxiv link and info here

The paper describes the automated involutive MCMC construct that was originally implemented in the [**Gen probabilistic programming system**](https://www.gen.dev). The code figures in the paper use Gen syntax.

## Automated involutive MCMC example using Gen
The implementation of the techniques described in the paper is in the Gen repository, and not in this repository.
This repository contains example code that uses the Gen implementation of automated involutive MCMC for a split-merge reversible jump move in an infinite mixture model (corresponding to Figure 1 of the paper). To run this example, use:
```
cd gen
julia --project=. mixture.jl
```

## Minimal PyTorch-based languages and automated involutive MCMC example
This repository also contains the implementation of a minimal probabilistic programming language and differentiable programming language for transforming traces, on top of PyTorch, and a minimal example of involutive MCMC written using that framework. The PyTorch implementation does not implement as many optimizations as the Gen implementation, and only supports involutive MCMC (whereas Gen provides an API for construting a variety of inference algorithms), but is short and self-contained. To run the example, ensure pytorch is installed in the python environment and run:
```
cd pytorch
python example.py
```
