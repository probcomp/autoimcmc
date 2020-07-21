# Automating Involutive MCMC using Probabilistic and Differentiable Programming

Cusumano-Towner, Marco F. and Alexander K. Lew and Vikash K. Mansinghka. "Automating Involutive MCMC using Probabilistic and Differentiable Programming." arXiv preprint arXiv:2007.09871 (2020) [PDF](https://arxiv.org/pdf/2007.09871.pdf)

The paper describes the automated involutive MCMC construct as implemented in the [**Gen probabilistic programming system**](https://www.gen.dev).

NOTE: The code in the paper uses a modified syntax that has not yet been merged into the master branch of Gen. The previous syntax is described as part of the current [Gen involution MCMC documentation](https://www.gen.dev/dev/ref/mcmc/#Involution-MCMC-1). The new Gen syntax, which matches that in the paper, is currently being reviewed in [this PR](https://github.com/probcomp/Gen.jl/pull/290) and is documented in the new [Involutive MCMC ](https://github.com/probcomp/Gen.jl/blob/20200416-marcoct-translatordsl/docs/src/ref/mcmc.md#involutive-mcmc) documentation, and the new [Trace Translator](https://github.com/probcomp/Gen.jl/blob/20200416-marcoct-translatordsl/docs/src/ref/trace_translators.md#trace-translators) documentation. 

## Examples using Gen
The Gen implementation of the techniques described in the paper is in the Gen repository, and not in this repository.
This repository contains example code that uses the Gen implementation of automated involutive MCMC for a split-merge reversible jump move in an infinite mixture model (corresponding to Figure 1 of the paper).

To run this example, first [obtain Julia](https://julialang.org/downloads/), and run:
```
cd gen
julia --project=. mixture_example.jl
```

## Minimal PyTorch-based languages and examples
This repository also contains the implementation of a minimal probabilistic programming language and differentiable programming language for transforming traces, on top of PyTorch, and a minimal example of involutive MCMC written using that framework. The PyTorch implementation does not implement as many optimizations as the Gen implementation, and only supports involutive MCMC (whereas Gen provides an API for construting a variety of inference algorithms), but is short and self-contained.

To run the examples, first [obtain PyTorch](https://pytorch.org/), ensure it is installed in your python environment and run:
```
cd pytorch
python polar_cartesian_example.py
python mixture_example.py
```
