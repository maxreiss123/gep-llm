# *geppy-memory-extension:* Memory extension for geppy to perform transfer learning
* The implementation extends the [geppy framework](https://github.com/ShuhuaGao/geppy) with a language model mechanism


## Abstract
Gene expression programming is an evolutionary optimization algorithm with the potential to generate interpretable and easily implementable equations for regression problems. Despite knowledge gained from previous optimizations being potentially available, the initial candidate solutions are typically generated randomly at the beginning and often only include features or terms based on preliminary user assumptions. This random initial guess, which lacks constraints on the search space, typically results in higher computational costs in the search for an optimal solution. Meanwhile, transfer learning, a technique to reuse parts of trained models, has been successfully applied to neural networks. However, no generalized strategy for its use exists for symbolic regression in the context of evolutionary algorithms. In this work, we propose an approach for integrating transfer learning with gene expression programming applied to symbolic regression. The constructed framework integrates Natural Language Processing techniques to discern correlations and recurring patterns from equations explored during previous optimizations. This integration facilitates the transfer of acquired knowledge from similar tasks to new ones. Through empirical evaluation of the extended framework across a range of univariate problems from an open database and from the field of computational fluid dynamics, our results affirm that initial solutions derived via a transfer learning mechanism enhance the algorithm's convergence rate towards improved solutions.

## What contains this extension?
* Package ``genememory``, which enables the capability to initialize and train a network (written in tensorflow)
* Trainer.py to perform a training of the memory
* Entities like ``GeneLM`` to ensure candidate creation from the language model
* The small training file contains some approx. testing the GEP on cases from https://github.com/omron-sinicx/srsd-benchmark
* An encoder for creating a latent representation of the problem (pointNet) similar to https://github.com/mojivalipour/symbolicgpt


## Installation
You can install it from sources.
1. First download or clone this repository
```bash
git clone https://github.com/maxreiss123/gep-llm.git
```
2. Change into the root directory, i.e., the one containing the *setup.py* file and install the package using *pip*
```bash
cd geppy-llm
pip install .
```
3. Unzip the ``srds_input_embd.tar.xz`` 

## Examples
* An example of the setup is given in geppy_reg.py
* A case setup is demonstrated within the class 01_perform_airfoil


## Requirements
- Python 3.9
- Further requirements are installed during the setup

## Reference
GEP - Ferreira, C.'s monograph: **Ferreira, C. (2006). Gene expression programming: mathematical modeling by an artificial intelligence (Vol. 21). Springer**.

Further Refs.

- [1] Ferreira, C. (2001). Gene Expression Programming: a New Adaptive Algorithm for Solving Problems. Complex Systems, 13.
- [2] Zhong, J., Feng, L., & Ong, Y. S. (2017). Gene expression programming: a survey. IEEE Computational Intelligence Magazine, 12(3), 54-72.
- [3] Gao, S., Sun, C., Xiang, C., Qin, K., & Lee, T. H. (2020). Learning Asynchronous Boolean Networks From Single-Cell Data Using Multiobjective Cooperative Genetic Programming. IEEE Transactions on Cybernetics, 1-15. https://doi.org/10.1109/TCYB.2020.3022430

