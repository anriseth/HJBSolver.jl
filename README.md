# HJBSolver

[![Build Status](https://travis-ci.org/anriseth/HJBSolver.jl.svg?branch=master)](https://travis-ci.org/anriseth/HJBSolver.jl)
[![Coverage Status](https://coveralls.io/repos/anriseth/HJBSolver.jl/badge.svg)](https://coveralls.io/r/anriseth/HJBSolver.jl)
[![codecov](https://codecov.io/gh/anriseth/HJBSolver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/anriseth/HJBSolver.jl)

## General solver for Hamilton-Jacobi-Bellman equations.
Solve one-dimensional HJB equations of the form
``` tex
v_t + \sup_{a\in A}\{ b(t,x,a)*v_x + \frac{1}{2}\sigma(t,x,a)^2v_{xx} + f(t,x,a)\}= 0
```

HJBSolver implements two Finite Difference solvers based on the algorithms described
in `forsyth2007numerical`:
- *Policy iteration*: Run a local optimisation for the policy on each time-step.
- *Piecewise constant policy timestepping*: Approximate the policy function from
  a discrete set of values.


## TODO:
- Show how to use it
- Warn about crappy code


## Citations
```
@article{forsyth2007numerical,
  title={Numerical methods for controlled Hamilton-Jacobi-Bellman PDEs in finance},
  author={Forsyth, Peter A and Labahn, George},
  journal={Journal of Computational Finance},
  volume={11},
  number={2},
  pages={1},
  year={2007}
}
```
