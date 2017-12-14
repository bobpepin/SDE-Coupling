# Fast Simulation of Stochastic Differential Equations with Parallel or Reflection Coupling

This includes a C++ Implementation of a standard Euler-Maruyama scheme for SDEs, 
together with a Python interface and a framework for setting up and running numerical 
experiments using ASDF files.

Comes with an Implementation of Parallel and Reflection Coupling of two-timescale diffusions for 
Quadratic and Double-Well potentials.
New dynamics can be added in src/sde.cpp.

See src/run-gradPt.sh for running a simulation of the semigroup P_t.

Requires Eigen C++ library and Python ASDF library from https://github.com/spacetelescope/asdf.
