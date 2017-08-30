#!/bin/bash -x

time python sde_eigen.py 0 &
time python sde_eigen.py 1 &
wait
time python sde_eigen.py 2 &
time python sde_eigen.py 3 &
wait
time python sde_eigen.py 4 &
time python sde_eigen.py 8 &
wait
time python sde_eigen.py 12 &
time python sde_eigen.py 16 &
wait
