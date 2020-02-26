#!/usr/bin/env python

import matplotlib.pyplot as plt
import re
import sys

# timings on stampede1

procs = [ 1,    2,    4,    8,    16,   32,   64,   128 ]
procs = [ 8*p for p in procs ]
#petsc = [ 1.21, 1.70, 2.22, 2.27, 3.10, 2.56, 2.25, 3.08 ]
#petsc = [ 1.21, 1.70, 2.22, 2.27, 2.31, 2.56, 2.25, 3.08 ] # filter out the blip
# job 8713050:
petsc = [ 0.86, 1.14, 1.32, 1.81, 1.90, 2.40, 2.54, 2.88 ]
imp =   [ 1.52, 1.59, 1.58, 1.61, 1.78, 1.83, 2.15, 2.81 ]

plt.xlabel('processors')
plt.semilogx( procs,petsc, label='PETSc' )

plt.ylabel('solution time (in .1 sec)')
plt.semilogx( procs,imp, label='IMP' )
plt.ylim( (0,3.5) )

plt.legend(loc='upper left')
plt.title('Runtime for 10 iterations CG\nunpreconditioned, 2D stencil')
#plt.show()
plt.savefig("petsccg.pdf")

#8713050
# KSPSolve               1 1.0 8.6369e-02 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 72100  0  0 11  72100  0  0 12    28
# KSPSolve               1 1.0 1.1406e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 53100  0  0 11  53100  0  0 12    42
# KSPSolve               1 1.0 1.3168e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 53100  0  0 11  53100  0  0 12    73
# KSPSolve               1 1.0 1.8139e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 53100  0  0 11  53100  0  0 12   106
# KSPSolve               1 1.0 1.9035e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 54100  0  0 11  54100  0  0 12   202
# KSPSolve               1 1.0 2.3985e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 54100  0  0 11  54100  0  0 12   320
# KSPSolve               1 1.0 2.5483e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 50100  0  0 11  50100  0  0 12   603
# KSPSolve               1 1.0 2.8845e-01 1.0 3.00e+05 1.0 0.0e+00 0.0e+00 3.0e+00 45100  0  0 11  45100  0  0 12  1065
