#ifndef THREEPOINT_KERNEL_H
#define THREEPOINT_KERNEL_H

// I don't think this file is used anywhere.

/*
  Local kernels. 
  For OMP, the kernel needs to know on what processing element it executes.
  In the MPI case this assignment is unique, so the "p" parameter is a dummy.
*/

// Kernel for initial data generation
void gen_kernel_execute
(int step,int p,std::vector<object*>*,object *outv);

// Kernel for update step
void threepoint_execute
(int step,int p,std::vector<object*>*,object *outvec);

// Kernel for norm calculation
void local_norm_function
(int step,int p,std::vector<object*>*,object *outv);

#endif
