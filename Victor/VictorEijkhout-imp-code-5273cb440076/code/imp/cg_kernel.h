/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** cg_kernel.h : headers for the cg local functions
 ****
 ****************************************************************/

#ifndef CG_KERNEL_H
#define CG_KERNEL_H

void central_difference(int,processor_coordinate*,std::vector<object*>*,object*,double*);

#endif
