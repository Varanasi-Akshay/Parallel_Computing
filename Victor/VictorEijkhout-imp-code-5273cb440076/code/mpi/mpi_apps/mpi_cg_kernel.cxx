/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** mpi_cg_kernel.cxx : 
 ****     local functions for the cg application
 ****
 ****************************************************************/

#include "mpi_base.h"
#include "cg_kernel.h"

// moved to mpi_functions.cxx
#if 0
void central_difference
    (int step,processor_coordinate *p,std::vector<object*> *invectors,object *outvector,void *ctx,double *flopcount)
{
  object *invector = invectors->at(0);
  double *outdata = outvector->get_data(p), *indata = invector->get_data(p);

  // figure out where the current subvector fits in the global numbering
  index_int
    tar0 = outvector->first_index(p).coord(0)-outvector->numa_first_index(),
    src0 = outvector->first_index(p).coord(0)-invector->numa_first_index(),
    len = outvector->local_size(p);

  // setting the boundaries is somewhat tricky
  index_int lo=0,hi=len;
  if (src0==0) lo++; if (src0==invector->local_size(p)) hi--;

  // ... but then we have a regular three-point stencil
  for (index_int i=lo; i<hi; i++)
    outdata[tar0+i] = 2*indata[src0+i] - indata[src0+i-1] - indata[src0+i+1];
  *flopcount += 3*(hi-lo+1);
}
#endif

