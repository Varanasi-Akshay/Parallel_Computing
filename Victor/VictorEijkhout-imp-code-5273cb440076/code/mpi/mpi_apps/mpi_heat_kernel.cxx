/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** mpi_heat_kernel.cxx : MPI local function for heat diffusion
 ****
 ****************************************************************/

#include "mpi_base.h"

void local_diffusion
(int step,processor_coordinate *p,std::vector<object*> *invectors,object *outvector,void *ctx,double *flopcount)
{
  object *invector = invectors->at(0);
  double *outdata = outvector->get_data(p), *indata = invector->get_data(p);

  // figure out where the current subvector fits in the global numbering
  index_int
    tar0 = outvector->location_of_first_index(p),
    src0 = outvector->location_of_first_index(p),
    len = outvector->local_size(p);

  // setting the boundaries is somewhat tricky
  index_int lo=0,hi=len;
  if (src0==0) lo++; if (src0==invector->local_size(p)) hi--;

  // ... but then we have a regular three-point stencil
  for (index_int i=lo; i<hi; i++)
    outdata[tar0+i] = 2*indata[src0+i] - indata[src0+i-1] - indata[src0+i+1];
  *flopcount += 3*(hi-lo+1);
}
