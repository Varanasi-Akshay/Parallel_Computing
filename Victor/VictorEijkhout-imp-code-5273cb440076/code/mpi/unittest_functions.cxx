/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** MPI implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>

#include "imp_base.h"
#include "unittest_functions.h"

//! \todo eliminate mention of distribution in these unittest functions
void vecset( kernel_function_args )
{
  double *outdata;
  try { outdata = outvector->get_data(p);
  } catch (std::string c) { fmt::print("Error <<{}>> in vecset\n",c);
    throw(fmt::format("vecset of object <<{}>> failed",outvector->get_name())); };

  index_int n = outvector->volume(p);

  for (index_int i=0; i<n; i++) {
    outdata[i] = 1.;
  }
  *flopcount += n;
}

//snippet mpishiftleft
void vecshiftleftmodulo( kernel_function_args )
{
  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->first_index_r(p).coord(0)-outvector->numa_first_index().coord(0),
    src0 = invector->first_index_r(p).coord(0)-invector->numa_first_index().coord(0),
    len = outdistro->volume(p);

  for (index_int i=0; i<len; i++) {
      outdata[tar0+i] = indata[src0+i+1];
  }
  *flopcount += len;
}
//snippet end

void vecshiftrightmodulo( kernel_function_args )
{
  auto invector = invectors.at(0);
  double *indata = invector->get_data(p);
  double *outdata = outvector->get_data(p);

  index_int n = outvector->volume(p); // the halo is one more

  for (index_int i=0; i<n; i++) {
    outdata[i] = indata[i];
  }
  *flopcount += n;
}

//! \todo make the void an explicit integer
void ksumming( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  int *k = (int*)ctx;
  double *outdata = outvector->get_data(p);
  double *indata = invector->get_data(p);
  index_int n = invector->volume(p);
  for (int ik=0; ik<(*k); ik++) {
    double s = 0;
    for (index_int i=0; i<n; i++) {
      double in = indata[ik+i*(*k)];
      s += in;
    } 
    outdata[ik] = s;
  }
  *flopcount += (*k)*n;
}

void threepointsummod( kernel_function_args )
{
  auto invector = invectors.at(0);
  double *outdata = outvector->get_data(p);
  double *indata = invector->get_data(p);
  index_int n = outvector->volume(p);
  double s = 0;
//snippet mpi3pmod
  for (index_int i=0; i<n; i++)
    outdata[i] = indata[i]+indata[i+1]+indata[i+2];
//snippet end
  *flopcount += n;
}

/*
 * Auxiliary stuff
 */
int pointfunc33(int i,int my_first) {return my_first+i;}
