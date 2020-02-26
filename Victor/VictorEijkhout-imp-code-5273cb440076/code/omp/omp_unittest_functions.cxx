/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** OpenMP implementations of the unittest support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "unittest_functions.h"

void vecset( kernel_function_args )
{
  double *outdata = outvector->get_data(p);
  //  distribution *outdistro = dynamic_cast<distribution*>(outvector);
  index_int
    my_first = outvector->first_index_r(p).coord(0),
    my_last = outvector->last_index_r(p).coord(0);

  index_int n = outvector->volume(p);

  for (index_int i=my_first; i<=my_last; i++) {
    outdata[i] = 1.;
  }
  *flopcount += n;
}

//snippet ompshiftleft
/*!
  The src0/tar0 indices were necessitated when this was run in a hybrid environment.
 */
void vecshiftleftmodulo( kernel_function_args ) {
  auto invector = invectors.at(0);
  // distribution
  //   *indistro = dynamic_cast<distribution*>(invector),
  //   *outdistro = dynamic_cast<distribution*>(outvector);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);

  for (index_int i=0; i<len; i++) {
      outdata[tar0+i] = indata[src0+i+1];
  }
  *flopcount += len;
}
//snippet end

#if 0
void vecshiftleftbump( kernel_function_args ) {
  auto invector = invectors.at(0);
  // distribution
  //   *indistro = dynamic_cast<distribution*>(invector),
  //   *outdistro = dynamic_cast<distribution*>(outvector);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);

  index_int
    gsize = outvector->global_volume(),
    ingsize = invector->global_volume();

  if (tar0+len-1>gsize-1) // omit elements that don't exist
    len--;
  for (index_int i=0; i<len; i++) {
      outdata[tar0+i] = indata[src0+i+1];
  }
  *flopcount += len;
}
#endif

void vecshiftrightmodulo( kernel_function_args )
{
  auto invector = invectors.at(0);
  // distribution
  //   *indistro = dynamic_cast<distribution*>(invector),
  //   *outdistro = dynamic_cast<distribution*>(outvector);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);
  index_int
    gsize = outvector->global_volume(),
    ingsize = invector->global_volume();

  // int lo = 0;
  // if (outvector->first_index_r(p).coord(0)==0) lo++;
  for (index_int i=0; i<len; i++)
    outdata[tar0+i] = indata[src0+i-1];
  *flopcount += len;
}

void ksumming( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  int *k = (int*)ctx; // printf("with k=%d\n",*k);
  double *outdata = outvector->get_data(p);
  double *indata = invector->get_data(p);
  index_int
    n = invector->volume(p),
    f = invector->first_index_r(p).coord(0);
  for (int ik=0; ik<(*k); ik++) {
    double s = 0;
    for (index_int i=f; i<f+n; i++) {
      double in = indata[ik+i*(*k)];
      s += in;
    } 
    outdata[ik] = s;
  }
  *flopcount += (*k)*n;
}
#if 0
/*!
  Three point averaging with bump connections:
  the halo has the same size as the alpha domain
  \todo this ignores non-zero first index and such.
*/ 
void threepointsumbump( kernel_function_args,void *ctx )
{
  if (outvector->get_dimensionality()>1)
    throw(std::string("threepointsumbump: only for 1-d"));

  auto invector = invectors.at(0);
  // distribution
  //   *indistro = dynamic_cast<distribution*>(invector),
  //   *outdistro = dynamic_cast<distribution*>(outvector);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    mfirst = outvector->first_index_r(p).coord(0),
    gfirst = outvector->global_first_index(0),
    mlast = outvector->last_index_r(p).coord(0),
    glast = outvector->global_last_index(0),
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->get_numa_structure()
            ->linear_location_of(outvector->get_processor_structure(p)),
    //    src0 = invector->linearize(outvector->first_index_r(p)),
    len = outvector->volume(p);

  index_int
    gsize = outvector->global_volume(),
    ingsize = invector->global_volume();

  //snippet threepointsumbumpomp
  int ilo = 0, ilen = len;
  if (mfirst==gfirst) {
    // first element is globally first: the invector does not stick out to the left
    int i = 0; 
    outdata[tar0+i] = indata[src0+i] + indata[src0+i+1];
    ilo++;
  }
  if (mlast==glast) {
     // local last is globally last; just compute and lower the length
    int i = ilen-1;
    outdata[tar0+i] = indata[src0+i-1] + indata[src0+i];
    ilen--;
  }
  for (index_int i=ilo; i<ilen; i++) {
    // regular case: the invector sticks out one to the left, so we shift.
    outdata[tar0+i] = indata[src0+i-1]+indata[src0+i]+indata[src0+i+1];
  }
  //snippet end
  *flopcount += 2*len;
}
#endif
/*!
  Three point averaging with modulo connections:
  the halo has a point left of 0 and right of N-1
*/ 
void threepointsummod( kernel_function_args )
{
  auto invector = invectors.at(0);
  // distribution
  //   *indistro = dynamic_cast<distribution*>(invector),
  //   *outdistro = dynamic_cast<distribution*>(outvector);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->linearize( outvector->first_index_r(p) ), //location_of_first_index(p),
    len = outvector->volume(p);

  index_int
    gsize = outvector->global_volume(),
    ingsize = invector->global_volume();

  printf("summod %d <- %d, #elts=%d\n",tar0,src0,len);
  //snippet threepointsummodomp
  for (index_int i=0; i<len; i++)
    outdata[tar0+i] = indata[src0+i-1]+indata[src0+i]+indata[src0+i+1];
  //snippet end
  *flopcount += 2*len;
}

/*!
  Test if the local execute function sees the right global size:
  the desired size is coming in as context
*/
void test_globalsize( kernel_function_args,index_int test_globalsize )
{
  auto invector = invectors.at(0);
  index_int is_globalsize=outvector->global_volume();
  if (is_globalsize!=test_globalsize)
    throw(fmt::format("global is {}, s/b {}",is_globalsize,test_globalsize));
}

/*!
  Test if the local execute function sees the number of processors;
  the desired number is coming in as context
*/
void test_nprocs( kernel_function_args,int test_nprocs )
{
  auto invector = invectors.at(0);
  int is_nprocs;

  is_nprocs=invector->domains_volume();
  if (is_nprocs!=test_nprocs) {
    throw(fmt::format("Incorrect nprocs tested on input: {} s/b {}",is_nprocs,test_nprocs));
  }

  is_nprocs=outvector->domains_volume();
  if (is_nprocs!=test_nprocs) {
    throw(fmt::format("Incorrect nprocs tested on output: {} s/b {}",is_nprocs,test_nprocs));
  }

}

void test_distr_nprocs( kernel_function_args,int test_nprocs )
{
  auto invector = invectors.at(0);
  int is_nprocs;

  is_nprocs=outvector->domains_volume();
  if (is_nprocs!=test_nprocs)
    throw(fmt::format("Incorrect distribution nprocs tested on output: {} s/b {}",
		      is_nprocs,test_nprocs));

  is_nprocs=invector->domains_volume();
  if (is_nprocs!=test_nprocs)
    throw(fmt::format("Incorrect distribution nprocs tested on input: {} s/b {}",
		      is_nprocs,test_nprocs));

}

/*
 * Auxiliary stuff
 */
int pointfunc33(int i,int my_first) {return my_first+i;}
