/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** OpenMP implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "omp_base.h"
#include "imp_functions.h"

/*!
  Sum two vectors.
  \todo we really need to test equality of distributions
*/
void vectorsum( kernel_function_args )
{
  auto invector1 = invectors.at(0);
  auto invector2 = invectors.at(1);
  double *indata1 = invector1->get_data(p);
  double *indata2 = invector2->get_data(p);
  double *outdata = outvector->get_data(p);

  index_int
    f = outvector->first_index_r(p).coord(0),
    l = outvector->last_index_r(p).coord(0);

  for (index_int i=f; i<=l; i++) {
    outdata[i] = indata1[i]+indata2[i];
  }
  *flopcount += l-f+1;
}

//! \todo we need to figure out how to set a flop count here
void local_sparse_matrix_vector_multiply( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);

  omp_sparse_matrix *matrix = (omp_sparse_matrix*)ctx;
  matrix->multiply(invector,outvector,p);
};

/*!
  An operation between replicated scalars. The actual operation
  is supplied as a character in the context.
 */
void char_scalar_op( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  char_object_struct *chobst = (char_object_struct*)ctx;
  //printf("scalar operation <<%s>>\n",chobst->op);

  double
    *arg1 = invector->get_data(p),
    *arg2 = chobst->obj->get_data(p),
    *out  = outvector->get_data(p);
  if (!strcmp(chobst->op,"/")) {
    *out = arg1[0] / arg2[0];
  } else
    throw("Unrecognized scalar operation\n");
  *flopcount += 2;
};

/*
 * Nbody stuff
 */
#if 0
void scansum( kernel_function_args )
{
  auto invector = invectors.at(0);
  double *indata = invector->get_data(p);
  int insize = invector->volume(p);

  double *outdata = outvector->get_data(p);
  index_int outsize = outvector->volume(p),
    first = outvector->first_index(p).coord(0);

  if (2*outsize!=insize) {
    printf("scansum: in/out not compatible: %d %d\n",insize,outsize); throw(6);}

  for (index_int i=first; i<+first+outsize; i++) {
    outdata[i] = indata[2*i]+indata[2*i+1];
  }
  *flopcount += outsize;
}
#endif

/*!
  Expand an array by 2. Straight copy and doubling.
*/
void scanexpand( kernel_function_args )
{
  auto invector = invectors.at(0);

  double
    *outdata = outvector->get_data(p), *indata = invector->get_data(p);
  index_int
    outsize  = outvector->volume(p),   insize = invector->volume(p),
    outfirst = outvector->first_index_r(p).coord(0), infirst = invector->first_index_r(p).coord(0),
    outlast  = outvector->last_index_r(p).coord(0),   inlast = invector->last_index_r(p).coord(0);

  if (outfirst/2<infirst || outfirst/2>inlast || outlast/2<infirst || outlast/2>outlast)
    throw(fmt::format("[{}] scanexpand: outrange [{}-{}] not sourced from [{}-{}]",
		      p.as_string(),outfirst,outlast,infirst,inlast));

  for (index_int i=0; i<outsize; i++) {
    outdata[outfirst+i] = indata[infirst+i/2];
  }
  *flopcount += outsize;
}
