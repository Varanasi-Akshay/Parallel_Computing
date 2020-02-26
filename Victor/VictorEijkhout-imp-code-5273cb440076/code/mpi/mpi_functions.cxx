/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** mpi_functions.cxx : implementations of the support functions
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "math.h"

#include "mpi_base.h"
#include "imp_functions.h"

//snippet crudecopympi
/*!
  A very crude copy, completely ignoring distributions.
 */
void crudecopy( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  double *indata = invector->get_data(p);
  double *outdata = outvector->get_data(p);
  
  int ortho = 1;
  if (ctx!=nullptr) ortho = *(int*)ctx;

  index_int n = outvector->volume(p);

  //  printf("[%d] veccopy of size %ld: %e\n",p,n*ortho,indata[0]);
  for (index_int i=0; i<n*ortho; i++) {
    outdata[i] = indata[i];
  }
  *flopcount += n*ortho;
}
//snippet end

/*!
  Sum two vectors.
*/
void vectorsum( kernel_function_args )
{
  auto invector = invectors.at(0);
  double *indata1 = invector->get_data(p);
  auto invector2 = invectors.at(1);
  double *indata2 = invector2->get_data(p);
  double *outdata = outvector->get_data(p);

  index_int n = invector->volume(p);

  for (index_int i=0; i<n; i++) {
    outdata[i] = indata1[i]+indata2[i];
  }
  *flopcount += n;
}

void local_sparse_matrix_vector_multiply( kernel_function_args,void *ctx ) {
  auto invector = invectors.at(0);

  mpi_sparse_matrix *matrix = (mpi_sparse_matrix*)ctx;
  matrix->multiply(invector,outvector,p);
  return;

};

/*!
  An operation between replicated scalars. The actual operation
  is supplied as a character in the context.
  - '/' : division
 */
void char_scalar_op( kernel_function_args, void *ctx )
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
};

#if 0
/*
 * Nbody stuff
 */
/*!
  Compute the center of mass of an array of particles by comparing two-and-two

  - k=1: add charges
  - k=2: 0=charges added, 1=new center
 */
void scansumk( kernel_function_args,int k )
{
  auto invector = invectors.at(0);
  double *indata = invector->get_data(p);
  int insize = invector->volume(p);

  double *outdata = outvector->get_data(p);
  int outsize = outvector->volume(p);

  if (k<1)
    throw(fmt::format("Illegal scansum value k={}",k));
  if (k>2)
    throw(fmt::format("Unimplemented scansum value k={}",k));

  if (2*outsize!=insize) {
    printf("scansum: in/out not compatible: %d %d\n",insize,outsize); throw(6);}

  // fmt::print("scansum on {}: {} elements, sum starts with {}\n",
  // 	     p->coord(0),outsize,outdata[0]);
  double cm;
  //printf("[%d] step %d",p->coord(0),step);
  for (index_int i=0; i<outsize; i++) {
    double v1 = indata[2*i*k], v2 = indata[(2*i+1)*k];
    //printf(", scan %e %e",v1,v2);
    outdata[k*i] = v1+v2;
    if (k==2) {
      double x1 = indata[2*i*k+1], x2 = indata[(2*i+1)*k+1];
      outdata[i*k+1] = (x1*v2+x2*v1)/(x1+x2);
    }
  } //printf("\n");
  *flopcount += k*outsize;
}

//! Short-cut of \ref scansumk for k=1
void scansum( kernel_function_args ) {
  scansumk(step,p,invectors,outvector,1,flopcount);
}
#endif

void scanexpand( kernel_function_args )
{
  auto invector = invectors.at(0);

  double
    *outdata = outvector->get_data(p), *indata = invector->get_data(p);
  index_int
    outsize = outvector->volume(p),   insize = invector->volume(p),
    outfirst = outvector->first_index_r(p).coord(0),
    infirst = invector->first_index_r(p).coord(0),
    outlast = outvector->last_index_r(p).coord(0),
    inlast = invector->last_index_r(p).coord(0);

  if (outfirst/2<infirst || outfirst/2>inlast || outlast/2<infirst || outlast/2>outlast) {
    fmt::MemoryWriter w;
    w.write("[{}] scanexpand: outrange [{}-{}] not sourced from [{}-{}]",
	    p.as_string(),outfirst,outlast,infirst,inlast);
    throw(w.str().data());
  }

  for (index_int i=0; i<outsize; i++) {
    index_int src = (outfirst+i)/2;
    outdata[i] = indata[src-infirst];
    //outdata[2*i] = indata[i]; outdata[2*i+1] = indata[i];
  }
}

/*
 * CG stuff
 */
// void central_difference
//     (int step,processor_coordinate &p,std::vector<object*> *invectors,auto outvector,
//      double *flopcount)
// {
//   central_difference_damp(step,p,invectors,outvector,flopcount,1.);
// }

// //! \todo turn void* into double
// void central_difference_damp
//     (int step,processor_coordinate &p,std::vector<object*> *invectors,auto outvector,
//      double *flopcount,double damp)
// {
//   auto invector = invectors.at(0);
//   double *outdata = outvector->get_data(p), *indata = invector->get_data(p);

//   // figure out where the current subvector fits in the global numbering
//   index_int
//     tar0 = outvector->first_index_r(p).coord(0)-outvector->numa_first_index().coord(0),
//     src0 = outvector->first_index_r(p).coord(0)-invector->numa_first_index().coord(0),
//     len = outvector->volume(p);
//   index_int 
//     myfirst = outvector->first_index_r(p).coord(0),
//     mylast = outvector->last_index(p).coord(0),
//     glast = outvector->global_volume()-1;
  
//   // setting the boundaries is somewhat tricky
//   index_int lo=0,hi=len;
//   if (myfirst==0) { // dirichlet boundary condition
//     outdata[tar0] = ( 2*indata[src0] - indata[src0+1] )*damp;
//     *flopcount += 2;
//     lo++;
//   }
//   if (mylast==glast) {
//     outdata[tar0+len-1] = ( 2*indata[src0+len-1] - indata[src0+len-2] )*damp;
//     *flopcount += 2;
//     hi--;
//   }

//   // ... but then we have a regular three-point stencil
//   for (index_int i=lo; i<hi; i++) {
//     outdata[tar0+i] = ( 2*indata[src0+i] - indata[src0+i-1] - indata[src0+i+1] )*damp;
//   }

//   *flopcount += 3*(hi-lo+1);
// }

//! \todo this sorely needs to be made independent
void local_diffusion( kernel_function_args )
{
  auto invector = invectors.at(0);
  double *outdata = outvector->get_data(p), *indata = invector->get_data(p);

  index_int
    tar0 = 0, //outvector->first_index_r(p)-outvector->global_first_index(),
    src0 = 1, //outvector->first_index_r(p)-invector->global_first_index(),
    len = outvector->volume(p);
  return;

  for (index_int i=0; i<len; i++)
    outdata[tar0+i] = 2*indata[src0+i] - indata[src0+i-1] - indata[src0+i+1];
  *flopcount += 3*len;
}

