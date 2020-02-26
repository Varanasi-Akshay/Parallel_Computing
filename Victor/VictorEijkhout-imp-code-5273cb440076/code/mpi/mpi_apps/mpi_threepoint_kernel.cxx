/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** mpi_threepoint_kernel.cxx : 
 **** local functions for the threepoint application
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#include "macros.h"
#include "mpi_base.h"
#include "threepoint_kernel.h"
#include "utils.h"

void gen_local(double *outdata,index_int first,index_int last);
void avg_local(double *indata,double *outdata,index_int first,index_int last,double *nops);
void vec_sum(double *outdata,double *indata,index_int first,index_int last);

/****
 **** Kernel for initial data generation
 ****/
#undef __FUNCT__
#define __FUNCT__ "gen_kernel_execute"
void gen_kernel_execute
(int step,int p,std::vector<object*> *invectors,object *outvector,void *ctx) {
  index_int first=outvector->first_index(p),last=outvector->last_index(p);
  double *outdata = outvector->get_data();

  if (step==0) {
    gen_local(outdata,first,last);
  } else {
    printf("gen kernel only for step 0, not %d\n",step);
    throw(10);
  }

};

/****
 **** Kernel for update step
 ****/
#undef __FUNCT__
#define __FUNCT__ "void threepoint_execute"
void threepoint_execute
(int step,int p,std::vector<object*> *invectors,object *outvector,void *ctx) {
  CHKINVEC(invectors);
  object *invector = invectors->at(0);
  index_int first=outvector->first_index(p),last=outvector->last_index(p);
  double *outdata = outvector->get_data();

  if (step==0) {
    throw("Step 0 has separate function");
  } else {
    double nops;
    //    mpi_std::vector<object*> *invectors = (mpi_object*)inv;
    double *indata = invector->get_data();
    avg_local(indata,outdata,first,last,&nops);
    outvector->get_environment()->register_flops(nops);
  }
};

/****
 **** Local norm calculation
 ****/
#undef __FUNCT__
#define __FUNCT__ "void local_norm_function"
void local_norm_function
(int step,int p,std::vector<object*> *invectors,object *outvector,void *ctx) {
  CHKINVEC(invectors);
  object *invector = invectors->at(0);
  index_int first=invector->first_index(p),last=invector->last_index(p);
  double *outdata = outvector->get_data();
  double *indata = invector->get_data();

  vec_sum(outdata,indata,first,last);

};

void gen_local(double *outdata,index_int first,index_int last) {

  for (index_int i=first; i<last; i++) {
    outdata[i-first] = (double)i;
  }
}

void avg_local(double *indata,double *outdata,index_int first,index_int last,double *nops) {
  int leftshift=1;
  // initialization
  for (index_int i=0; i<last-first; i++)
    outdata[i] = 0.;
  // shift 0
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift;
    outdata[i_out] += indata[i_in];
  }
  // shift to the right
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift-1;
    outdata[i_out] += indata[i_in];
  }
  // shift to the left
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift+1;
    outdata[i_out] += indata[i_in];
  }
  *nops = 3.*(last-first);
  return;
}

void vec_sum(double *outdata,double *indata,index_int first,index_int last) {
  *outdata = 0;
  for (index_int i=first; i<last; i++) {
    index_int i_in = i-first;
    *outdata += indata[i_in];
  }
}
