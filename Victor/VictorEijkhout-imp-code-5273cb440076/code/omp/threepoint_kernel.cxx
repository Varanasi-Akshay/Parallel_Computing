#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#include "omp_base.h"
#include "threepoint_kernel.h"
#include "threepoint_local.h"

/****
 **** Kernel for initial data generation
 ****/
void gen_kernel_execute
(int step,int p,object *inv,object *outv) {
  omp_object *outvector = (omp_object*)outv;
  omp_distribution *outdistro = (omp_distribution*)outvector->distro;
  int first=outdistro->first_index(p),last=outdistro->last_index(p);
  double *outdata = outvector->get_data(p);

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
void threepoint_execute
(int step,int p,object *inv,object *outv) {
  omp_object *outvector = (omp_object*)outv;
  omp_distribution *outdistro = (omp_distribution*)outvector->distro;
  int first=outdistro->first_index(p),last=outdistro->last_index(p);
  double *outdata = outvector->get_data(p);

  int globalsize = outvector->distro->global_size();
  if (step==0) {
    printf("Step 0 has separate function");
    throw(0);
  } else {
    double nops;
    omp_object *invector = (omp_object*)inv;
    double *indata = invector->get_data(p);
    avg_local(indata,outdata,first,last,globalsize,&nops);
    outdistro->get_environment()->register_flops(nops);
  }
};
