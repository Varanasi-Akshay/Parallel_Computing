/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** kmeans_local.h : 
 ****     prototypes for the kmeans application
 ****
 ****************************************************************/

#ifndef KMEANS_LOCAL_H
#define KMEANS_LOCAL_H

#include "utils.h"

void gen_local(double *outdata,index_int first,index_int last);
void avg_local(double *indata,double *outdata,index_int first,index_int last,double *nops);
void vec_sum(double *outdata,double *indata,index_int first,index_int last);

#endif
