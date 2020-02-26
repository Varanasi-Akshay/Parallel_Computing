/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** mpi_kmeans_kernel.cxx : 
 **** local functions for the kmeans application
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include "math.h"
#include <iostream>
using namespace std;

#include "mpi_base.h"
#include "kmeans_functions.h"

#include "utils.h"

/*
 * Functions for masked reductions,
 * which we use in computing new centers
 *
 * - indata: strictly input
 * - outdata: throughput
 * - len: unused, required by MPI prototype for MPI_Op_create
 * - type: unused, required by MPI prototype for MPI_Op_create
 */
void add_if_mask1( void *indata, void *outdata,int *len,MPI_Datatype *type ) {
  int dim = 1;
  double *incoord = (double*)indata;
  double *outcoord = (double*)outdata;
  // printf("combine in:(%3.1f,%3.1f) with inout:(%3.1f,%3.1f)\n",
  // 	 incoord[0],incoord[1],outcoord[0],outcoord[1]);
  if (incoord[0]>0) {
    if (outcoord[0]>0) {
      for (int id=1; id<=dim; id++) {
	outcoord[id] += incoord[id];
      }
    } else {
      outcoord[0] = 1;
      for (int id=1; id<=dim; id++) {
	outcoord[id] = incoord[id];
      }
    }
  } // if the input is masked, we leave the inout alone.
}

/*
 * Invector:
 * Outvector: replicated size nclusters x orthogonal dim 
 */
void masked_reduction_1d( kernel_function_args ) {
  int dim = outvector->get_orthogonal_dimension();
  index_int nclusters = outvector->volume(p);


  MPI_Datatype dim1_type;
  MPI_Type_contiguous(2,MPI_DOUBLE,&dim1_type);
  MPI_Type_commit(&dim1_type);

  MPI_Op masked_add;
  MPI_Datatype double_type = MPI_DOUBLE;
  MPI_Op_create(add_if_mask1,0,&dim1_type);

  double
    *indata = invectors.at(0)->get_data(p),
    *outdata = outvector->get_data(p);

  MPI_Allreduce(indata,outdata,1,dim1_type,masked_add,MPI_COMM_WORLD);

  MPI_Type_free(&dim1_type);
}
