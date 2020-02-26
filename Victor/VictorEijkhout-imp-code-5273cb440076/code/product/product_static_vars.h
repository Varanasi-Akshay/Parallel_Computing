/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** Statically defined variables for a product-based run
 ****
 ****************************************************************/

#ifndef PRODUCT_STATIC_VARS_H
#define PRODUCT_STATIC_VARS_H

#include <mpi.h>
#include <omp.h>
#include "product_base.h"

#ifndef EXTERN
#ifdef product_STATIC_VARS_HERE
#define EXTERN
#else
#define EXTERN extern
#endif
#endif

EXTERN int mytid,mpi_nprocs,omp_nprocs,product_nprocs;
EXTERN processor_coordinate mycoord;
EXTERN MPI_Comm comm;
EXTERN product_decomposition *decomp;
EXTERN product_environment *env; 
EXTERN architecture *arch; 

#endif
