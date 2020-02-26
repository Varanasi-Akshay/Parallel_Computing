// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** product_environment.h: headers for defining an MPI/OMP product environment
 ****
 ****************************************************************/

#ifndef PRODUCT_ENV_H
#define PRODUCT_ENV_H 1

#include "imp_base.h"
#include "omp_environment.h"
#include "mpi_environment.h"

class product_architecture;
class product_environment : public mpi_environment {
private:
public:
  product_environment(int argc,char **argv);
};

#endif
