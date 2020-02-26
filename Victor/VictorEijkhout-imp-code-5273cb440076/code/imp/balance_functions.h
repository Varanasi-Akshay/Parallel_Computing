// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** balance_functions.cxx : header file for load balancing support
 ****
 ****************************************************************/

#ifndef BALANCE_FUNCTIONS_H
#define BALANCE_FUNCTIONS_H

#include <memory>

#include "imp_base.h"

distribution *transform_by_average(distribution*,double*);
void setmovingweight( kernel_function_args , int laststep );

#endif
