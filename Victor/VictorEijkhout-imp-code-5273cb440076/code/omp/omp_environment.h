// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** omp_environment.h: headers for defining an OpenMP environment
 ****
 ****************************************************************/

#ifndef OMP_ENV_H
#define OMP_ENV_H

#include "imp_base.h"

class omp_architecture;
class omp_environment : public environment {
private:
public:
  omp_environment(int,char**);
  omp_environment( omp_environment& other ) : environment( other ) {
    arch = other.get_architecture(); };
  ~omp_environment();
  virtual architecture *make_architecture(); // pure virtual
  virtual void print_options() override;
  void print_stats();
  void record_task_executed();
};

#endif
