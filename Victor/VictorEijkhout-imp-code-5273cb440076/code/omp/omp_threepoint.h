// -*- c++ -*-
#if defined(OMP_THREEPOINT_H)
#else
#define OMP_THREEPOINT_H 1

#include "omp_base.h"

/*
 * Environment customized for the threepoint problem
 */
class threepoint_environment : public omp_environment {
 private:
 protected:
 public:
  threepoint_environment(int argc,char **argv) : omp_environment(argc,argv) {
  };
};

#endif
