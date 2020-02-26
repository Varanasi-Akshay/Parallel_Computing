#if defined(MPI_THREEPOINT_H)
#else
#define MPI_THREEPOINT_H 1

#include "mpi_base.h"

/*
 * Environment customized for the threepoint problem
 */
class threepoint_environment : public mpi_environment {
 private:
 protected:
 public:
 threepoint_environment(int argc,char **argv) : mpi_environment(argc,argv) {
  };
};

#endif
