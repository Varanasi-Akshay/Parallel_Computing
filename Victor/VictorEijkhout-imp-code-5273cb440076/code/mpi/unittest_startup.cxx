/*
 * Unit tests for the MPI backend of IMP
 * based on the CATCH framework (https://github.com/philsquared/Catch)
 */
#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include <mpi.h>

#include "mpi_static_vars.h"

#include "mpi_base.cxx"

template <typename MsgType>
int object<MsgType>::count = 0;

void unittest_mpi_setup(int argc,char **argv) {
  fprintf(stderr,"starting up\n");
  try {
    env = new mpi_environment(argc,argv); }
  catch (int x) {
    printf("Could not even get started\n"); throw(1); 
  }
  fprintf(stderr,"created the environment\n");
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);

  return;
}

TEST_CASE( "Environment is proper","[environment][init][0]" ) {

  int tmp;
  CHECK_NOTHROW( tmp = env->nprocs() );
}

