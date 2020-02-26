/*
 * Unit tests for the MPI version of the `kmeans' program
 * based on the CATCH framework (https://github.com/philsquared/Catch)
 */
#include <stdlib.h>
#include <math.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#define MPI_VARS_HERE
#include "mpi_static_vars.h"

#include "mpi_base.h"

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

void unittest_kmeans_setup();

int main(int argc,char **argv) {

  unittest_mpi_setup(argc,argv); // this writes file variables in the unittest file

  unittest_kmeans_setup();

  int result = Catch::Session().run( argc, argv );

  return 0;
}
