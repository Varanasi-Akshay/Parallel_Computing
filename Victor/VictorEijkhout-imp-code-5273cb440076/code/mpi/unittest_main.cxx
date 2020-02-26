/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** Main for the MPI backend tests
 ****
 ****************************************************************/

#include <stdlib.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "fmt/format.h"

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define mpi_VARS_HERE
#include "mpi_static_vars.h"

void unittest_mpi_setup(int argc,char **argv) {

  try {
    env = new mpi_environment(argc,argv);
  }
  catch (int x) {
    printf("Could not even get started\n"); throw(1); 
  }

  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);

  arch = dynamic_cast<mpi_architecture*>(env->get_architecture());
  mycoord = processor_coordinate(1);
  mycoord.set(0,mytid); // 1-d by default
  //fmt::print("MPI proc {} at coord {}\n",mytid,mycoord->as_string());

  try {
    decomp = new mpi_decomposition(arch);
  } catch (std::string c) {
    fmt::print("Error <<{}>> while making top mpi decomposition\n",c);
  } catch (...) {
    fmt::print("Unknown error while making top mpi decomposition\n");
  }
  try {
    int gd = decomp->domains_volume();
  } catch (std::string c) {
    fmt::print("Error <<{}>> in mpi setup\n",c);
    throw(1);
  }

  return;
}

#include "mpi_base.h"
#include "mpi_ops.h"

bool algorithm::do_optimize = false;
int algorithm::queue_trace_summary = 0;

int entity_name::entity_number = 0;
trace_level entity::tracing = trace_level::NONE;
std::vector<entity*> environment::list_of_all_entities;
std::function< void(void) > environment::print_application_options{
  [] (void) -> void { return; } };
environment *entity::env = nullptr;
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) > kernel::make_reduction_kernel{
  [] ( std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* {
    return new mpi_reduction_kernel(vec,scal); } };
int object::count = 0;
double object_data::create_data_count = 0.;
bool object_data::trace_create_data = false;
int sparse_matrix::sparse_matrix_trace = 0;
int task::count = 0;

int main(int argc,char **argv) {

  unittest_mpi_setup(argc,argv);

  for (int a=0; a<argc; a++)
    if (!strcmp(argv[a],"--imp")) { argc = a; break; }

  int result = Catch::Session().run( argc, argv );

  delete env;
  printf("disabled finalize test!\n");
  // {
  //   int flg;
  //   MPI_Finalized(&flg);
  //   if (!flg) { printf("No finalize from process %d\n",mytid); throw(1); }
  // }
  return result;
}
