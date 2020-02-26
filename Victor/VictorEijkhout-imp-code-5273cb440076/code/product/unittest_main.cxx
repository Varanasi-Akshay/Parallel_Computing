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
 **** Main for the MPI+OpenMP backend tests
 ****
 ****************************************************************/

#include <stdlib.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define product_VARS_HERE
#include "product_static_vars.h"

void unittest_product_setup(int argc,char **argv) {
  try {
    env = new product_environment(argc,argv);
  }
  catch (int x) {
    throw("Could not even get started\n");
  }
  mpi_nprocs = env->get_architecture()->nprocs();
  omp_nprocs = env->get_architecture()->get_embedded_architecture()->nprocs();
  fprintf(stderr,
	  "created the environment with %d nodes and %d threads/node\n",
	  mpi_nprocs,omp_nprocs);
  product_nprocs = mpi_nprocs * omp_nprocs;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm,&mytid);

  arch = dynamic_cast<product_architecture*>(env->get_architecture());
  mycoord = processor_coordinate(1); mycoord.set(0,mytid); // 1-d by default
  decomp = new product_decomposition(arch);
  decomp->set_name
    (fmt::format("product decomposition on {}x{}",mpi_nprocs,omp_nprocs));
  fmt::print("Create decomposition <<{}>>\n",decomp->as_string());
  return;
}

#include "product_base.h"
#include "product_ops.h"

// initialization of some class static variables
bool algorithm::do_optimize = false;
int entity_name::entity_number = 0;
trace_level entity::tracing = trace_level::NONE;
std::vector<entity*> environment::list_of_all_entities;
std::function< void(void) > environment::print_application_options{ [] (void) -> void { return; } };
environment *entity::env = nullptr;
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) >
    kernel::make_reduction_kernel{
      [] (std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* {
        return new product_reduction_kernel(vec,scal); } };
int object::count = 0;
double object_data::create_data_count = 0.;
bool object_data::trace_create_data = false;
int algorithm::queue_trace_summary = 0;
int sparse_matrix::sparse_matrix_trace = 0;
int task::count = 0;

int main(int argc,char **argv) {

  unittest_product_setup(argc,argv);

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
