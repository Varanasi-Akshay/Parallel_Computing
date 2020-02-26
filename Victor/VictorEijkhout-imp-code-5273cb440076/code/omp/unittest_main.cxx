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
 **** Main for the OpenMP backend tests
 ****
 ****************************************************************/

#include <stdlib.h>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#define STATIC_VARS_HERE
#include "imp_static_vars.h"
#define omp_VARS_HERE
#include "omp_static_vars.h"

void unittest_omp_setup(int argc,char **argv) {

  try {
    env = new omp_environment(argc,argv);
  }
  catch (int x) {
    printf("Could not even get started\n"); throw(1);
  }

#pragma omp parallel
#pragma omp single
  ntids = omp_get_num_threads();

  arch = (omp_architecture*)env->get_architecture();
  decomp = new omp_decomposition(arch);

  return;
}

#include "omp_base.h"
#include "omp_ops.h"

// initialization of some class static variables
bool algorithm::do_optimize = false;
int entity_name::entity_number = 0;
trace_level entity::tracing = trace_level::NONE;
std::vector<entity*> environment::list_of_all_entities;
std::function< void(void) > environment::print_application_options{ [] (void) -> void { return; } };
environment *entity::env = nullptr;
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) > kernel::make_reduction_kernel{
  [] ( std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* {
    return new omp_reduction_kernel(vec,scal); } };
int object::count = 0;
double object_data::create_data_count = 0.;
bool object_data::trace_create_data = false;
int algorithm::queue_trace_summary = 0;
int sparse_matrix::sparse_matrix_trace = 0;
int task::count = 0;

int main(int argc,char **argv) {

  unittest_omp_setup(argc,argv);

  for (int a=0; a<argc; a++)
    if (!strcmp(argv[a],"--imp")) { argc = a; break; }

  int result = Catch::Session().run( argc, argv );

  delete env;

  return result;
}
