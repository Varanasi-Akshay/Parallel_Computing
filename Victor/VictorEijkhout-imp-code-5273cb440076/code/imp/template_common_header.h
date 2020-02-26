/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** template_common_header.h :
 ****     general include stuff for all application templates
 ****
 ****************************************************************/

// #include <stdlib.h>
// #include <stdio.h>
#include <iostream>
using namespace std;

#define STATIC_VARS_HERE
#include "imp_static_vars.h"

#if defined(IMPisOMP)
#include <omp.h>
#include "omp_base.h"
#include "omp_ops.h"
#endif

#if defined(IMPisMPI)
#include <mpi.h>
#include "mpi_base.h"
#include "mpi_ops.h"
#endif

#if defined(IMPisPRODUCT)
#include <omp.h>
#include <mpi.h>
#include "product_base.h"
#include "product_ops.h"
#endif

// initialization of some class static variables
bool algorithm::do_optimize = false;
int algorithm::queue_trace_summary = 0;

int entity_name::entity_number = 0;
trace_level entity::tracing = trace_level::NONE;
std::vector<entity*> environment::list_of_all_entities;
environment *entity::env = nullptr;
std::function< void(void) > environment::print_application_options{ [] (void) -> void { return; } };
#if defined(IMPisOMP)
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) >
    kernel::make_reduction_kernel{
  [] (std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* {
    return new omp_reduction_kernel(vec,scal); } };
#endif
#if defined(IMPisMPI)
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) > kernel::make_reduction_kernel{
  [] (std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* { return new mpi_reduction_kernel(vec,scal); } };
#endif
#if defined(IMPisPRODUCT)
std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) > kernel::make_reduction_kernel{
  [] (std::shared_ptr<object> vec,std::shared_ptr<object> scal) -> kernel* { return new product_reduction_kernel(vec,scal); } };
#endif
int object::count = 0;
double object_data::create_data_count = 0.;
bool object_data::trace_create_data = false;
int sparse_matrix::sparse_matrix_trace = 0;
int task::count = 0;
/*
 * -- end of headers and static vars
 */
