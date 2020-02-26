/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** product_base.cxx: Implementations of the product class
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <iostream>
using namespace std;
#include <mpi.h>

#include "product_base.h"

/****
 **** Basics
 ****/

/*!
  A product environment constructs a product architecture. See elsewhere for its structure.
 */
product_environment::product_environment(int argc,char **argv) : mpi_environment(argc,argv) {
  type = environment_type::PRODUCT;

  // discover MPI architecture
  int mytid,ntids;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
  // discover OpenMP architecture
  int nthreads;
#pragma omp parallel shared(nthreads)
#pragma omp master
  nthreads = omp_get_num_threads();

  try {
    arch = new product_architecture(mytid,ntids,nthreads);
    //arch->add_domain(mytid);
    {
      architecture *omp_arch = arch->get_embedded_architecture();
      if (has_argument("split"))
	  omp_arch->set_split_execution();
    }
  } catch (std::string c) { fmt::print("Environment error <<{}>>\n",c); }

};

std::string product_decomposition::as_string() {
  fmt::MemoryWriter w;
  w.write("Product decomposition based on <<{}>> and <<{}>>",
	  mpi_decomposition::as_string(),get_embedded_decomposition()->as_string());
  return w.str(); };

/****
 **** Distribution
 ****/

void product_distribution::set_dist_factory() {
  //! Factory method for cloning objects. This is used in \ref task::allocate_halo_vector.
  new_object = [this] (distribution *d) -> std::shared_ptr<object> {
    return std::shared_ptr<object>( new product_object(d) ); };
  //! Factory method used for cloning objects while reusing data.
  new_object_from_data = [this] ( double *d ) -> std::shared_ptr<object>
    { return std::shared_ptr<object>( new product_object(this,d) ); };
};

/****
 **** Object: it's all in the .h file
 ****/

/****
 **** Message: nothing particular
 ****/

/****
 **** Task
 ****/

/*!
  Make an \ref omp_distribution from an \ref mpi_distribution on
  a \ref processor_coordinate.

  \todo make actually private
*/
distribution *product_task::omp_distribution_from_mpi
    ( distribution *mpi_distr,processor_coordinate &mytid ) {
  decomposition *omp_decomp = mpi_distr->get_embedded_decomposition();
  distribution_type mpi_type = mpi_distr->get_type();

  distribution *omp_distr; 
  parallel_structure *omp_struct = new parallel_structure(omp_decomp);
  //snippet mpi2ompblock
  auto outidx = mpi_distr->get_processor_structure(mytid);
  if (mpi_type==distribution_type::REPLICATED) {
    omp_struct->create_from_replicated_indexstruct(outidx.get());
  } else {
    omp_struct->create_from_indexstruct(outidx.get());
  }
  omp_distr = new omp_distribution(omp_struct);
  omp_distr->lock_global_structure( mpi_distr->get_global_structure() );
  omp_distr->set_name( fmt::format("<<{}>>-embedded-on-p{}",
				   mpi_distr->get_name(),mytid.as_string()) );
  //snippet end
  // fmt::print("Global structure on {} set to {},{}\n",
  // 	     get_domain()->as_string(),
  // 	     mpi_distr->get_global_structure()->as_string(),
  // 	     omp_distr->get_global_structure()->as_string()
  // 	     );
  return omp_distr;
};

/*!
  Private method used in \ref product_task::local_analysis
  \todo not actually private right now
*/
std::shared_ptr<object> product_task::omp_object_from_mpi( std::shared_ptr<object> mpi_obj ) {
  if (!mpi_obj->has_data_status_allocated())
    throw(std::string("Product object needs allocated mpi object"));
  auto mycoord = get_domain();
  decomposition *omp_decomp = mpi_obj->get_embedded_decomposition();
  distribution *mpi_distr = dynamic_cast<distribution*>(mpi_obj.get());
  if (mpi_distr==nullptr)
    throw(std::string("Could not upcast to mpi distr"));
  distribution_type mpi_type = mpi_distr->get_type();

  distribution *omp_distr; std::shared_ptr<object> omp_obj;
  omp_distr = omp_distribution_from_mpi(mpi_distr,mycoord);
  if (mpi_type==distribution_type::REPLICATED) {
    index_int s = mpi_distr->local_allocation_p(mycoord);
    { auto outidx = mpi_distr->get_processor_structure(mycoord);
      if (s!=outidx->volume())
	throw(std::string("size incompatibility")); }
    // VLE final `s' parameter missing here
    omp_obj = omp_distr->new_object_from_data(mpi_obj->get_data(mycoord));
  } else {
    omp_obj = omp_distr->new_object_from_data(mpi_obj->get_data(mycoord));
  }
  omp_obj->set_name( fmt::format("<<{}>>-omp-alias",mpi_obj->get_name()) );
  // fmt::print("Create omp obj from mpi {} with global struct <<{}>>\n",
  // 	     omp_obj->get_name(),omp_obj->get_global_structure()->as_string());
  return omp_obj;
};

/*!
  This routine pertains to a task on the outer, MPI level. Since it contains
  a complete OpenMP task queue, the task requires local analysis.

  Since product is the simplest hybrid model this is is all fairly easy:
  - we already have an \ref omp_algorithm in the task,
  - we add a single kernel to it 
  - the output object of that kernel is a wrapping of the \ref outvector as an \ref omp_object
  - the input object a wrapping of either the invector or the halo
 */
void product_task::local_analysis() {
  decomposition *omp_decomp = get_out_object()->get_embedded_decomposition();
  auto mycoord = get_domain();

  omp_outobject = omp_object_from_mpi(get_out_object());
  omp_kernel *k;
  if (has_type_origin()) {
    omp_inobject = nullptr;
    k = new omp_origin_kernel(omp_outobject);
    k->set_name("embedded-omp-origin");
  } else {
    k = new omp_kernel(omp_outobject);
    k->set_name( fmt::format("<<{}>>-embedded-omp-kernel",get_name()) );
    if (get_dependencies().size()==0)
      throw(std::string("Suspiciously no dependencies for compute kernel"));
    for ( auto d : get_dependencies() ) {
      // origin kernel for the halo of this dependency
      auto inobject = omp_object_from_mpi( d->get_beta_object() );
      kernel *ko = new omp_origin_kernel(inobject);
      ko->set_name("embedded-omp-origin");
      node_queue->add_kernel(ko);
      // compute kernel for this dependency
      k->add_in_object(inobject);
      if (d->has_type_explicit_beta()) {
	k->last_dependency()->set_explicit_beta_distribution
	  ( omp_distribution_from_mpi(d->get_explicit_beta_distribution(),mycoord) );
      } else {
	k->last_dependency()->copy_from(d); // this copies the signature function
      }
      k->last_dependency()->set_name( fmt::format("{}-dependency",k->get_name()) );
    }
    if (k->get_dependencies().size()==0)
      throw("Suspiciously no dependencies for embedded compute kernel\n");
    k->set_name("embedded-omp-compute");
    k->set_localexecutectx( tasklocalexecutectx );
  }
  k->set_localexecutefn( tasklocalexecutefn );
  node_queue->add_kernel(k);

  node_queue->analyze_dependencies();
  if (omp_outobject->get_split_execution())
    node_queue->set_outer_as_synchronization_points();
  node_queue->determine_locally_executable_tasks();
};

/*!
  Execute the embedded tasks of \ref product_task: the \ref ctx argument is really
  an \ref omp_algorithm.

  In a product task, the local execute is called twice with a test that is not
  all-or-nothing, so we pass the test to the queue execute.
  \todo can we make that cast more elegant?
 */
void product_task::local_execute
    (std::vector<std::shared_ptr<object>> &beta_objects,std::shared_ptr<object> outobj,void *ctx,
     int(*tasktest)(std::shared_ptr<task> t)) {
  //  omp_algorithm *queue = (omp_algorithm*)ctx;
  //  printf("start embedded node queue conditional execute for step %d\n",get_step());
  if (!node_queue->has_type(algorithm_type::OMP)) // sanity check on that cast.....
    throw("Embedded queue not of OMP type");
  node_queue->execute(tasktest);
};

/****
 **** Kernel
 ****/

//! Construct the right kind of task for the base class method \ref kernel::split_to_tasks.
std::shared_ptr<task> product_kernel::make_task_for_domain(processor_coordinate &dom) {
  return std::shared_ptr<task>( new product_task(dom,this) );
};

/****
 **** Queue
 ****/

