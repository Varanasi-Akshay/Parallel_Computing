/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** Unit tests for the MPI+OMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for collective operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "product_base.h"
#include "product_ops.h"
#include "product_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Analyze gather dependencies","[collective][dependence][10]") {
  /*
   * this test case is basically spelling out "analyze_dependencies"
   * for the case of an allgather
   */

  // alpha distribution is one scalar per processor
  product_distribution *alpha;
  REQUIRE_NOTHROW( alpha = new product_block_distribution(decomp,product_nprocs) );
  index_int av;
  REQUIRE_NOTHROW( av = alpha->volume(mycoord) );
  CHECK( av==omp_nprocs );
  product_object *scalar;
  REQUIRE_NOTHROW( scalar = new product_object(alpha) );
  // gamma distribution is gathering those scalars
  product_distribution *gamma;
  REQUIRE_NOTHROW( gamma = new product_gathered_distribution(decomp) );
  product_object *gathered;
  REQUIRE_NOTHROW( gathered = new product_object(gamma) );
  product_kernel *gather;
  REQUIRE_NOTHROW( gather = new product_kernel(scalar,gathered) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );

  { // in the alpha structure I have only myself
    parallel_indexstruct *alpha_struct;
    REQUIRE_NOTHROW( alpha_struct = alpha->get_dimension_structure(0) );
    CHECK( alpha_struct->local_size(mytid)==omp_nprocs );
    CHECK( alpha_struct->first_index_r(mycoord)==omp_nprocs*mytid );
    CHECK( alpha->local_size_r(mycoord)==domain_coordinate(1,omp_nprocs) );
  }
  {
    distribution *omp_gamma;
    REQUIRE_NOTHROW
      ( omp_gamma = dynamic_cast<product_distribution*>(gamma)->get_omp_distribution() );
    CHECK( omp_gamma->get_type()==distribution_type::REPLICATED );
    CHECK( omp_gamma->/*brick*/domains_volume()==omp_nprocs );
    for (int p=0; p<omp_nprocs; p++) {
      processor_coordinate *pcoord;
      decomposition *ompdecomp;
      REQUIRE_NOTHROW( ompdecomp = decomp->get_embedded_decomposition() );
      REQUIRE_NOTHROW( pcoord = ompdecomp->coordinate_from_linear(p) );
      CHECK( omp_gamma->local_size_r(pcoord)==domain_coordinate(1,product_nprocs) );
    }
  }

  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK_NOTHROW( gather_task = dynamic_cast<product_task*>(gather->get_tasks().at(0)) );
  {
    std::vector<message*> *msgs;
    distribution *d = gather->get_beta_distribution();
    auto betablock = d->get_processor_structure(mycoord);
    {
      auto all_struct =
	std::shared_ptr<multi_indexstruct>
	( new multi_indexstruct( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,product_nprocs-1) ) ) );
      INFO( "comparing beta block " << betablock->as_string() <<
	    " to product_nprocs " << all_struct->as_string() );
      CHECK( betablock->equals(all_struct) );
    }
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   (mycoord,self_treatment::INCLUDE/* 0? */,betablock,betablock) );
    CHECK( msgs->size()==mpi_nprocs );
    for (int imsg=0; imsg<msgs->size(); imsg++) {
      message *msg = (*msgs)[imsg];
      //printf("msg to/from %d/%d\n",msg->get_receiver(),msg->get_sender());
      CHECK( msg->get_receiver().equals(mycoord) );
    }
  }
  // same but now as one call
  REQUIRE_NOTHROW( gather_task->last_dependency()->create_beta_vector(scalar) );
  CHECK_NOTHROW( gather_task->derive_receive_messages() );
  int nsends;
  CHECK_NOTHROW( nsends = gather_task->get_nsends() );
  CHECK( nsends==env->get_architecture()->nprocs() );

}

TEST_CASE( "Analyze gather dependencies in one go","[collective][dependence][11]") {

  // alpha distribution is one scalar per processor
  product_distribution *alpha;
  REQUIRE_NOTHROW( alpha = new product_block_distribution(decomp,-1,product_nprocs) );
  // gamma distribution is gathering those scalars
  product_distribution *gamma = new product_gathered_distribution(decomp);
  // general stuff
  product_object
    *scalar = new product_object(alpha),
    *gathered = new product_object(gamma);
  product_kernel *gather = new product_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( gamma );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  int nsends,sum;
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  sum=0; // check that we receive from everyone
  for (std::vector<message*>::iterator msg=gather_task->get_receive_messages()->begin();
       msg!=gather_task->get_receive_messages()->end(); ++msg) {
    sum += (*msg)->get_sender().coord(0);
  }
  CHECK( sum==(mpi_nprocs*(mpi_nprocs-1)/2) );

  CHECK_NOTHROW( gather_task->get_send_messages() );
  auto msgs = gather_task->get_send_messages();
  CHECK( msgs.size()==mpi_nprocs );
  sum=0; // check that we send to everyone
  for ( auto msg : msgs ) { //->begin(); msg!=msgs->end(); ++msg) {
    sum += msg->get_receiver().coord(0);
  }
  CHECK( sum==(mpi_nprocs*(mpi_nprocs-1)/2) );

  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK( gather_task->get_beta_object(0)!=nullptr );
  index_int hsize;
  CHECK_NOTHROW( hsize = gather_task->get_out_object()->volume(mycoord) );
  CHECK( hsize==product_nprocs );
  CHECK_NOTHROW( hsize = gather_task->get_beta_object(0)->volume(mycoord) );
  CHECK( hsize==product_nprocs );
}

TEST_CASE( "Actually gather something","[collective][dependence][12]") {

  // alpha distribution is one scalar per processor
  product_distribution *alpha;
  REQUIRE_NOTHROW( alpha = new product_block_distribution(decomp,product_nprocs) );
  // gamma distribution is gathering those scalars
  // 
  distribution *gamma = new product_gathered_distribution(decomp);
  CHECK( gamma->get_type()==distribution_type::REPLICATED );
  //  CHECK( gamma->local_allocation()==omp_nprocs*product_nprocs );

  // general stuff
  product_object
    *scalar = new product_object(alpha),
    *gathered = new product_object(gamma);
  product_kernel *gather = new product_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutefn( veccopy );

  std::shared_ptr<task> gather_task;
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  //CHECK_NOTHROW( gather_task->last_dependency()->allocate_beta_vector() );

  { // investigate the structure of the MPI task
    std::vector<message*> *msgs; object *obj; processor_coordinate *tid;
    REQUIRE_NOTHROW( tid = gather_task->get_domain() );
    CHECK( tid->equals(mycoord));
    REQUIRE_NOTHROW( msgs = gather_task->get_receive_messages() );
    CHECK( msgs->size()==mpi_nprocs );
    { INFO( "mpi input" );
      REQUIRE_NOTHROW( obj = gather_task->get_in_object(0) );
      CHECK( obj->local_size_r(tid)==domain_coordinate(1,omp_nprocs) );
      CHECK( obj->get_type()==distribution_type::BLOCKED );
    }
    { INFO( "mpi halo" );
      REQUIRE_NOTHROW( obj = gather_task->get_beta_object(0) );
      CHECK( obj->local_size_r(tid)==domain_coordinate(1,product_nprocs) );
      CHECK( obj->get_type()==distribution_type::REPLICATED );
    }
    { INFO( "mpi output" );
      REQUIRE_NOTHROW( obj = gather_task->get_out_object() );
      CHECK( obj->local_size_r(tid)==domain_coordinate(1,product_nprocs) );
      CHECK( obj->get_type()==distribution_type::REPLICATED );
    }
  }

  double *in = scalar->get_data(mycoord);
  for (int i=0; i<scalar->volume(mycoord); i++)
    REQUIRE_NOTHROW( in[i] = (double)mytid );
  CHECK_NOTHROW( gather_task->execute() );
  { // investigate the OpenMP stuff
    algorithm *queue;
    REQUIRE_NOTHROW( queue = dynamic_cast<product_task*>(gather_task)->get_node_queue() );
    std::vector<kernel*> *kerns;
    REQUIRE_NOTHROW( kerns = queue->get_kernels() );
    CHECK( kerns->size()==2 );
    for (auto k=kerns->begin(); k!=kerns->end(); ++k) {
      if ((*k)->has_type_origin()) {
	std::vector<task*> *tsks; object *obj;
	REQUIRE_NOTHROW( tsks = (*k)->get_tasks() );
	CHECK( tsks->size()==omp_nprocs );
	REQUIRE_NOTHROW( obj = tsks->at(0)->get_out_object() );
	CHECK( obj->get_type()==distribution_type::REPLICATED );
      } else {
	distribution *indist,*halodist;
	CHECK( (*k)->get_out_object()->get_type()==distribution_type::REPLICATED );
	std::vector<task*> *tsks;
	REQUIRE_NOTHROW( tsks = (*k)->get_tasks() );
	CHECK( tsks->size()==omp_nprocs );
	for (auto t=tsks->begin(); t!=tsks->end(); ++t) {
	  std::vector<message*> *msgs; object *obj; processor_coordinate *tid;
	  REQUIRE_NOTHROW( tid = (*t)->get_domain() );
	  REQUIRE_NOTHROW( msgs = (*t)->get_receive_messages() );
	  CHECK( msgs->size()==1 ); // ? omp_nprocs );
	  { INFO( "embedded input" ); 
	    REQUIRE_NOTHROW( obj = (*t)->get_in_object(0) );
	    CHECK( obj->get_type()==distribution_type::REPLICATED );
	    CHECK( obj->local_size_r(tid)==domain_coordinate(1,product_nprocs) );
	  }
	  { INFO( "embedded halo" );
	    REQUIRE_NOTHROW( obj = (*t)->get_beta_object(0) );
	    CHECK( obj->local_size_r(tid)==domain_coordinate(1,product_nprocs) );
	  }
	  { INFO( "embedded output" );
	    REQUIRE_NOTHROW( obj = (*t)->get_out_object() );
	    CHECK( obj->local_size_r(tid)==domain_coordinate(1,product_nprocs) );
	  }
	}
      }
    }
    std::vector<task*> *tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks->size()==2*omp_nprocs );
  }
  {
    // architecture *omp_arch;
    // REQUIRE_NOTHROW( omp_arch = gathered->get_embedded_architecture() );
    product_distribution *product_distr; distribution *pre_product_distr;
    //REQUIRE_NOTHROW( pre_product_distr = gathered->get_distribution() );
    //REQUIRE_NOTHROW( product_distr = dynamic_cast<product_distribution*>(pre_product_distr) );
    REQUIRE_NOTHROW( product_distr = dynamic_cast<product_distribution*>(gathered) );
    CHECK( product_distr->get_type()==distribution_type::REPLICATED );
    for (int p=0; p<omp_nprocs; p++) {
      processor_coordinate *pcoord;
      decomposition *ompdecomp;
      REQUIRE_NOTHROW( ompdecomp = decomp->get_embedded_decomposition() );
      REQUIRE_NOTHROW( pcoord = ompdecomp->coordinate_from_linear(p) );
      CHECK( product_distr->embedded_volume(mycoord,pcoord)==product_nprocs );
      // ?? ==domain_coordinate(1,product_nprocs) );
      double *out;
      //      REQUIRE_NOTHROW( out = gathered->get_data(mycoord,p) );
    }
  }
}

TEST_CASE( "Gather more than one something","[collective][dependence][13]") {
  printf("ortho probably borken in test 13\n"); return;

  INFO( "mytid=" << mytid );
  // alpha distribution is k scalars per processor
  int k; const char *testcase;
  SECTION( "k=1" ) {
    k = 1; testcase = "k=1";
  }
  SECTION( "k=4" ) {
    k = 4; testcase = "k=4";
  }
  INFO( "test case: " << testcase );

  // k scalars per proc
  product_distribution *alpha = new product_block_distribution(decomp,k,1,-1);
  product_object *scalars = new product_object(alpha);
  CHECK( alpha->local_allocation()==k );
  double *in = scalars->get_data(mycoord);
  for (int ik=0; ik<k; ik++)
    in[ik] = (ik+1)*(double)mytid;
  // gamma distribution is gathering those scalars
  product_distribution *gamma = new product_gathered_distribution(decomp,k);
  product_object *gathered = new product_object(gamma);
  CHECK( gamma->local_allocation()==k*mpi_nprocs );

  product_kernel *gather = new product_kernel(scalars,gathered);
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutectx( (void*)&k );
  gather->set_localexecutefn( &veccopy );

  std::shared_ptr<task> gather_task;
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = (product_task*) ( (*(gather->get_tasks()))[0] ) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  //CHECK_NOTHROW( gather_task->last_dependency()->allocate_beta_vector() );
  CHECK_NOTHROW( gather->execute() );
  double *out = gathered->get_data(mycoord);
  for (int p=0; p<mpi_nprocs; p++) {
    INFO( "p contribution from " << p );
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      CHECK( out[ik+p*k]==(ik+1)*p );
    }
  }
}

TEST_CASE( "Gather and sum as two kernels","[collective][dependence][14]") {

  INFO( "mytid: " << mytid );

  // input distribution is one scalar per processor
  product_distribution *distributed = new product_block_distribution(decomp,product_nprocs);
  product_object       *scalar      = new product_object(distributed);
  double *scalar_data = scalar->get_data(mycoord);
  for (index_int i=0; i<scalar->volume(mycoord); i++)
    REQUIRE_NOTHROW( scalar_data[i] = (double)mytid );

  // intermediate distribution is gathering those scalars replicated
  product_distribution *collected = new product_gathered_distribution(decomp);
  product_object       *gathered  = new product_object(collected);

  // final distribution is redundant one scalar
  product_distribution *replicated     = new product_replicated_distribution(decomp);
  product_object       *sum      = new product_object(replicated);

  // first kernel: gather
  product_kernel *gather = new product_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( collected );

  std::shared_ptr<task> gather_task;
  gather->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = dynamic_cast<product_task*>(gather->get_tasks().at(0)) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  //CHECK_NOTHROW( gather_task->last_dependency()->allocate_beta_vector() );
  CHECK_NOTHROW( gather_task->execute() );
  // let's look at the gathered data
  CHECK( gathered->volume(mycoord)==product_nprocs );
  CHECK( gathered->get_orthogonal_dimension()==1 );
  {
    double *gathered_data = gathered->get_data(mycoord);
    for (int i=0; i<product_nprocs; i++) {
      INFO( "i=" << i );
      CHECK( gathered_data[i]==(double)i );
    }
  }
  
  // second kernel: local sum
  product_kernel *localsum = new product_kernel(gathered,sum);
  localsum->set_localexecutefn( &summing );
  std::shared_ptr<task> sum_task;

  const char *beta_strategy = "none";
  //  SECTION( "explicit beta works" ) {
    beta_strategy = "beta is collected distribution";
    localsum->set_explicit_beta_distribution( collected );
    //  }
  // SECTION(  "derivation does not work" ) {
  //   beta_strategy = "beta derived";
  //   localsum->last_dependency()->set_type_local();
  // }
    //  INFO( beta_strategy );

  REQUIRE_NOTHROW( localsum->last_dependency()->ensure_beta_distribution(sum) );
  CHECK( localsum->get_beta_distribution()->volume(mycoord)==mpi_nprocs );
  CHECK_NOTHROW( localsum->split_to_tasks() );
  CHECK( localsum->get_tasks().size()==1 );
  CHECK_NOTHROW( sum_task = (*(localsum->get_tasks()))[0] );
  CHECK_NOTHROW( sum_task->analyze_dependencies(/*gathered*/) );
  CHECK_NOTHROW( sum_task->last_dependency()->create_beta_vector(sum) );
  //CHECK_NOTHROW( sum_task->last_dependency()->allocate_beta_vector() );

  std::vector<message*> *msgs;
  REQUIRE_NOTHROW( msgs = sum_task->get_receive_messages() );
  CHECK( msgs->size()==1 );
  CHECK( msgs->at(0)->get_sender().coord(0)==mytid );
  CHECK_NOTHROW( sum_task->execute() );

  double *in = scalar->get_data(mycoord),*out = sum->get_data(mycoord);
  CHECK( (*out)==(mpi_nprocs*(mpi_nprocs-1)/2) );
}

#if 0

TEST_CASE( "Inner product in three kernels","[collective][15]" ) {
  INFO( "mytid: " << mytid );

  // let's go for an irregular distribution
  product_distribution 
    *alpha = new product_block_distribution(decomp,10+2*mytid,-1),
    *local_scalar = new product_block_distribution(decomp,1,-1),
    *gathered_scalar = new product_gathered_distribution(decomp),
    *summed_scalar = new product_replicated_distribution(decomp);
  index_int my_first = alpha->first_index_r(mycoord);
  {
    index_int g = mpi_nprocs*10+mpi_nprocs*(mpi_nprocs-1);
    CHECK( alpha->global_volume()==g );
  }
  product_object 
    *x = new product_object(alpha), *y = new product_object(alpha);
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  double *xdata,*ydata;
  REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
  for (index_int i=0; i<x->local_size(mycoord); i++)
    xdata[i] = 2.;
  REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
  for (index_int i=0; i<y->local_size(mycoord); i++)
    ydata[i] = my_first+i;

  // output object
  product_object
    *global_sum = new product_object(summed_scalar);
  global_sum->set_name("inprod-value");

  const char *path;
  SECTION( "explicit intermediates" ) {
    path = "explicit intermediates";
    // intermediate objects
    product_object
      *local_value = new product_object(local_scalar),
      *gatheredvalue = new product_object(gathered_scalar);

    // local product is local, second vector comes in as context
    product_kernel
      *local_product = new product_kernel(x,local_value);
    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(x) );
      //set_type_local();
    REQUIRE_NOTHROW( local_product->add_in_object(y) );
    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(y) );
    //set_type_local() );
    local_product->set_localexecutefn( &local_inner_product );
    REQUIRE_NOTHROW( local_product->analyze_dependencies() );
    
    //    SECTION( "two kernels" ) {
    // cout << "two kernels\n";
      product_kernel
  	*gather_and_sum = new product_kernel(local_value,global_sum);
      gather_and_sum->set_explicit_beta_distribution(gathered_scalar);
      gather_and_sum->set_localexecutefn( &summing );

      REQUIRE_NOTHROW( gather_and_sum->analyze_dependencies() );
      REQUIRE_NOTHROW( local_product->execute() );
      REQUIRE_NOTHROW( gather_and_sum->execute() );
      //}
  }

  SECTION( "inner product kernel" ) {
    path = "inner product kernel";
    product_kernel *innerproduct = new product_innerproduct_kernel(x,y,global_sum);
    REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
    REQUIRE_NOTHROW( innerproduct->execute() );
  }
  INFO( "path is: " << path );

  index_int g = alpha->global_volume(); 
  double *zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_data(mycoord) );
  CHECK( global_sum->local_size(mycoord)==1 );
  CHECK( zdata[0]==g*(g-1.) );

}

#endif
