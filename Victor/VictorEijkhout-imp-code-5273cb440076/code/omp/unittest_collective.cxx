/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for collective operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_ops.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Analyze gather dependencies","[collective][dependence][10]") {
  /*
   * this test case is basically spelling out "analyze_dependencies"
   * for the case of an allgather
   */

  //snippet gatherdists
  omp_distribution *alpha,*gamma;
  std::shared_ptr<object> scalar,gathered;
  // alpha distribution is one scalar per processor
  REQUIRE_NOTHROW( alpha = new omp_block_distribution(decomp,ntids) );
  REQUIRE_NOTHROW( scalar = std::shared_ptr<object>( new omp_object(alpha) ) );
  // gamma distribution is gathering those scalars
  REQUIRE_NOTHROW( gamma = new omp_gathered_distribution(decomp) );
  REQUIRE_NOTHROW( gathered = std::shared_ptr<object>( new omp_object(gamma) ) );
  omp_kernel
    *gather = new omp_kernel(scalar,gathered);
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  //snippet end

  // in the alpha structure I have only myself
  parallel_structure *alpha_struct = dynamic_cast<parallel_structure*>(alpha); //->get_processor_structure();
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( alpha->get_processor_structure(mycoord)->volume()==1 );
    CHECK( alpha->get_processor_structure(mycoord)->first_index_r()[0]==mytid );
  }

  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  distribution *d = gather->get_beta_distribution();

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    std::vector<message*> msgs;
    auto betablock = d->get_processor_structure(mycoord);
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   ( mycoord,self_treatment::INCLUDE,betablock,betablock) );
    CHECK( msgs.size()==ntids );
    for ( auto msg : msgs ) { //int imsg=0; imsg<msgs->size(); imsg++) {
      CHECK( msg->get_receiver().equals(mycoord) );
    }
    // same but now as one call
    REQUIRE_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
    REQUIRE_NOTHROW( gather_task->derive_receive_messages(/*0,mytid*/) );
    // int nsends;
    // CHECK_NOTHROW( nsends = gather_task->get_nsends(env,gather_task->get_receive_messages()) );
    // CHECK( nsends==ntids );
  }
}

TEST_CASE( "Analyze gather dependencies in one go","[collective][dependence][11]") {

  // alpha distribution is one scalar per processor
  omp_distribution *alpha = 
    new omp_block_distribution(decomp,ntids);
  // gamma distribution is gathering those scalars
  omp_gathered_distribution *gamma = new omp_gathered_distribution(decomp);
  // general stuff
  auto
    scalar = std::shared_ptr<object>( new omp_object(alpha) ),
    gathered = std::shared_ptr<object>( new omp_object(gamma) );
  omp_kernel *gather = new omp_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( gamma );
  //  REQUIRE_NOTHROW( gather->derive_beta_distribution() );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==ntids );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  int nsends,sum;
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  sum=0;
  auto msgs = gather_task->get_receive_messages();
  for ( auto msg : msgs ) {
    sum += msg->get_sender().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  // CHECK_NOTHROW( gather_task->set_send_messages
  // 		 (gather_task->create_send_structure_for_task(0,mytid,alpha) ) );
  // CHECK_NOTHROW( gather_task->get_send_messages() );
  // std::vector<message*> *msgs = gather_task->get_send_messages();
  // CHECK( msgs->size()==ntids );
  // sum=0;
  // for (std::vector<message*>::iterator msg=msgs->begin(); msg!=msgs->end(); ++msg) {
  //   sum += (*msg)->get_receiver();
  // }
  // CHECK( sum==(ntids*(ntids-1)/2) );

  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK( gather_task->get_beta_object(0)!=nullptr );
  index_int hsize;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK_NOTHROW( hsize = gather_task->get_out_object()->volume(mycoord) );
    CHECK( hsize==ntids );
    CHECK_NOTHROW( hsize = gather_task->get_beta_object(0)->volume(mycoord) );
    CHECK( hsize==ntids );
    // CHECK_NOTHROW( hsize = gather_task->get_invector()->volume(mycoord) );
    // CHECK( hsize==1 );
  }
}

TEST_CASE( "Actually gather and reduce something","[collective][dependence][reduction][12]") {
  /*
    This test is not quite correct. The `summing' routine
    really wants a scalar output, not a gathered array.
   */

  // alpha distribution is one scalar per processor
  omp_distribution *alpha = new omp_block_distribution(decomp,ntids);
  // gamma distribution is gathering those scalars
  omp_distribution *gamma = new omp_gathered_distribution(decomp);
  // general stuff
  auto
    scalar = std::shared_ptr<object>( new omp_object(alpha) ),
    gathered = std::shared_ptr<object>( new omp_object(gamma) );
  omp_kernel *gather;

  omp_algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(scalar) ) );
  double *in;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( scalar->volume(mycoord)==1 );
    REQUIRE_NOTHROW( in = scalar->get_data(mycoord) );
    *in = (double)mytid;
  }

  std::shared_ptr<task> gather_task;
  const char *path;
  SECTION( "spell out the kernel" ) {
    path = "spelled out";
    gather = new omp_kernel(scalar,gathered);
    gather->set_explicit_beta_distribution( gamma );
    gather->set_localexecutefn( &summing );
  }
  // SECTION( "use a reduction kernel" ) {
  //   path = "kernel";
  //   REQUIRE_NOTHROW( gather = new omp_reduction_kernel(scalar,gathered) );
  // }
  // INFO( "gather strategy: " << path );

  REQUIRE_NOTHROW( queue->add_kernel( gather ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  CHECK( gather->get_tasks().size()==ntids );
  // REQUIRE_NOTHROW( queue->execute() );
  // double *out = gathered->get_data(new processor_coordinate_zero(1));
  // CHECK( (*out)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Gather more than one something","[collective][dependence][13]") {
  printf("13: multi-component reduction disabled\n"); return;

  // alpha distribution is k scalars per processor
  int k, P = ntids;
  const char *path;
  SECTION( "reproduce single gather" ) {
    path = "single"; k = 1;
  }
  printf("[13] multiple gather disabled\n");
  // SECTION( "new: multiple gather" ) {
  //   path = "multiple" ; k = 4;
  // }
  INFO( "path: " << path );
  
  omp_distribution *alpha ,*gamma;
  alpha = new omp_block_distribution(decomp,k,P);
  // gamma distribution is gathering those scalars
  REQUIRE_NOTHROW( gamma = new omp_gathered_distribution(decomp,k) );
  CHECK( gamma->has_type_replicated() );
  CHECK( gamma->get_orthogonal_dimension()==k );
  // general stuff
  std::shared_ptr<object> scalars,gathered;
  REQUIRE_NOTHROW( scalars = std::shared_ptr<object>( new omp_object(alpha) ) );
  REQUIRE_NOTHROW( gathered = std::shared_ptr<object>( new omp_object(gamma) ) );
  omp_kernel *gather;
  REQUIRE_NOTHROW(  gather = new omp_kernel(scalars,gathered) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  gather->set_localexecutectx( (void*)&k );
  gather->set_localexecutefn( &veccopy );

  std::shared_ptr<task> gather_task;
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==P );
  omp_algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(scalars) ) );
  REQUIRE_NOTHROW( queue->add_kernel( gather ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  double *in,*out;
  REQUIRE_NOTHROW( in = scalars->get_data(new processor_coordinate_zero(1)) );
  for (int p=0; p<ntids; p++) {
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      in[p*k+ik] = (ik+1)*(double)p;
    }
  }
  REQUIRE_NOTHROW( queue->execute() );
  for (int p=0; p<ntids; p++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(p) );
    REQUIRE_NOTHROW( out = gathered->get_data(mycoord) );
    //printf("%d checking out  data at %ld\n",p,(long)out);
    INFO( "on proc " << p );
    for (int ik=0; ik<k; ik++) {
      INFO( "ik: " << ik );
      CHECK( out[ik+p*k]==(ik+1)*p );
      //CHECK( out[ik]==((ik+1)*ntids*(ntids-1)/2) );
    }
  }
}

TEST_CASE( "Gather and sum as two kernels","[collective][dependence][14]") {

  // input distribution is one scalar per processor
  omp_distribution *distributed,*collected,*replicated;
  REQUIRE_NOTHROW( distributed = new omp_block_distribution(decomp,ntids) );
  auto scalar      = std::shared_ptr<object>( new omp_object(distributed) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( scalar->volume(mycoord)==1 );
  }

  double *scalar_data, *gathered_data;
  REQUIRE_NOTHROW( scalar_data = scalar->get_data(processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    // could actually do get_data(mycoord) here, but still have to index:
    scalar_data[mytid] = (double)mytid+.5;
  }
  
  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(scalar) ) );

  // intermediate distribution is gathering those scalars replicated
  REQUIRE_NOTHROW( collected = new omp_gathered_distribution(decomp) );
  auto gathered  = std::shared_ptr<object>( new omp_object(collected) );
  CHECK( gathered->get_orthogonal_dimension()==1 );
  REQUIRE_NOTHROW( gathered->allocate() );
  double *data0;
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "mytid=" << mytid );
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( gathered->volume(mycoord)==ntids );
    double *data;
    REQUIRE_NOTHROW( data = gathered->get_data(mycoord) );
    if (mytid==0) data0 = data;
    CHECK( (data-data0)==(size_t)(mytid*ntids*sizeof(double)) );
  }

  // final distribution is redundant one scalar
  REQUIRE_NOTHROW( replicated     = new omp_replicated_distribution(decomp) );
  auto sum      = std::shared_ptr<object>( new omp_object(replicated) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( sum->volume(mycoord)==1 );
  }

  // first kernel: gather
  omp_kernel *gather;
  REQUIRE_NOTHROW( gather = new omp_kernel(scalar,gathered) );
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( collected ) );
  gather->set_localexecutefn( &veccopy );

  REQUIRE_NOTHROW( queue->add_kernel( gather ) );

  SECTION( "analyze intermediate" ) {
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = queue->get_tasks() );
    REQUIRE( tasks.size()==2*ntids );
    for (auto t : tasks) { //->begin(); t!=tasks->end(); ++t) {
      if (!t->has_type_origin()) {
	std::vector<message*> msgs;
	REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
	REQUIRE( msgs.size()==ntids );
	int *senders = new int[ntids]; for (int i=0; i<ntids; i++) senders[i] = 0;
	for (auto m : msgs) { //->begin(); m!=msgs->end(); ++m) {
	  senders[ m->get_sender().coord(0) ]++; // mark all senders
	}
	for (int i=0; i<ntids; i++)
	  REQUIRE( senders[i]==1 ); // make sure all senders occur once
      }
    }
    REQUIRE_NOTHROW( queue->execute() );
    
    double *prev_data;
    CHECK( gathered->has_type_replicated() );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      INFO( "investigating gathered data on p=" << mytid );
      REQUIRE_NOTHROW( gathered_data = gathered->get_data(mycoord) );
      if (mytid>0) // check stride: each proc has ntids elements
	CHECK( (long)gathered_data==(long)( prev_data+ntids ) );
      for (int i=0; i<ntids; i++) {
	INFO( "i: " << i );
	double d;
	REQUIRE_NOTHROW( d = gathered_data[i] );
	CHECK( d==Approx((double)i+.5) );
      }
      prev_data = gathered_data;
    }
  }
  SECTION( "add the second kernel" ) {
    // second kernel: local sum
    omp_kernel *localsum = new omp_kernel(gathered,sum);
    localsum->set_localexecutefn( &summing );

    const char *beta_strategy = "none";
    SECTION( "explicit beta works" ) {
      beta_strategy = "beta is collected distribution";
      localsum->set_explicit_beta_distribution( collected );
    }
    // SECTION(  "derivation does not work" ) {
    //   beta_strategy = "beta derived";
    //   localsum->set_type_local();
    // }
    INFO( beta_strategy );

    REQUIRE_NOTHROW( queue->add_kernel(localsum) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      CHECK( localsum->get_beta_distribution()->volume(mycoord)==ntids );
    }
    CHECK( localsum->get_tasks().size()==ntids );

    REQUIRE_NOTHROW( queue->execute() );

    double *out;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      INFO( "thread " << mytid );
      CHECK( sum->volume(mycoord)==1 );
      REQUIRE_NOTHROW( out = sum->get_data(mycoord) );
      REQUIRE( out!=nullptr );
      //      CHECK( (*out)==(ntids*(ntids-1)/2) );
    }
  }
}

TEST_CASE( "Inner product in three kernels","[collective][30]" ) {

  // let's go for an irregular distribution
  std::vector<index_int> local_sizes;
  for (int mytid=0; mytid<ntids; mytid++)
    local_sizes.push_back(10+2*mytid);
  omp_distribution 
    *alpha = new omp_block_distribution(decomp,local_sizes,-1),
    *localscalar = new omp_block_distribution(decomp,1,-1),
    *gathered_distribution = new omp_gathered_distribution(decomp),
    *summed_scalar = new omp_replicated_distribution(decomp);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_volume()==g );
  }
  auto
    x = std::shared_ptr<object>( new omp_object(alpha) ),
    y = std::shared_ptr<object>( new omp_object(alpha) );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  double *xdata,*ydata;
  REQUIRE_NOTHROW( xdata = x->get_raw_data() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<x->global_volume(); i++)
    xdata[i] = 2.;
  REQUIRE_NOTHROW( ydata = y->get_raw_data() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<y->global_volume(); i++)
    ydata[i] = i;

  // output object
  auto globalvalue = std::shared_ptr<object>( new omp_object(summed_scalar) );
  globalvalue->set_name("inprod-value");

  algorithm *inprod = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( inprod->add_kernel( new omp_origin_kernel(x) ) );
  REQUIRE_NOTHROW( inprod->add_kernel( new omp_origin_kernel(y) ) );
  
  SECTION( "explicit intermediates" ) {
    std::cout << "explicit intermediates\n";
    // intermediate objects
    auto
      localvalue = std::shared_ptr<object>( new omp_object(localscalar) ),
      gatheredvalue = std::shared_ptr<object>( new omp_object(gathered_distribution) );

    // local product is local, second vector comes in as context
    omp_kernel
      *local_product = new omp_kernel(x,localvalue);
    local_product->set_localexecutefn( &local_inner_product );

    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(x.get()) );
    REQUIRE_NOTHROW( local_product->add_in_object(y) );
    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(y.get()) );

    REQUIRE_NOTHROW( inprod->add_kernel(local_product) );
    
    // SECTION( "three kernels" ) {
    //   std::cout << "three kernels\n";
    //   omp_kernel
    // 	*gather = new omp_kernel(localvalue,gatheredvalue),
    // 	*sum = new omp_kernel(gatheredvalue,globalvalue);
    //   gather->set_explicit_beta_distribution(gathered_distribution);
    //   gather->set_localexecutefn( &veccopy );
    //   REQUIRE_NOTHROW( gather->ensure_beta_distribution(gatheredvalue) );
    //   REQUIRE_NOTHROW( gather->analyze_dependencies() );
    //   // sum is local
    //   sum->set_type_local();
    //   sum->set_localexecutefn( &summing );

    //   inprod->add_kernel(sum);
    //   inprod->add_kernel(gather);
    // }

    // SECTION( "two kernels" ) {
    std::cout << "two kernels\n";
    omp_kernel
      *gather_and_sum = new omp_kernel(localvalue,globalvalue);
    gather_and_sum->set_explicit_beta_distribution(gathered_distribution);
    gather_and_sum->set_localexecutefn( &summing );

    inprod->add_kernel(gather_and_sum);
      // }
  }

  SECTION( "inner product kernel" ) {
    omp_kernel *innerproduct = new omp_innerproduct_kernel(x,y,globalvalue);
    REQUIRE_NOTHROW( inprod->add_kernel( innerproduct ) );
  }
  
  REQUIRE_NOTHROW( inprod->analyze_dependencies() );
  REQUIRE_NOTHROW( inprod->execute() );

  index_int g = alpha->global_volume(); 
  double *zdata;
  REQUIRE_NOTHROW( zdata = globalvalue->get_raw_data() );
  CHECK( globalvalue->global_volume()==1 );
  CHECK( zdata[0]==g*(g-1.) );

}

TEST_CASE( "Inner product with different strategies","[collective][31]" ) {

  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  
  std::string path;
  SECTION( "point-to-point" ) { path = std::string("point-to-point");
    aa->set_collective_strategy_ptp();
  }
  SECTION( "grouping" ) { path = std::string("grouping");
    aa->set_collective_strategy_group();
  }
  SECTION( "treewise" ) { path = std::string("treewise");
    aa->set_collective_strategy_recursive();
  }
  SECTION( "collectivewise" ) { path = std::string("collective");
    aa->set_collective_strategy(collective_strategy::MPI);
  }
  INFO( "collectives are done " << path );

  decomposition *decomp = new omp_decomposition(aa);

  // let's go for an irregular distribution
  std::vector<index_int> local_sizes;
  for (int mytid=0; mytid<ntids; mytid++)
    local_sizes.push_back(10+2*mytid);
  omp_distribution 
    *alpha = new omp_block_distribution(decomp,local_sizes,-1),
    *local_scalar = new omp_block_distribution(decomp,1,-1),
    *gathered_scalar = new omp_gathered_distribution(decomp),
    *summed_scalar = new omp_replicated_distribution(decomp);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_volume()==g );
  }
  auto
    x = std::shared_ptr<object>( new omp_object(alpha) ),
    y = std::shared_ptr<object>( new omp_object(alpha) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  double *xdata,*ydata;
  REQUIRE_NOTHROW( xdata = x->get_raw_data() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<x->global_volume(); i++)
    xdata[i] = 2.;
  REQUIRE_NOTHROW( ydata = y->get_raw_data() ); //(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<y->global_volume(); i++)
    ydata[i] = i;

  // output object
  auto global_sum = std::shared_ptr<object>( new omp_object(summed_scalar) );
  global_sum->set_name("inprod-value");

  algorithm *inprod = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( inprod->add_kernel( new omp_origin_kernel(x) ) );
  REQUIRE_NOTHROW( inprod->add_kernel( new omp_origin_kernel(y) ) );

  omp_kernel *innerproduct;
  REQUIRE_NOTHROW( innerproduct = new omp_innerproduct_kernel(x,y,global_sum) );
  REQUIRE_NOTHROW( inprod->add_kernel( innerproduct ) );

  REQUIRE_NOTHROW( inprod->analyze_dependencies() );
  REQUIRE_NOTHROW( inprod->execute() );

  index_int g = alpha->global_volume(); 
  double *zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_raw_data() ); //(new processor_coordinate_zero(1)) );
  CHECK( global_sum->global_volume()==1 );
  CHECK( zdata[0]==g*(g-1.) );

}
