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
 **** unit tests for collective operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Analyze gather dependencies","[collective][dependence][10]") {
  /*
   * this test case is basically spelling out "analyze_dependencies"
   * for the case of an allgather
   */

  //snippet gatherdists
  // alpha distribution is one scalar per processor
  mpi_distribution 
    *alpha = new mpi_block_distribution(decomp,env->get_architecture()->nprocs());
  auto scalar = std::shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( scalar->allocate() );
  // gamma distribution is gathering those scalars
  mpi_distribution
    *gamma = new mpi_gathered_distribution(decomp);
  auto gathered = std::shared_ptr<object>( new mpi_object(gamma) );
  mpi_kernel
    *gather = new mpi_kernel(scalar,gathered);
  REQUIRE_NOTHROW( gather->set_explicit_beta_distribution( gamma ) );
  //snippet end

  // in the alpha structure I have only myself
  auto alpha_struct = alpha->get_dimension_structure(0);
  CHECK( alpha_struct->get_processor_structure(mytid)->local_size()==1 );
  CHECK( alpha_struct->get_processor_structure(mytid)->first_index()==mytid );

  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  {
    std::vector<message*> msgs;
    distribution *d = gather->get_beta_distribution();
    auto betablock = d->get_processor_structure(mycoord);//get_dimension_structure(0)->
    CHECK_NOTHROW( msgs = alpha->messages_for_segment
		   (mycoord,self_treatment::INCLUDE,betablock,betablock) );
    CHECK( msgs.size()==ntids );
    for ( auto msg : msgs ) { //int imsg=0; imsg<msgs->size(); imsg++) {
      CHECK( msg->get_receiver()==mycoord );
    }
  }
  // same but now as one call
  REQUIRE_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK_NOTHROW( gather_task->derive_receive_messages(/*0,mytid*/) );
  int nsends;
  CHECK_NOTHROW( nsends = gather_task->get_nsends() );
  CHECK( nsends==env->get_architecture()->nprocs() );

}

TEST_CASE( "Analyze gather dependencies in one go","[collective][dependence][11]") {

  // alpha distribution is one scalar per processor
  mpi_distribution *alpha = 
    new mpi_block_distribution(decomp,-1,env->get_architecture()->nprocs());
  // gamma distribution is gathering those scalars
  mpi_distribution *gamma = new mpi_gathered_distribution(decomp);
  // general stuff
  auto scalar = std::shared_ptr<object>( new mpi_object(alpha) );
  auto gathered = std::shared_ptr<object>( new mpi_object(gamma) );
  mpi_kernel *gather = new mpi_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( gamma );
  //  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(o1) );

  std::shared_ptr<task> gather_task;
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );

  int nsends,sum;
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  sum=0;
  // for (std::vector<message*>::iterator msg=gather_task->get_receive_messages()->begin();
  //      msg!=gather_task->get_receive_messages()->end(); ++msg) {
  for ( auto msg : gather_task->get_receive_messages() ) {
    sum += msg->get_sender().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  CHECK_NOTHROW( gather_task->get_send_messages() );
  std::vector<message*> msgs = gather_task->get_send_messages();
  CHECK( msgs.size()==ntids );
  sum=0;
  //for (std::vector<message*>::iterator msg=msgs->begin(); msg!=msgs->end(); ++msg) {
  for ( auto msg : msgs ) {
    sum += msg->get_receiver().coord(0);
  }
  CHECK( sum==(ntids*(ntids-1)/2) );

  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK( gather_task->get_beta_object(0)!=nullptr );
  index_int hsize;
  CHECK_NOTHROW( hsize = gather_task->get_out_object()->volume(mycoord) );
  CHECK( hsize==ntids );
  CHECK_NOTHROW( hsize = gather_task->get_beta_object(0)->volume(mycoord) );
  CHECK( hsize==ntids );
}

TEST_CASE( "Actually gather something","[collective][dependence][12]") {

  // alpha distribution is one scalar per processor
  mpi_distribution *alpha = 
    new mpi_block_distribution(decomp,-1,env->get_architecture()->nprocs());
  // gamma distribution is gathering those scalars
  mpi_distribution *gamma = new mpi_gathered_distribution(decomp);
  // general stuff
  auto scalar = std::shared_ptr<object>( new mpi_object(alpha) );
  auto gathered = std::shared_ptr<object>( new mpi_object(gamma) );
  REQUIRE_NOTHROW( scalar->allocate() );
  mpi_kernel *gather = new mpi_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutefn( summing );

  std::shared_ptr<task> gather_task;
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  double *in; REQUIRE_NOTHROW( in = scalar->get_data(mycoord) );
  *in = (double)mytid;
  CHECK_NOTHROW( gather_task->execute() );
  double *out; REQUIRE_NOTHROW( out = gathered->get_data(mycoord) );
  CHECK( (*out)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Gather more than one something","[collective][dependence][ortho][13]") {

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
  mpi_distribution *alpha = new mpi_block_distribution(decomp,k,1,-1);
  auto scalars = std::shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( scalars->allocate() );
  CHECK( alpha->local_allocation()==k );
  double *in; REQUIRE_NOTHROW( in = scalars->get_data(mycoord) );
  for (int ik=0; ik<k; ik++)
    in[ik] = (ik+1)*(double)mytid;
  // gamma distribution is gathering those scalars
  mpi_distribution *gamma = new mpi_gathered_distribution(decomp,k,1);
  auto gathered = std::shared_ptr<object>( new mpi_object(gamma) );
  CHECK( gamma->local_allocation()==k*ntids );

  mpi_kernel *gather = new mpi_kernel(scalars,gathered);
  gather->set_explicit_beta_distribution( gamma );
  gather->set_localexecutectx( (void*)&k );
  gather->set_localexecutefn( &veccopy );

  std::shared_ptr<task> gather_task;
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK_NOTHROW( gather->execute() );
  double *out = gathered->get_data(mycoord);
  for (int p=0; p<ntids; p++) {
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
  mpi_distribution *distributed = new mpi_block_distribution(decomp,1,-1);
  auto scalar = std::shared_ptr<object>( new mpi_object(distributed) );
  REQUIRE_NOTHROW( scalar->allocate() );
  double *scalar_data; REQUIRE_NOTHROW( scalar_data = scalar->get_data(mycoord) );
  *scalar_data = (double)mytid;

  // intermediate distribution is gathering those scalars replicated
  mpi_distribution *collected = new mpi_gathered_distribution(decomp);
  auto gathered = std::shared_ptr<object>( new mpi_object(collected) );

  // final distribution is redundant one scalar
  mpi_distribution *replicated     = new mpi_replicated_distribution(decomp);
  auto sum  = std::shared_ptr<object>( new mpi_object(replicated) );

  // first kernel: gather
  mpi_kernel *gather = new mpi_kernel(scalar,gathered);
  gather->set_explicit_beta_distribution( collected );

  std::shared_ptr<task> gather_task;
  gather->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( gather->last_dependency()->ensure_beta_distribution(gathered) );
  CHECK_NOTHROW( gather->split_to_tasks() );
  CHECK( gather->get_tasks().size()==1 );
  CHECK_NOTHROW( gather_task = gather->get_tasks().at(0) );
  CHECK_NOTHROW( gather_task->analyze_dependencies() );
  CHECK_NOTHROW( gather_task->last_dependency()->create_beta_vector(gathered) );
  CHECK_NOTHROW( gather_task->execute() );
  // let's look at the gathered data
  CHECK( gathered->volume(mycoord)==ntids );
  CHECK( gathered->get_orthogonal_dimension()==1 );
  {
    double *gathered_data = gathered->get_data(mycoord);
    for (int i=0; i<ntids; i++)
      CHECK( gathered_data[i]==(double)i );
  }
  
  // second kernel: local sum
  mpi_kernel *localsum = new mpi_kernel(gathered,sum);
  localsum->set_localexecutefn( &summing );
  std::shared_ptr<task> sum_task;

  const char *beta_strategy = "none";
  beta_strategy = "beta is collected distribution";
  localsum->set_explicit_beta_distribution( collected );
  INFO( beta_strategy );

  REQUIRE_NOTHROW( localsum->last_dependency()->ensure_beta_distribution(sum) );
  CHECK( localsum->get_beta_distribution()->volume(mycoord)==ntids );
  CHECK_NOTHROW( localsum->split_to_tasks() );
  CHECK( localsum->get_tasks().size()==1 );
  CHECK_NOTHROW( sum_task = localsum->get_tasks().at(0) );
  CHECK_NOTHROW( sum_task->analyze_dependencies() );
  CHECK_NOTHROW( sum_task->last_dependency()->create_beta_vector(sum) );

  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = sum_task->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_sender()==mycoord );
  CHECK_NOTHROW( sum_task->execute() );

  double *in = scalar->get_data(mycoord),*out = sum->get_data(mycoord);
  CHECK( (*out)==(ntids*(ntids-1)/2) );
}

TEST_CASE( "Analyze MPI non-blocking collective dependencies","[collective][nonblock][20]" ) {

  INFO( "mytid: " << mytid);
  // declare that we will use MPI collective routines
  mpi_decomposition *recomp;
  REQUIRE_NOTHROW( recomp = new mpi_decomposition(decomp) );
  REQUIRE_NOTHROW( recomp->set_collective_strategy( collective_strategy::MPI  ) );

  mpi_distribution
    *local_scalar = new mpi_scalar_distribution(recomp),
    *reduc_scalar = new mpi_replicated_distribution(recomp,1);
  auto local_value = std::shared_ptr<object>( new mpi_object(local_scalar) );
  auto reduc_value = std::shared_ptr<object>( new mpi_object(reduc_scalar) );
  double *sdata;
  REQUIRE_NOTHROW( sdata = local_value->get_data(mycoord) );
  sdata[0] = mytid;

  mpi_kernel *reduce;
  REQUIRE_NOTHROW( reduce = new mpi_kernel(local_value,reduc_value) );
  REQUIRE_NOTHROW( reduce->last_dependency()->set_is_collective() );
  REQUIRE_NOTHROW( reduce->set_explicit_beta_distribution
		   ( new mpi_gathered_distribution(recomp) ) );
  REQUIRE_NOTHROW( reduce->set_localexecutefn( &summing ) );
  REQUIRE_NOTHROW( reduce->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks; std::shared_ptr<task> tsk;
  REQUIRE_NOTHROW( tsks = reduce->get_tasks() );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  auto deps = tsk->get_dependencies(); int icol=0;
  for ( auto d : deps ) {
    REQUIRE_NOTHROW( icol += d->get_is_collective() );
  }
  CHECK( icol>0 );
  
  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  CHECK( msgs.size()==1 );

  REQUIRE_NOTHROW( reduce->execute() );
  double *rdata;
  REQUIRE_NOTHROW( rdata = reduc_value->get_data(mycoord) );
  CHECK( rdata[0]==Approx( ntids*(ntids-1)/2 ) );
}

TEST_CASE( "Inner product in three kernels","[collective][30]" ) {
  INFO( "mytid: " << mytid );

  // let's go for an irregular distribution
  mpi_distribution 
    *alpha = new mpi_block_distribution(decomp,10+2*mytid,-1),
    *local_scalar = new mpi_block_distribution(decomp,1,-1),
    *gathered_scalar = new mpi_gathered_distribution(decomp),
    *summed_scalar = new mpi_replicated_distribution(decomp);
  index_int my_first = alpha->first_index_r(mycoord).coord(0);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_size().at(0)==g );
  }
  auto x = std::shared_ptr<object>( new mpi_object(alpha) );
  auto y = std::shared_ptr<object>( new mpi_object(alpha) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  double *xdata,*ydata;
  REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
  for (index_int i=0; i<x->volume(mycoord); i++)
    xdata[i] = 2.;
  REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
  for (index_int i=0; i<y->volume(mycoord); i++)
    ydata[i] = my_first+i;

  // output object
  auto global_sum = std::shared_ptr<object>( new mpi_object(summed_scalar) );
  global_sum->set_name("inprod-value");

  const char *path;
  SECTION( "explicit intermediates" ) {
    path = "explicit intermediates";
    // intermediate objects
    std::shared_ptr<object> local_value;
    REQUIRE_NOTHROW( local_value = std::shared_ptr<object>( new mpi_object(local_scalar) ) );
    //REQUIRE_NOTHROW( gatheredvalue = new mpi_object(gathered_scalar) );

    // local product is local
    mpi_kernel
      *local_product = new mpi_kernel(x,local_value);
    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(x.get()) );
    REQUIRE_NOTHROW( local_product->add_in_object(y) );
    REQUIRE_NOTHROW( local_product->last_dependency()->set_explicit_beta_distribution(y.get()) );
    local_product->set_localexecutefn( &local_inner_product );
    REQUIRE_NOTHROW( local_product->analyze_dependencies() );
    
    mpi_kernel *gather_and_sum;

    // VLE can we turn this into an inner section?
    // REQUIRE_NOTHROW( gather_and_sum = new mpi_kernel(local_value,global_sum) );
    // REQUIRE_NOTHROW( gather_and_sum->set_explicit_beta_distribution(gathered_scalar) );
    // gather_and_sum->set_localexecutefn( &summing );

    // VLE here is how it is done in the mpi_innerproduct_kernel
    REQUIRE_NOTHROW( gather_and_sum = new mpi_reduction_kernel(local_value,global_sum) );

    REQUIRE_NOTHROW( gather_and_sum->analyze_dependencies() );
    REQUIRE_NOTHROW( local_product->execute() );
    REQUIRE_NOTHROW( gather_and_sum->execute() );
  }

  SECTION( "inner product kernel" ) {
    path = "inner product kernel";
    mpi_kernel *innerproduct = new mpi_innerproduct_kernel(x,y,global_sum);
    REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
    REQUIRE_NOTHROW( innerproduct->execute() );
  }
  INFO( "path is: " << path );

  index_int g = alpha->global_size().at(0); 
  double *zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_data(mycoord) );
  CHECK( global_sum->volume(mycoord)==1 );
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
  INFO( "mytid: " << mytid );
  INFO( "collectives are done " << path );

  decomposition *decomp ;
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(aa) );

  // let's go for an irregular distribution
  mpi_distribution 
    *alpha,  *local_scalar,  *gathered_scalar,  *summed_scalar ;
  REQUIRE_NOTHROW( alpha = new mpi_block_distribution(decomp,10+2*mytid,-1) );
  REQUIRE_NOTHROW( local_scalar = new mpi_block_distribution(decomp,1,-1) );
  REQUIRE_NOTHROW( gathered_scalar = new mpi_gathered_distribution(decomp) );
  REQUIRE_NOTHROW( summed_scalar = new mpi_replicated_distribution(decomp) );

  index_int my_first = alpha->first_index_r(mycoord).coord(0);
  {
    index_int g = ntids*10+ntids*(ntids-1);
    CHECK( alpha->global_size().at(0)==g );
  }
  std::shared_ptr<object> x,  y ;
  REQUIRE_NOTHROW( x = std::shared_ptr<object>( new mpi_object(alpha) ) );
  REQUIRE_NOTHROW( y = std::shared_ptr<object>( new mpi_object(alpha) ) );

  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( y->allocate() );
  x->set_name("inprod-x"); y->set_name("inprod-y");

  // set local data
  double *xdata,*ydata;
  REQUIRE_NOTHROW( xdata = x->get_data(mycoord) );
  for (index_int i=0; i<x->volume(mycoord); i++)
    xdata[i] = 2.;
  REQUIRE_NOTHROW( ydata = y->get_data(mycoord) );
  for (index_int i=0; i<y->volume(mycoord); i++)
    ydata[i] = my_first+i;
  { double s=0; for (int i=0; i<y->volume(mycoord); i++) s += xdata[i]*ydata[i];
    fmt::print("{}: local value={}\n",mycoord.as_string(),s); }

  // output object
  std::shared_ptr<object> global_sum;
  REQUIRE_NOTHROW( global_sum = std::shared_ptr<object>( new mpi_object(summed_scalar) ) );
  global_sum->set_name("inprod-value");

  mpi_kernel *innerproduct ;
  REQUIRE_NOTHROW( innerproduct = new mpi_innerproduct_kernel(x,y,global_sum) );
  REQUIRE_NOTHROW( innerproduct->analyze_dependencies() );
  REQUIRE_NOTHROW( innerproduct->execute() );

  index_int g = alpha->global_size().at(0); 
  double *zdata;
  REQUIRE_NOTHROW( zdata = global_sum->get_data(mycoord) );
  CHECK( global_sum->volume(mycoord)==1 );
  CHECK( zdata[0]==g*(g-1.) );

}
