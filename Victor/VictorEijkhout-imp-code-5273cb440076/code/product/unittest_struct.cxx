/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the MPI/OMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for communication structure
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "product_base.h"
#include "product_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"
#include "mpi_ops.h"

TEST_CASE( "Environment is proper","[environment][init][0]" ) {

  INFO( "decomposition: " << decomp->as_string() );

  int tmp;
  CHECK_NOTHROW( tmp = env->get_architecture()->nprocs() );
  CHECK_NOTHROW( tmp = env->get_architecture()->mytid() );

  // is the enviroment non-empty?
  CHECK( env->get_architecture()->nprocs() > 0 );

  // check the mpi part of the environment
  CHECK( mpi_nprocs>0 );
  CHECK( mpi_nprocs==env->get_architecture()->nprocs() );
  CHECK( mytid==env->get_architecture()->mytid() );

  // check the omp part of the environment
  architecture *omp_arch;
  REQUIRE_NOTHROW( omp_arch = arch->get_embedded_architecture() );
  REQUIRE( omp_arch!=nullptr );
  CHECK( omp_nprocs==omp_arch->nprocs() );

  // for completeness
  int *tids = new int[1000]; //[mpi_nprocs];
  MPI_Allgather(&mytid,1,MPI_INT,tids,1,MPI_INT,comm);
  for (int p=0; p<mpi_nprocs; p++) 
    REQUIRE(tids[p]==p);

  {
    int nlocal = 10; product_distribution *d;
    CHECK_NOTHROW( d = new product_block_distribution(decomp,nlocal,-1) );
    CHECK( d->global_volume()==nlocal*mpi_nprocs );
    CHECK( d->volume(mycoord)==nlocal );

    std::shared_ptr<object> v;
    REQUIRE_NOTHROW( v = std::shared_ptr<object>( new product_object(d) ) );
    int nt;
    REQUIRE_NOTHROW( nt = v->domains_volume() );
    CHECK( nt == mpi_nprocs );
  }
};

TEST_CASE( "test product structure","[embed][distribution][2]" ) {

  INFO( "mytid=" << mytid );

  index_int nlocal = 12,nodelocal = nlocal*omp_nprocs, nglobal=nodelocal*mpi_nprocs;
  auto no_op = ioperator("none");
  decomposition *embedded_decomp;
  product_distribution *block = new product_block_distribution(decomp,nglobal);
  REQUIRE_NOTHROW( embedded_decomp = decomp->get_embedded_decomposition() );
  REQUIRE_NOTHROW( embedded_decomp = block->get_embedded_decomposition() );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nodelocal );
  CHECK( my_last==(mytid+1)*nodelocal-1 );

  auto xdata = new double[nodelocal];
  for (int i=0; i<nodelocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  auto
    xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  REQUIRE_NOTHROW( embedded_decomp = xvector->get_embedded_decomposition() );
  REQUIRE_NOTHROW( embedded_decomp = yvector->get_embedded_decomposition() );
  xvector->set_name("33x"); yvector->set_name("33y");

  product_kernel *scale = new product_kernel(xvector,yvector);
  scale->add_sigma_operator( no_op );
  REQUIRE_NOTHROW( scale->last_dependency()->set_name("33dep") );
  REQUIRE_NOTHROW( embedded_decomp = scale->get_out_object()->get_embedded_decomposition() );

  std::shared_ptr<task> scale_task;
  const char *path;
  int *saved_omp_nprocs = new int{omp_nprocs};
  index_int *saved_nodelocal = new index_int{nodelocal};
  SECTION( "proc count" ) { path = "proc count";
    scale->set_name("testp");
    CHECK_NOTHROW
      ( scale->set_localexecutefn
	( [saved_omp_nprocs] ( kernel_function_args ) -> void {
	  test_nprocs( kernel_function_call, *saved_omp_nprocs ); } ) );
  }
  SECTION( "distr proc count" ) {path = "distr proc count";
    scale->set_name("testp");
    CHECK_NOTHROW
      ( scale->set_localexecutefn
	( [saved_omp_nprocs] ( kernel_function_args ) -> void {
	  test_distr_nprocs( kernel_function_call, *saved_omp_nprocs ); } ) );
  }
  SECTION( "global size" ) {path = "global size";
    scale->set_name("tests");
    CHECK_NOTHROW
      ( scale->set_localexecutefn
	( [saved_nodelocal] ( kernel_function_args ) -> void {
	  test_globalsize( kernel_function_call,*saved_nodelocal ); } ) );
  }
  INFO( "path: " << path );
  CHECK_NOTHROW( scale->analyze_dependencies() );
  CHECK_NOTHROW( scale->execute() );
}

TEST_CASE( "Analyze single interval","[index][structure][12]" ) {
  int localsize = 5;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *dist = new product_distribution(decomp); 
  parallel_structure *pstruct;
  REQUIRE_NOTHROW( pstruct = dynamic_cast<parallel_structure*>(dist) );
  CHECK_NOTHROW( pstruct->create_from_global_size(localsize*mpi_nprocs) );
  std::vector<message*> mm;
  message *m; std::shared_ptr<multi_indexstruct> s,segment;

  SECTION( "find an exact processor interval" ) {
    REQUIRE_NOTHROW( segment = std::shared_ptr<multi_indexstruct>
		     ( new multi_indexstruct( std::shared_ptr<indexstruct>
					      ( new contiguous_indexstruct(0,localsize-1) ) ) ) );
    CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( mm.size()==1 );
    m = mm.at(0);
    CHECK( m->get_sender().coord(0)==0 );
    CHECK( m->get_receiver().equals(mycoord) );
    s = m->get_global_struct();
    CHECK( s->first_index_r()==domain_coordinate_zero(1) );
    CHECK( s->last_index_r()==last_coordinate+1-1 );
  }

  SECTION( "find a sub interval" ) {
    REQUIRE_NOTHROW
      ( segment = std::shared_ptr<multi_indexstruct>
	( new multi_indexstruct
	  ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(1,localsize-2) ) ) ) );
    //segment = new multi_indexstruct( new contiguous_indexstruct(1,localsize-2) );
    CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( mm.size()==1 );
    m = mm.at(0);
    CHECK( m->get_sender().coord(0)==0 );
    CHECK( m->get_receiver().equals(mycoord) );
    REQUIRE_NOTHROW( s = m->get_global_struct() );
    REQUIRE( s!=nullptr );
    CHECK( s->first_index_r()==1 );
    CHECK( s->last_index_r()==last_coordinate+1-2 );
  }

  SECTION( "something with nontrivial intersection" ) {
    if (mpi_nprocs<2) return;

    INFO( "mytid=" << mytid );
    REQUIRE_NOTHROW( segment = std::shared_ptr<multi_indexstruct>
		     ( new multi_indexstruct
		       ( std::shared_ptr<indexstruct>
			 ( new contiguous_indexstruct(0,localsize) ) ) ) );
    //segment = new multi_indexstruct( new contiguous_indexstruct(0,localsize) ); // one element beyond the 1st proc

    // first manually
    for (int ip=0; ip<mpi_nprocs; ip++) {
      int p = (mytid+ip)%mpi_nprocs; // start with self first
      auto pcoord = decomp->coordinate_from_linear(p);
      INFO( "intersect with p=" << p );
      std::shared_ptr<multi_indexstruct> intersect;
      REQUIRE_NOTHROW( intersect = segment->intersect( pstruct->get_processor_structure(pcoord) ) );
      if (p==0) {
	CHECK( !intersect->is_empty() );
      } else if (p==1) {
	CHECK( !intersect->is_empty() );
      } else {
	CHECK( intersect->is_empty() );
      }
    }

    // now with the system call
    CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
    CHECK( mm.size()==2 );
    for ( auto m : mm ) { // int im=0; im<2; im++) { // we don't insist on ordering
      CHECK( m->get_receiver().equals(mycoord) );
      CHECK( ( (m->get_sender().coord(0)==0) || (m->get_sender().coord(0)==1) ) );
      auto range = m->get_global_struct();
      if (m->get_sender().coord(0)==0) {
	CHECK( range->first_index_r()==domain_coordinate_zero(1) );
	CHECK( range->last_index_r()==last_coordinate+1-1 );
      } else {
	CHECK( range->first_index_r()==last_coordinate+1 );
	CHECK( range->last_index_r()==last_coordinate+1 );
      }
    }
  }
}

TEST_CASE( "Analyze one dependency","[operate][dependence][14]") {

  INFO( "mytid=" << mytid );

  CHECK( omp_nprocs>=1 );
  int localsize = 100*omp_nprocs; // node local size
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *alpha = 
    new product_block_distribution(decomp,localsize,-1);
  std::shared_ptr<multi_indexstruct> alpha_block;
  std::vector<message*> mm; message *m;
  index_int my_first,my_last;
  std::shared_ptr<multi_indexstruct> segment;

  //  for (int mytid=0; mytid<env->nprocs(); mytid++) {
  {
    my_first = alpha->first_index_r(mycoord).coord(0),
    my_last = alpha->last_index_r(mycoord).coord(0);
    CHECK( (my_last-my_first+1)==localsize );

    // right bump
    auto shiftop = ioperator(">=1");
    CHECK( shiftop.is_right_shift_op() );
    CHECK( !shiftop.is_modulo_op() );
    alpha_block = alpha->get_processor_structure(mycoord); // should be from alpha/gamma
    CHECK( alpha_block->first_index_r()==my_first );
    CHECK( alpha_block->last_index_r()==my_last );
    segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure());
    CHECK( segment->first_index_r()==my_first+1 );
    if (mytid<env->get_architecture()->nprocs()-1)
      CHECK( segment->last_index_r()==my_last+1 );
    else
      CHECK( segment->last_index_r()==my_last );
    CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
    if (mytid<env->get_architecture()->nprocs()-1) {
      CHECK( mm.size()==2 );
    } else {
      CHECK( mm.size()==1 );
    }
    m = mm.at(0); 
    CHECK( m->get_sender().coord(0)==mytid );
    CHECK( m->get_global_struct()->volume()==localsize-1 );
    CHECK( m->get_global_struct()->first_index_r()==my_first+1 );
    if (mytid<env->get_architecture()->nprocs()-1) {
      m = mm.at(1);
      CHECK( m->get_sender().coord(0)==mytid+1 );
      CHECK( m->get_global_struct()->volume()==1 );
      CHECK( m->get_global_struct()->first_index_r()==my_last+1 );
    }

    // left bump
    shiftop = ioperator("<=1");
    CHECK( shiftop.is_left_shift_op() );
    CHECK( !shiftop.is_modulo_op() );
    alpha_block = alpha->get_processor_structure(mycoord);
    CHECK( alpha_block->volume()==localsize );
    segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure());
    CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
    if (mytid>0) {
      CHECK( mm.size()==2 );
    } else {
      CHECK( mm.size()==1 );
    }
    if (mytid>0) {
      for (int im=0; im<2; im++) {
	m = mm.at(im);
	if ( m->get_sender().coord(0)==mytid-1 ) {
	  CHECK( m->get_global_struct()->volume()==1 );
	} else if ( m->get_sender().coord(0)==mytid ) {
	  CHECK( m->get_global_struct()->volume()==localsize-1 );
	} else {
	  CHECK( 1==0 );
	}
      }
    } else {
      m = mm.at(0);
      CHECK( m->get_sender().coord(0)==mytid );
      CHECK( m->get_global_struct()->volume()==localsize-1 );
    }
  }
    //  } // end of mytid loop
}

TEST_CASE( "Analyze one dependency modulo","[operate][dependence][modulo][16]") {

  INFO( "mytid=" << mytid );

  int localsize = 100*omp_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  mpi_distribution *alpha = 
    new mpi_block_distribution(decomp,localsize,-1);
  ioperator shiftop; std::shared_ptr<multi_indexstruct> alpha_block;
  std::vector<message*> mm; message *m;
  index_int my_first,my_last,
    gsize = alpha->global_volume();
  std::shared_ptr<multi_indexstruct> segment,halo;

  //for (int mytid=0; mytid<env->get_architecture()->nprocs(); mytid++) { ????
  {

    auto my_first = alpha->first_index_r(mycoord),
      my_last = alpha->last_index_r(mycoord);

    SECTION( "right modulo" ) {
      shiftop = ioperator(">>1");
      CHECK( shiftop.is_right_shift_op() );
      CHECK( shiftop.is_modulo_op() );

      // my block in my alpha distribution
      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->volume()==localsize );
      CHECK( alpha_block->first_index_r()==my_first );
      CHECK( alpha_block->last_index_r()==my_last );
      CHECK( (alpha_block->last_index_r()-alpha_block->first_index_r()+1)==localsize );

      // operated alpha block is shifted, can stick out beyond
      segment = alpha_block->operate(shiftop);
      halo = alpha_block->struct_union(segment);
      CHECK( segment->first_index_r()==my_first+1 );
      CHECK( segment->last_index_r()==my_last+1 );
      if (mytid==env->get_architecture()->nprocs()-1) {
	CHECK( segment->last_index_r()>alpha_block->last_index_r() );
      }

      // the halo is the union my the alpha block and the operated alpha block
      CHECK( halo->first_index_r()==my_first );
      CHECK( halo->last_index_r()==my_last+1 );

      // it takes two messages to construct the halo
      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,halo ) );
      CHECK( mm.size()==2 );
      if ( env->get_architecture()->nprocs()>1 ) {
	for (int imsg=0; imsg<2; imsg++) {
	  auto m = mm.at(imsg);
	  auto gstruct = m->get_global_struct();
	  INFO( m->get_sender().as_string() << " sends "
		<< gstruct->first_index_r().as_string() << "--"
		<< gstruct->last_index_r().as_string() );
	  CHECK( m->get_receiver().equals(mycoord) );
	  if (m->get_sender().coord(0)==mytid) {
	    CHECK( gstruct->volume()==localsize-1 );
	    CHECK( gstruct->first_index_r()==my_first+1 );
	    CHECK( m->get_local_struct()->first_index_r()==1 );
	  } else {
	    CHECK( m->get_sender().coord(0)==(mytid+1)%env->get_architecture()->nprocs() );
	    CHECK( gstruct->volume()==1 );
	    CHECK( gstruct->first_index_r()==(my_last+1)%gsize );
	    CHECK( m->get_local_struct()->first_index_r()==last_coordinate+1 );
	  }
	}
      } else printf("skipping part of [16] on one proc\n");
    }

    SECTION( "left modulo" ) {
      // this example does modulo truncate, which I'm not sure is ever needed
      shiftop = ioperator("<<1");
      CHECK( shiftop.is_left_shift_op() );
      CHECK( shiftop.is_modulo_op() );

      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->volume()==localsize );

      segment = alpha_block->operate(shiftop); //,alpha->get_enclosing_structure());
      halo = alpha_block->struct_union(segment);
      CHECK( halo->first_index_r()==my_first-1 );
      CHECK( halo->last_index_r()==my_last );
      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,halo ) );
      CHECK( mm.size()==2 );
      for ( auto m : mm ) { // (int imsg=0; imsg<2; imsg++) {
    	//message *m = (*mm)[imsg];
    	if (m->get_sender().coord(0)==mytid) {
    	  CHECK( m->get_global_struct()->volume()==localsize-1 );
    	  CHECK( m->get_global_struct()->first_index_r()==my_first );
    	  CHECK( m->get_local_struct()->first_index_r()==1 );
    	} else {
    	  CHECK( m->get_sender().coord(0)==MOD(mytid-1,env->get_architecture()->nprocs()) );
    	  CHECK( m->get_global_struct()->volume()==1 );
    	  CHECK( m->get_global_struct()->first_index_r()==(my_first-1+gsize)%gsize );
    	  CHECK( m->get_local_struct()->first_index_r()==domain_coordinate_zero(1) );
    	}
      }
    }
  }
}

TEST_CASE( "Task dependencies threepoint bump","[task][message][object][21]" ) {

  // create distributions and objects for threepoint combination

  INFO( "mytid=" << mytid );
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">=1"),
    left_shift  = ioperator("<=1");

  index_int gsize = 10*omp_nprocs*mpi_nprocs;
  product_distribution *d1 = 
    new product_block_distribution(decomp,gsize);

  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new product_object(d1) ) );
  CHECK( r1->global_volume()==gsize );
  CHECK_NOTHROW( r1->set_name("21result") );
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new product_object(d1) ) );
  CHECK_NOTHROW( o1->set_name("21in") );

  // declare the sigma vectors
  signature_function *sigma_opers;
  REQUIRE_NOTHROW( sigma_opers = new signature_function() );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( no_op ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( left_shift ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( right_shift ) );

  parallel_structure *beta_struct;  product_distribution *beta_dist;
  REQUIRE_NOTHROW( beta_struct = sigma_opers->derive_beta_structure(d1,d1->get_enclosing_structure()) );
  REQUIRE_NOTHROW( beta_struct->set_type( beta_struct->infer_distribution_type() ) );
  INFO( "beta parallel structure: " << beta_struct->as_string() );
  CHECK( beta_struct->volume(mycoord)>0 );
  INFO( "beta structure type: " << beta_struct->type_as_string() );
  CHECK_NOTHROW( beta_dist = new product_distribution(beta_struct) );
  CHECK( beta_dist->volume(mycoord)>0 );
  CHECK_NOTHROW( beta_dist->set_name("22beta_dist") );
  INFO( "beta distribution type: " << beta_dist->type_as_string() );
  CHECK( beta_dist->has_defined_type() );

  std::shared_ptr<task> combine_task;
  const char *path;
  SECTION( "make kernel" ) {
    path = "by kernel";
    kernel *combine;
    REQUIRE_NOTHROW( combine = new product_kernel(o1,r1) );
    combine->set_name("combine-kernel-21");
    REQUIRE_NOTHROW( combine->set_explicit_beta_distribution( beta_dist ) );
    CHECK_NOTHROW( combine->last_dependency()->set_name("22xbeta") );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
  }
  // SECTION( "make task" ) {
  //   path = "by task";
  //   CHECK_NOTHROW( combine_task = new product_task(0,mytid,o1,r1) );
  //   CHECK_NOTHROW( combine_task->set_explicit_beta_distribution( beta_dist ) );
  //   CHECK_NOTHROW( combine_task->last_dependency()->set_name("22xbeta") );
  //   //CHECK_NOTHROW( combine_task->set_beta_distribution( 0,beta_dist ) );
  //   REQUIRE_NOTHROW( combine_task->analyze_dependencies() );
  // }

  REQUIRE_NOTHROW( combine_task->last_dependency()->create_beta_vector(r1) );
  REQUIRE_NOTHROW( combine_task->derive_receive_messages(/*0,mytid*/) );
  INFO( "path: " << path );
}

TEST_CASE( "Task send structure threepoint modulo","[task][message][object][modulo][24]" ) {

  // create distributions and objects for threepoint combination

  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">>1"),
    left_shift  = ioperator("<<1");

  product_distribution *d1 = 
    new product_block_distribution(decomp,10*omp_nprocs*mpi_nprocs);

  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new product_object(d1) ) );
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new product_object(d1) ) );

  // declare the beta vectors
  signature_function *sigma_opers = new signature_function();
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( no_op ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( left_shift ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( right_shift ) );

  std::shared_ptr<task> combine_task;
  kernel *combine = new product_kernel(o1,r1);

  SECTION( "spell out the beta construction" ) {
    parallel_structure *beta_struct;
    REQUIRE_NOTHROW( beta_struct = sigma_opers->derive_beta_structure(d1,r1->get_enclosing_structure()) );
    product_distribution *beta_dist;
    CHECK_NOTHROW( beta_dist = new product_distribution(beta_struct) );
    CHECK_NOTHROW( combine->set_explicit_beta_distribution( beta_dist ) );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE( combine->get_tasks().size()==1 );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
  }
  // SECTION( "in one fell swoop" ) {
  //   CHECK_NOTHROW( combine_task->ensure_beta_distribution() );
  // REQUIRE_NOTHROW( combine_task->last_dependency()->create_beta_vector(r1) );
  // REQUIRE_NOTHROW( combine_task->derive_receive_messages(/*0,mytid*/) );
  // // and for the mirror, what is going out
  // CHECK_NOTHROW( combine_task->derive_send_messages(/*0,mytid*/) );
  // }

  // see what is coming on
  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = combine_task->get_send_messages() );
  CHECK( msgs.size()==3 );

  for (auto m : msgs ) { //->begin(); m!=msgs->end(); ++m) {
    auto glb = m->get_global_struct();
    auto sf = glb->first_index_r(),
      sl = glb->last_index_r();
    CHECK( ( sf>=d1->first_index_r(mycoord)-1 && sl<=d1->last_index_r(mycoord)+1 ) );
    sf = m->get_local_struct()->first_index_r();
    sl = m->get_local_struct()->last_index_r();
    CHECK( ( sf.coord(0)>=0 && sl.coord(0)<d1->volume(m->get_sender()) ) );
  }
}

TEST_CASE( "Halo object for task modulo","[task][halo][modulo][25]" ) {

  // create distributions and objects for threepoint combination
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">>1"),
    left_shift  = ioperator("<<1");
  index_int nodesize = 10*omp_nprocs;
  product_distribution *d1 = 
    new product_block_distribution(decomp,nodesize*mpi_nprocs);
  std::shared_ptr<object> o1,r1;
  const char *r1name = "result_vector_1",
    *o1name = "origin_vector_1";
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new product_object(d1) ) ); // result vector
  r1->set_name( r1name );
  CHECK( r1->get_name().compare(r1name)==0 );

  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new product_object(d1) ) ); // origin vector
  o1->set_name( o1name );

  //  parallel_indexstruct *beta_struct;
  std::shared_ptr<object> halo;
  kernel *combine; std::shared_ptr<task> combine_task;
  CHECK_NOTHROW( combine = new product_kernel(o1,r1) );
  signature_function *beta = new signature_function();

  SECTION( "no-op" ) {
    REQUIRE_NOTHROW( combine->last_dependency()->add_sigma_operator( no_op ) );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
    REQUIRE_NOTHROW( halo = combine_task->get_beta_object(0) );

    REQUIRE( combine_task->get_out_object()!=nullptr );
    CHECK( r1->get_name().compare(r1name)==0 ); // just making sure it's not been nixed
    CHECK( combine_task->get_out_object()->get_name().compare(r1name)==0 );
    CHECK( combine_task->get_out_object()->volume(mycoord)==nodesize );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==nodesize );
  }
  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( left_shift ) );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
    REQUIRE_NOTHROW( halo = combine_task->get_beta_object(0) );

    CHECK( combine_task->get_out_object()!=nullptr );
    CHECK( combine_task->get_out_object()->volume(mycoord)==nodesize );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==(nodesize+1) );
  }
  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
    REQUIRE_NOTHROW( halo = combine_task->get_beta_object(0) );

    CHECK( combine_task->get_out_object()!=nullptr );
    CHECK( combine_task->get_out_object()->volume(mycoord)==nodesize );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==(nodesize+1) );
  }
  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine->last_dependency()->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->analyze_dependencies() );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(0) );
    REQUIRE_NOTHROW( halo = combine_task->get_beta_object(0) );

    CHECK( combine_task->get_out_object()!=nullptr );
    CHECK( combine_task->get_out_object()->volume(mycoord)==nodesize );
    CHECK( halo!=nullptr );
    CHECK( halo->volume(mycoord)==(nodesize+2) );
  }
}

TEST_CASE( "Halo object for kernel modulo","[kernel][halo][modulo][27]" ) {

  // create distributions and objects for threepoint combination
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">>1"),
    left_shift  = ioperator("<<1");
  index_int nodesize = 10*omp_nprocs;
  product_distribution *d1 = 
    new product_block_distribution(decomp,nodesize*mpi_nprocs);
  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new product_object(d1) ) ); // result vector
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new product_object(d1) ) ); // origin vector

  product_kernel *combine;
  std::shared_ptr<task> tsk;
  CHECK_NOTHROW( combine = new product_kernel(o1,r1) );

  SECTION( "no-op" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==nodesize );
  }

  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( left_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==(nodesize+1) );
  }

  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==(nodesize+1) );
  }

  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==(nodesize+2) );
  }
}

TEST_CASE( "Halo object for kernel bump","[kernel][halo][28]" ) {

  INFO( "mytid=" << mytid );

  // create distributions and objects for threepoint combination
  auto
    no_op       = ioperator("none"),
    right_shift = ioperator(">=1"),
    left_shift  = ioperator("<=1");
  product_distribution *d1 = 
    new product_block_distribution(decomp,10*mpi_nprocs);
  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new product_object(d1) ) ); // result vector
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new product_object(d1) ) ); // origin vector

  product_kernel *combine;
  CHECK_NOTHROW( combine = new product_kernel(o1,r1) );
  std::shared_ptr<task> tsk;
  index_int checksize;

  SECTION( "no-op" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK( tsk->get_beta_object(0)!=nullptr );
    CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
  }

  SECTION( "no-op and left" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( left_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    std::shared_ptr<object> h;
    REQUIRE_NOTHROW( h = tsk->get_beta_object(0) );
    REQUIRE_NOTHROW( checksize = h->volume(mycoord) );
    if (d1->is_first_proc(mytid)) {
      CHECK( checksize==10 );
    } else {
      CHECK( checksize==11 );
    }
  }

  SECTION( "no-op and right" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    if (d1->is_last_proc(mytid)) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==10 );
    } else {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
    }
  }

  SECTION( "no-op and left and right" ) {
    CHECK_NOTHROW( combine->add_sigma_operator( no_op ) );
    CHECK_NOTHROW( combine->add_sigma_operator( left_shift ) );
    CHECK_NOTHROW( combine->add_sigma_operator( right_shift ) );
    REQUIRE_NOTHROW( combine->last_dependency()->ensure_beta_distribution(o1) );
    CHECK_NOTHROW( combine->split_to_tasks() );
    CHECK( combine->get_tasks().size()==1 );
    CHECK_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
    CHECK_NOTHROW( tsk = combine->get_tasks().at(0) );
    REQUIRE( tsk->get_beta_object(0)!=nullptr );
    if (d1->is_last_proc(mytid) || d1->is_first_proc(mytid)) {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==11 );
    } else {
      CHECK( tsk->get_beta_object(0)->volume(mycoord)==12 );
    }
  }
}

TEST_CASE( "Origin task execute on local data","[mpi][object][task][kernel][execute][29]" ) {

  int ss=10, s = omp_nprocs*ss;
  product_distribution *block = new product_block_distribution(decomp,s*mpi_nprocs);
  CHECK( block->volume(mycoord)==s );
  auto vector = std::shared_ptr<object>( new product_object(block) );
  double *data;
  REQUIRE_NOTHROW( data = vector->get_data(mycoord) );
  INFO( "object has data @ " << (long)data );
  vector->set_name("product29 vector");
  
  // define a task
  kernel *k;
  REQUIRE_NOTHROW( k = new product_origin_kernel(vector) );
  REQUIRE_NOTHROW( k->split_to_tasks() );
  std::shared_ptr<task> t;  
  REQUIRE_NOTHROW( t = k->get_tasks().at(0) );
  CHECK( t->has_type_origin() );
  REQUIRE_NOTHROW( data = t->get_out_object()->get_data(mycoord) );
  INFO( "task out object has data @ " << (long)data );
  CHECK( t->get_out_object()->volume(mycoord)==s );
  REQUIRE_NOTHROW( t->set_localexecutefn( &vector_gen ) );

  // analyze
  REQUIRE_NOTHROW( k->analyze_dependencies() );
  //  REQUIRE( k->get_has_been_analyzed() );
  REQUIRE( t->get_has_been_analyzed() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = ( dynamic_cast<product_task*>(t.get()) )->get_omp_tasks() );
  CHECK( tsks.size()==omp_nprocs );
  int sump=0;
  for (int p=0; p<omp_nprocs; p++) {
    processor_coordinate pcoord;
    decomposition *ompdecomp;
    REQUIRE_NOTHROW( ompdecomp = decomp->get_embedded_decomposition() );
    REQUIRE_NOTHROW( pcoord = ompdecomp->coordinate_from_linear(p) );
    std::shared_ptr<object> task_out;
    INFO( "proc " << p << " out of " << omp_nprocs << "; node local size=" << s );
    REQUIRE_NOTHROW( task_out = tsks.at(p)->get_out_object() );
    CHECK( task_out->volume(pcoord)==ss );
    REQUIRE_NOTHROW( sump += tsks.at(p)->get_domain().coord(0) );
  }
  CHECK( sump==( omp_nprocs*(omp_nprocs-1)/2 ) );

  // execute
  REQUIRE_NOTHROW( k->execute() );

  // did we corrupt something?
  REQUIRE_NOTHROW( tsks = ( dynamic_cast<product_task*>(t.get()) )->get_omp_tasks() );
  REQUIRE( tsks.size()==omp_nprocs );

  // analyze output, first of embedded task
  for (int p=0; p<omp_nprocs; p++) {
    std::shared_ptr<object> task_out;
    REQUIRE_NOTHROW( task_out = tsks.at(p)->get_out_object() );
    double *tdata;
    REQUIRE_NOTHROW( tdata = task_out->get_raw_data() );
    for (int i=0; i<ss; i++) {
      INFO( "process: " << mytid << "; task = " << p << "; i=" << i );
      CHECK( tdata[p*ss+i]==p+.5 );
    }
  }

  // then of the main task
  REQUIRE_NOTHROW( data = vector->get_data(mycoord) );
  INFO( "after exec: object has data @ " << (long)data );
  for (int p=0; p<omp_nprocs; p++) {
    for (int i=0; i<ss; i++) {
      INFO( "thread=" << p << ", i: " << i );
      CHECK( data[p*ss+i]==p+.5 );
    }
  }
}

TEST_CASE( "Task execute on external data","[mpi][object][task][kernel][execute][30]" ) {

  // same test as before but now with externally allocated data
  int ss=20, s = omp_nprocs*ss;
  double *xdata = new double[s];
  product_distribution *block = new product_block_distribution(decomp,s*mpi_nprocs);
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) );

  // define a task
  product_kernel *xk = new product_kernel(xvector);
  auto xt = std::shared_ptr<task>( new product_task(mycoord,xk) );
  CHECK( xt->has_type_origin() );
  signature_function *f;
  REQUIRE_NOTHROW( f = new signature_function() );
  REQUIRE_NOTHROW( f->set_type_local() );
  xt->set_localexecutefn( &vector_gen );

  // analyze
  REQUIRE_NOTHROW( xt->analyze_dependencies() );

  // execute
  xt->execute();

  double *data;
  REQUIRE_NOTHROW( data = xvector->get_data(mycoord) );
  for (int p=0; p<omp_nprocs; p++) {
    for (int i=0; i<ss; i++) {
      INFO( "thread=" << p << ", i: " << i );
      CHECK( data[p*ss+i]==p+.5 );
    }
  }

}

TEST_CASE( "Shift left bump kernel, execute","[task][kernel][halo][execute][32]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  double *xdata = new double[nlocal];
  auto 
    xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  xvector->set_name("36x"); yvector->set_name("32y");
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("36shift");
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator(  ioperator(">=1")  );
  shift->set_localexecutefn( &vecshiftleftbump );
  
  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  REQUIRE_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift->analyze_dependencies() );
  CHECK_NOTHROW( shift->last_dependency()->set_name("36mpi-dependency") );

  std::shared_ptr<object> halo;
  CHECK_NOTHROW( halo = shift->get_beta_object(0) );
  {
    INFO( "investigating halo <<" << halo->get_name() << ">>" );
    // " of type " << halo->architecture::as_string()
    INFO( "halo distribution: " << halo->as_string() );
    INFO( "vector distribution: " << xvector->as_string() );
    index_int halo_first, halo_last;
    REQUIRE_NOTHROW( halo_first = halo->first_index_r(mycoord).coord(0) );
    REQUIRE_NOTHROW( halo_last = halo->last_index_r(mycoord).coord(0) );
    CHECK( halo_first==yvector->first_index_r(mycoord).coord(0));
    if (mytid==mpi_nprocs-1)
      CHECK( halo_last==yvector->last_index_r(mycoord).coord(0));
    else
      CHECK( halo_last==yvector->last_index_r(mycoord).coord(0)+1);
  }

  CHECK_NOTHROW( shift->execute() );

  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  if (mytid==mpi_nprocs-1) { // there is no right halo
    for (int i=0; i<nlocal; i++) {
      INFO( "i=" << i << " hi=" << halo_data[i] );
      CHECK( halo_data[i] == Approx(pointfunc33(i,my_first)) );
    }
  } else { // there is a right halo, so we test one more
    for (int i=0; i<nlocal+1; i++) {
      INFO( "i=" << i << " hi=" << halo_data[i] );
      CHECK( halo_data[i] == Approx(pointfunc33(i,my_first)) );
    }
  }

  /*
   * Inspect the OMP queue in the local MPI task
   */
  std::shared_ptr<algorithm> queue;
  //REQUIRE_NOTHROW( queue = dynamic_cast<product_task*>(shift_task.get())->get_node_queue() );
  REQUIRE_NOTHROW( queue = shift_task->get_node_queue() );
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    auto thr = t->get_domain();
    INFO( "process " << mytid << ", thread " << thr.as_string() );	  
    std::shared_ptr<object> omp_in,omp_out; double *data; distribution *halodist;
    index_int thr_first, thr_last, offset; double *indata,*outdata;
    REQUIRE_NOTHROW( omp_out = t->get_out_object() );
    INFO( "output object: " << omp_out->get_name() );
    REQUIRE_NOTHROW( outdata = omp_out->get_data(mycoord) );
    thr_first = omp_out->first_index_r(thr).coord(0);
    thr_last = omp_out->last_index_r(thr).coord(0);
    if (t->has_type_origin()) { // MPI halo
      if (thr.coord(0)==0) // special cases: all other thread endpoints are somewhere in between
	CHECK( thr_first==my_first );
      else if (thr.coord(0)==omp_nprocs-1) {
	if (mytid==mpi_nprocs-1)
	  CHECK( thr_last==my_last );
	else
	  CHECK( thr_last==my_last+1 );
      }
      offset = omp_out->get_numa_structure()->first_index_r()[0]
	- omp_out->get_global_structure()->first_index_r()[0];
      for (index_int i=thr_first; i<=thr_last; i++)
	if (!(mytid==mpi_nprocs-1&&thr_first==omp_nprocs-1)) // undefined case
	  CHECK( outdata[i-offset]==Approx( pointfunc33(i,0) ) );
    } else { // type compute
      // the OMP halo looks just like the MPI one
      REQUIRE_NOTHROW( omp_in = t->get_in_object(0) );
      thr_first = omp_in->first_index_r(thr).coord(0);
      thr_last = omp_in->last_index_r(thr).coord(0);
      if (thr.is_zero()) // first thread: aligned on mpiproc 0, otherwise stick out to the left
	CHECK( thr_first==my_first );
      if (thr.coord(0)==omp_nprocs-1) { // thread last, depends on mpi being last
	if (mytid==mpi_nprocs-1)
	  CHECK( thr_last==my_last );
	else
	  CHECK( thr_last==my_last+1 );
      }
      if (mytid==mpi_nprocs-1)
	CHECK( omp_in->global_volume()==nlocal );
      else
	CHECK( omp_in->global_volume()==nlocal+1 );
      REQUIRE_NOTHROW( indata = omp_in->get_data(mycoord) );
      offset = omp_in->get_numa_structure()->first_index_r()[0]
	- omp_in->get_global_structure()->first_index_r()[0];
      for (index_int i=thr_first; i<=thr_last; i++)
	if (!(mytid==0&&thr_first==0)) // undefined case
	  CHECK( indata[i-offset]==Approx( pointfunc33(i,0) ) );
      if (1) { // OMP output looks like global output
	thr_first = omp_out->first_index_r(thr).coord(0);
	thr_last = omp_out->last_index_r(thr).coord(0);
	if (thr.is_zero()) // first thread left aligned with mpi
	  CHECK( thr_first==my_first );
	if (thr.coord(0)==omp_nprocs-1) // last thread right aligned with mpi
	  CHECK( thr_last==my_last );
	INFO( "computing output offset as numa=" << omp_out->get_numa_structure()->as_string() <<
	      " in global=" << omp_out->get_global_structure()->as_string() );
	offset = omp_out->get_numa_structure()->first_index_r()[0] - omp_out->get_global_structure()->first_index_r()[0];
	for (index_int i=thr_first; i<=thr_last; i++) {
	  INFO( "i=" << i << " with offset " << offset );
	  if (i!=omp_out->get_global_structure()->last_index_r()[0])
	    CHECK( outdata[i-offset]==Approx( pointfunc33(i+1,0) ) );
	}
      }
    }
  }

  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int len = nlocal;
    if (mytid==mpi_nprocs-1) nlocal--;
    for (int i=0; i<nlocal; i++) {
      INFO( "i=" << i << " yi=" << ydata[i] );
      CHECK( ydata[i] == Approx(pointfunc33(i+1,my_first)) );
    }
  }
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  INFO( "mytid=" << mytid );

  index_int nlocal = 12,nodelocal = nlocal*omp_nprocs, nglobal=nodelocal*mpi_nprocs;
  auto no_op =  ioperator("none") ;
  product_distribution *block = new product_block_distribution(decomp,nglobal);
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nodelocal );
  CHECK( my_last==(mytid+1)*nodelocal-1 );

  double *xdata = new double[nodelocal];
  for (int i=0; i<nodelocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  auto
    xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  xvector->set_name("33x"); yvector->set_name("33y");

  product_kernel *scale = new product_kernel(xvector,yvector);
  scale->add_sigma_operator( no_op );
  REQUIRE_NOTHROW( scale->last_dependency()->set_name("33dep") );

  std::shared_ptr<task> scale_task;
  std::shared_ptr<object> halo;
  double *halo_data,*ydata;

  scale->set_name("33scale");
  CHECK_NOTHROW( scale->set_localexecutefn( &vecscalebytwo ) );
  CHECK_NOTHROW( scale->analyze_dependencies() );
  CHECK_NOTHROW( scale->execute() );

  CHECK( scale->get_tasks().size()==1 );
  CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );
  CHECK_NOTHROW( halo = scale_task->get_beta_object(0) );
  CHECK( halo->volume(mycoord)==nodelocal );
  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) {
    CHECK( halo_data[i] == Approx( pointfunc33(i,my_first) ) );
  }
    
  {
    std::shared_ptr<algorithm> queue;
    REQUIRE_NOTHROW( queue = scale_task->get_node_queue() );
    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks.size()==2*omp_nprocs );
    for ( auto t : tsks ) {
      double *data; index_int thread_myfirst,thread_mylast,offset;
      if (t->has_type_origin()) { // the out object should be the mpi in object
	std::shared_ptr<object> out_object;
	REQUIRE_NOTHROW( out_object = t->get_out_object() );
	REQUIRE_NOTHROW( offset =
			 out_object->get_numa_structure()->first_index_r()[0] -
			 out_object->get_global_structure()->first_index_r()[0] );
	//REQUIRE_NOTHROW( offset = out_object->global_first_index_r() );
	for (int p=0; p<omp_nprocs; p++) {
	  processor_coordinate pcoord;
	  decomposition *ompdecomp;
	  REQUIRE_NOTHROW( ompdecomp = decomp->get_embedded_decomposition() );
	  REQUIRE_NOTHROW( pcoord = ompdecomp->coordinate_from_linear(p) );
	  INFO( "thread " << p );
	  REQUIRE_NOTHROW( data = out_object->get_data(pcoord) );
	  REQUIRE_NOTHROW( thread_myfirst = out_object->first_index_r(pcoord).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = out_object->last_index_r(pcoord).coord(0) );
	  INFO( "displaying " << thread_myfirst << "-" << thread_mylast );
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    CHECK( data[i-offset]==Approx( pointfunc33(i,0/*my_first*/) ) );
	  }
	}
      } else { // compute task
	std::shared_ptr<object> in_object,out_object;
	REQUIRE_NOTHROW( in_object = t->get_in_object(0) );
	//REQUIRE_NOTHROW( offset = in_object->global_first_index_r() );
	REQUIRE_NOTHROW( offset =
			 in_object->get_numa_structure()->first_index_r()[0] -
			 in_object->get_global_structure()->first_index_r()[0] );
	INFO( "in object: " << in_object->get_name() );
	for (int p=0; p<omp_nprocs; p++) {
	  auto pcoord = decomp->get_embedded_decomposition()->coordinate_from_linear(p);
	  INFO( "thread " << p );
	  REQUIRE_NOTHROW( data = in_object->get_data(pcoord) );
	  REQUIRE_NOTHROW( thread_myfirst = in_object->first_index_r(pcoord).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = in_object->last_index_r(pcoord).coord(0) );
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    CHECK( data[i-offset]==Approx( pointfunc33(i,0) ) );
	  }
	}
	// same with out
	REQUIRE_NOTHROW( out_object = t->get_out_object() );
	//REQUIRE_NOTHROW( offset = out_object->global_first_index_r() );
	REQUIRE_NOTHROW( offset =
			 out_object->get_numa_structure()->first_index_r()[0] -
			 out_object->get_global_structure()->first_index_r()[0] );
	INFO( "out object: " << out_object->get_name() );
	for (int p=0; p<omp_nprocs; p++) {
	  auto pcoord = decomp->get_embedded_decomposition()->coordinate_from_linear(p);
	  INFO( "thread " << p );
	  REQUIRE_NOTHROW( data = out_object->get_data(pcoord) );
	  REQUIRE_NOTHROW( thread_myfirst = out_object->first_index_r(pcoord).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = out_object->last_index_r(pcoord).coord(0) );
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    INFO( "index: g=" << i << ", relative to thread first: " << i-thread_myfirst );
	    CHECK( data[i-offset]==Approx( 2*pointfunc33(i,0) ) );
	  }
	}
      }
    }
  }

  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) {
    CHECK( ydata[i] == Approx( 2*pointfunc33(i,my_first)) );
  }
}

TEST_CASE( "Shift kernel modulo","[task][kernel][halo][modulo][execute][34]" ) {

  INFO( "mytid=" << mytid );

  index_int nlocal = 5,nodelocal = nlocal*omp_nprocs, gsize = nodelocal*mpi_nprocs;

  product_distribution *block = new product_block_distribution(decomp,nodelocal*mpi_nprocs);
  auto 
    xvector = std::shared_ptr<object>( new product_object(block) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  CHECK( block->volume(mycoord)==nodelocal );
  {
    double *data;
    REQUIRE_NOTHROW( data = xvector->get_data(mycoord) );
    for (int i=0; i<nodelocal; i++)
      data[i] = pointfunc33(i,my_first);
    REQUIRE_NOTHROW( data = yvector->get_data(mycoord) );
    for (int i=0; i<nodelocal; i++)
      data[i] = -37; // put a recognizable value
  }
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("34shift");
  shift->add_sigma_operator(  ioperator("none")  );
  shift->add_sigma_operator(  ioperator(">>1")  );
  shift->set_localexecutefn( &vecshiftleftmodulo );
  
  std::vector<std::shared_ptr<task>> tsks;
  std::shared_ptr<task> shift_task;
  const char *path;

  SECTION( "by task" ) {
    path = "by task";
    REQUIRE_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
    CHECK_NOTHROW( shift->split_to_tasks() );
    REQUIRE_NOTHROW( tsks = shift->get_tasks() );
    CHECK_NOTHROW( shift_task = tsks.at(0) );


    CHECK_NOTHROW( shift_task->analyze_dependencies() );

    // see what the message structure looks like
    std::vector<message*> msgs; int mcount;
    CHECK_NOTHROW( msgs = shift_task->get_receive_messages() );
    mcount = 0;
    for ( auto m : msgs ) {
      mcount += m->get_sender().coord(0);
    }
    CHECK( msgs.size()==2 );
    CHECK( mcount==( mytid+MOD(mytid+1,mpi_nprocs) ) );
    
    CHECK_NOTHROW( msgs = shift_task->get_send_messages() );
    mcount = 0;
    for ( auto m : msgs ) {
      mcount += m->get_receiver().coord(0);
    }
    CHECK( msgs.size()==2 );
    CHECK( mcount==( mytid+MOD(mytid-1,mpi_nprocs) ) );

    CHECK_NOTHROW( shift_task->execute() );
  }

  SECTION( "by kernel" ) {
    path = "by kernel";
    CHECK_NOTHROW( shift->analyze_dependencies() );
    CHECK_NOTHROW( shift->execute() );
    REQUIRE_NOTHROW( tsks = shift->get_tasks() );
    CHECK_NOTHROW( shift_task = tsks.at(0) );
  }
  INFO( "path: " << path );

  { // check the structure of the MPI halo
    std::shared_ptr<object> halo;
    CHECK_NOTHROW( halo = shift_task->get_beta_object(0) );
    CHECK( halo->volume(mycoord)==nodelocal+1 );
    CHECK( halo->first_index_r(mycoord).coord(0)==my_first );
    double *halo_data;
    CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
    for (int i=0; i<nodelocal+1; i++) {
      INFO( "p=" << mytid << ", i=" << i << "; halo_data[i]=" << halo_data[i] );
      int iglobal = MOD(my_first+i,gsize);
      CHECK( halo_data[i] == Approx(pointfunc33(0,iglobal)) );
    }
  }

  {
    std::shared_ptr<algorithm> queue;
    REQUIRE_NOTHROW( queue = shift_task->get_node_queue() );
    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks.size()==2*omp_nprocs );

    printf("need better OMP load distribution!\n");
    // analyze omp origin tasks
    for ( auto t : tsks ) {
      std::shared_ptr<object> in_object,out_object;
      double *data; index_int thread_myfirst,thread_mylast, offset;
      if (t->has_type_origin()) { // the out object should be the mpi in object
	REQUIRE_NOTHROW( out_object = t->get_out_object() );
	CHECK( out_object->global_volume()==nodelocal+1 );
	REQUIRE_NOTHROW( offset = out_object->global_first_index().coord(0) );
	for (int p=0; p<omp_nprocs; p++) {
	  INFO( "thread " << p );
	  processor_coordinate pcoord;
	  decomposition *ompdecomp;
	  REQUIRE_NOTHROW( ompdecomp = decomp->get_embedded_decomposition() );
	  REQUIRE_NOTHROW( pcoord = ompdecomp->coordinate_from_linear(p) );
	  if (p==0)
	    CHECK( out_object->volume(pcoord)==nlocal+1 );
	  else
	    CHECK( out_object->volume(pcoord)==nlocal );
	  REQUIRE_NOTHROW( data = out_object->get_data(pcoord) );
	  REQUIRE_NOTHROW( thread_myfirst = out_object->first_index_r(pcoord).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = out_object->last_index_r(pcoord).coord(0) );
	  INFO( "displaying " << thread_myfirst << "-" << thread_mylast );
	  // check content
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    int iglobal = MOD(i,gsize);
	    CHECK( data[i-offset]==Approx( pointfunc33(iglobal,0) ) );
	  }
	}
      }
    }

    // analyze omp compute tasks
    for ( auto t : tsks ) {
      std::shared_ptr<object> in_object,out_object;
      double *data;
      index_int thread_myfirst,thread_mylast, offset;
      if (!t->has_type_origin()) { // compute kernels go from OMP halo to real output
	REQUIRE_NOTHROW( in_object = t->get_in_object(0) );
	CHECK( in_object->global_volume()==nodelocal+1 );
	REQUIRE_NOTHROW( data = in_object->get_data(mycoord) );
	REQUIRE_NOTHROW( offset = in_object->global_first_index().coord(0) );
	int m=0; if (mytid==mpi_nprocs-1) m=1;
	for (index_int i=0; i<in_object->global_volume()-m; i++)
	  CHECK( data[i]==pointfunc33(offset,i) );
	if (mytid==mpi_nprocs-1) // wrap around connection from zero
	  CHECK( data[in_object->global_volume()-1]==pointfunc33(0,0) );
	auto p = t->get_domain();

	// analyze in object
	{
	  INFO( "thread " << p.as_string() );
	  if (p==0) // halos abut?
	    CHECK( in_object->volume(p)==nlocal+1 );
	  else
	    CHECK( in_object->volume(p)==nlocal );
	  REQUIRE_NOTHROW( data = in_object->get_data_p(p) );
	  REQUIRE_NOTHROW( thread_myfirst = in_object->first_index_r(p).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = in_object->last_index_r(p).coord(0) );
	  INFO( "range [" << thread_myfirst << "," << thread_mylast << "]" );
	  CHECK( thread_mylast==thread_myfirst+in_object->volume(p)-1 );
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    int iglobal = MOD(i,gsize);
	    INFO( "i=" << i << " (global=" << iglobal << ") halo_data[i]=" << data[i-offset] );
	    CHECK( data[i-offset]==Approx( pointfunc33(iglobal,0) ) );
	  }
	}
	// same with out
	{
	  REQUIRE_NOTHROW( out_object = t->get_out_object() );
	  CHECK( out_object->global_volume()==nodelocal );
	  REQUIRE_NOTHROW( offset = out_object->global_first_index().coord(0) );
	  CHECK( out_object->volume(p)==nlocal );
	  REQUIRE_NOTHROW( data = out_object->get_data_p(p) );
	  REQUIRE_NOTHROW( thread_myfirst = out_object->first_index_r(p).coord(0) );
	  REQUIRE_NOTHROW( thread_mylast = out_object->last_index_r(p).coord(0) );
	  if (p==0) // when we fix omp load distribution this will change
	    CHECK( thread_myfirst==in_object->first_index_r(p).coord(0) );
	  else
	    CHECK( thread_myfirst==in_object->first_index_r(p).coord(0)-1 ); // halo has surplus elt
	  CHECK( thread_mylast==in_object->last_index_r(p).coord(0)-1 );
	  INFO( "range [" << thread_myfirst << "," << thread_mylast << "]" );
	  for (int i=thread_myfirst; i<=thread_mylast; i++) {
	    int iglobal = MOD(i+1,gsize);
	    INFO( "i=" << i << " (global=" << iglobal << ") out data[i]=" << data[i-offset] );
	    CHECK( data[i-offset]==Approx( pointfunc33(iglobal,0) ) );
	  }
	}
      }
    }
  }

  double *ydata;
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  CHECK( yvector->volume(mycoord)==nodelocal );
  {
    if (mytid==mpi_nprocs-1) {
      for (int i=0; i<nodelocal-1; i++) {
  	INFO( "p=" << mytid << ", i=" << i << " yi=" << ydata[i] );
  	CHECK( ydata[i] == Approx(pointfunc33(i+1,my_first)) );
      }
      int i = nodelocal-1; // globally last element is wrapped modulo
      INFO( "i=" << i << " yi=" << ydata[i] );
      CHECK( ydata[i] == Approx(pointfunc33(0,0)) );
    } else {
      for (int i=0; i<nodelocal; i++) {
  	INFO( "p=" << mytid << ", i=" << i << " yi=" << ydata[i] );
  	CHECK( ydata[i] == Approx(pointfunc33(i+1,my_first)) );
      }
    }
  }
}

TEST_CASE( "Shift from left kernel, message structure","[task][kernel][halo][35]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10,nodelocal=nlocal*omp_nprocs,gsize=nodelocal*mpi_nprocs;
  product_distribution *block = new product_block_distribution(decomp,nodelocal,-1);
  double *xdata = new double[nodelocal];
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  for (int i=0; i<xvector->volume(mycoord); i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("35shift");
  shift->add_sigma_operator(  ioperator("none")  );
  shift->add_sigma_operator(  ioperator("<=1")  );
  shift->set_localexecutefn( &vecshiftrightbump );
  
  std::shared_ptr<task> shift_task;

  REQUIRE_NOTHROW( shift->analyze_dependencies() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );

  // see if the halo is properly transmitted
  auto rmsgs = shift_task->get_receive_messages();
  if (mytid>0) { // everyone but the first receives from the left
    CHECK( rmsgs.size()==2 );
    for (int i=0; i<2; i++) {
      auto msg = rmsgs.at(i);
      auto rstruct = msg->get_local_struct(); // local wrt the halo
      if (msg->get_sender().coord(0)==mytid-1) { // msg to the right
	CHECK( msg->get_local_struct()->first_index_r()==domain_coordinate_zero(1) );
	CHECK( msg->get_local_struct()->last_index_r()==domain_coordinate_zero(1) );
      } else {
	// CHECK( msg->get_sender().coord(0)==mytid);
	// CHECK( rstruct->first_index_r()==1 );
	// CHECK( rstruct->last_index_r()==nlocal+1 );
      }
    }
  } else {
    CHECK( rmsgs.size()==1 );
  }
  auto smsgs = shift_task->get_send_messages();
  if (mytid<mpi_nprocs-1) {
    int i;
    CHECK( smsgs.size()==2 ); // everyone but the last sends to the right
    for ( auto msg : smsgs ) {
      INFO( "message: " << msg->as_string() );
      auto sstruct = msg->get_global_struct();
      if (msg->get_receiver().coord(0)==mytid+1) { // msg to the right
	CHECK( sstruct->first_index_r()==my_last );
	CHECK( sstruct->last_index_r()==my_last );
      } else {
	CHECK( sstruct->first_index_r()==my_first );
	CHECK( sstruct->last_index_r()==my_last );
      }
    }
  } else {
    CHECK( smsgs.size()==1 );
  }

}

TEST_CASE( "Shift right kernel, execute","[task][kernel][halo][execute][36]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  double *xdata = new double[nlocal];
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  xvector->set_name("36x"); yvector->set_name("36y");
  index_int my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("36shift");
  shift->add_sigma_operator(  ioperator("none")  );
  shift->add_sigma_operator(  ioperator("<=1")  );
  shift->set_localexecutefn( &vecshiftrightbump );
  
  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  REQUIRE_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks().at(0) );
  CHECK_NOTHROW( shift->analyze_dependencies() );
  CHECK_NOTHROW( shift->last_dependency()->set_name("36mpi-dependency") );

  std::shared_ptr<object> halo;
  CHECK_NOTHROW( halo = shift->get_beta_object(0) );
  {
    INFO( "investigating halo <<" << halo->get_name() << ">>" );
    // " of type " << halo->architecture::as_string()
    INFO( "halo distribution: " << halo->as_string() );
    INFO( "vector distribution: " << xvector->as_string() );
    index_int halo_first;
    REQUIRE_NOTHROW( halo_first = halo->first_index_r(mycoord).coord(0) );
    if (mytid==0)
      CHECK( halo_first==0 );
    else
      CHECK( halo_first==yvector->first_index_r(mycoord).coord(0)-1);
  }

  CHECK_NOTHROW( shift->execute() );

  CHECK_NOTHROW( halo_data = halo->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // there is no left halo, so hi==fi
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i,my_first)) );
      }
    } else { // there is a left halo, so hi==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }

  /*
   * Inspect the OMP queue in the local MPI task
   */
  std::shared_ptr<algorithm> queue;
  REQUIRE_NOTHROW( queue = shift_task->get_node_queue() );
  //REQUIRE_NOTHROW( queue = dynamic_cast<product_task*>(shift_task.get())->get_node_queue() );
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    auto thr = t->get_domain();
    INFO( "process " << mytid << ", thread " << thr.as_string() );	  
    std::shared_ptr<object> omp_in,omp_out; double *data; distribution *halodist;
    index_int thr_first, thr_last, offset; double *indata,*outdata;
    REQUIRE_NOTHROW( omp_out = t->get_out_object() );
    INFO( "output object: " << omp_out->get_name() );
    REQUIRE_NOTHROW( outdata = omp_out->get_data(mycoord) );
    thr_first = omp_out->first_index_r(thr).coord(0);
    thr_last = omp_out->last_index_r(thr).coord(0);
    if (t->has_type_origin()) { // MPI halo
      if (thr.is_zero()) // special cases: all other thread endpoints are somewhere in between
	if (mytid==0)
	  CHECK( thr_first==my_first );
	else
	  CHECK( thr_first==my_first-1 );
      else if (thr.coord(0)==omp_nprocs-1)
	CHECK( thr_last==my_last );
      offset = omp_out->get_numa_structure()->first_index_r()[0] - omp_out->get_global_structure()->first_index_r()[0];
      for (index_int i=thr_first; i<=thr_last; i++)
	if (!(mytid==0&&thr_first==0)) // undefined case
	  CHECK( outdata[i-offset]==Approx( pointfunc33(i,0) ) );
    } else { // type compute
      // the OMP halo looks just like the MPI one
      REQUIRE_NOTHROW( omp_in = t->get_in_object(0) );
      thr_first = omp_in->first_index_r(thr).coord(0);
      thr_last = omp_in->last_index_r(thr).coord(0);
      if (thr.is_zero()) // first thread: aligned on mpiproc 0, otherwise stick out to the left
	if (mytid==0)
	  CHECK( thr_first==my_first );
	else
	  CHECK( thr_first==my_first-1 );
      if (thr.coord(0)==omp_nprocs-1) // last thread: right aligned
	CHECK( thr_last==my_last );
      if (mytid==0) // omp global size equal mpi local on mpi proc 0, otherwise one more
	CHECK( omp_in->global_volume()==nlocal );
      else
	CHECK( omp_in->global_volume()==nlocal+1 );
      REQUIRE_NOTHROW( indata = omp_in->get_data(mycoord) );
      offset = omp_in->get_numa_structure()->first_index_r()[0] - omp_in->get_global_structure()->first_index_r()[0];
      for (index_int i=thr_first; i<=thr_last; i++)
	if (!(mytid==0&&thr_first==0)) // undefined case
	  CHECK( indata[i-offset]==Approx( pointfunc33(i,0) ) );
      if (1) { // OMP output looks like global output
	thr_first = omp_out->first_index_r(thr).coord(0);
	thr_last = omp_out->last_index_r(thr).coord(0);
	if (thr.is_zero()) // first thread left aligned with mpi
	  CHECK( thr_first==my_first );
	if (thr.coord(0)==omp_nprocs-1) // last thread right aligned with mpi
	  CHECK( thr_last==my_last );
	INFO( "computing output offset as numa=" << omp_out->get_numa_structure()->as_string() <<
	      " in global=" << omp_out->get_global_structure()->as_string() );
	offset = omp_out->get_numa_structure()->first_index_r()[0] - omp_out->get_global_structure()->first_index_r()[0];
	for (index_int i=thr_first; i<=thr_last; i++) {
	  INFO( "i=" << i << " with offset " << offset );
	  if (!(mytid==0&&thr_first==0)) // undefined case
	    CHECK( outdata[i-offset]==Approx( pointfunc33(i-1,0) ) );
	}
      }
    }
  }

  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
  	INFO( "i=" << i << " yi=" << ydata[i] );
  	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    } else {
      for (i=0; i<nlocal; i++) {
  	INFO( "i=" << i << " yi=" << ydata[i] );
  	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
}

TEST_CASE( "Shift from left kernel modulo, execute","[task][kernel][halo][execute][modulo][37]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  int nglobal = block->global_volume();
  double *xdata = new double[nlocal];
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  index_int my_first = block->first_index_r(mycoord).coord(0);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("37shift");
  shift->add_sigma_operator(  ioperator("none")  );
  shift->add_sigma_operator(  ioperator("<<1")  );
  shift->set_localexecutefn( &vecshiftrightmodulo );
  
  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  REQUIRE_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks()[0] );
  CHECK_NOTHROW( shift_task->analyze_dependencies() );

  CHECK_NOTHROW( shift->execute() );
  CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // hi=f(i-1) except for i=0: hi=f(n-1)
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
      i = 0;
      CHECK( halo_data[i] == Approx(pointfunc33(nglobal-1,my_first)) );
    } else { // there is a left halo, so hi==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
  {
    int i;
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata[i] );
	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
      i = 0;
      CHECK( ydata[i] == Approx(pointfunc33(nglobal-1,my_first)) );
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata[i] );
	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
}

TEST_CASE( "Scale queue","[queue][execute][100]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=3;
  auto no_op =  ioperator("none") ;
  product_distribution *block = 
    new product_block_distribution(decomp,nlocal*mpi_nprocs);
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  double *xdata = new double[nlocal],*ydata;
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = std::shared_ptr<object>( new product_object(block) );
  }

  std::shared_ptr<algorithm> queue;
  CHECK_NOTHROW( queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) ) );
  queue->set_name("scale queue");
  {
    product_kernel *k = new product_kernel(xvector);
    k->set_name( "generate" );
    k->set_localexecutefn( &vecnoset );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    product_kernel *k; char name[20];
    if (iv==0) {
      k = new product_kernel(xvector,yvector[0]);
    } else {
      k = new product_kernel(yvector[iv-1],yvector[iv]);
    }
    sprintf(name,"update-%d",iv);
    k->set_name( name );
    k->set_localexecutefn( &vecscalebytwo );
    k->add_sigma_operator( no_op );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  REQUIRE_NOTHROW( queue->analyze_dependencies() ); // this creates node queues
  auto tsks = queue->get_tasks();
  CHECK( tsks.size()==(nsteps+1) );

  { // analyze the embedded queue
    for ( auto mpitsk : tsks ) {
      INFO( "mpi task " << mpitsk->as_string() );
      std::shared_ptr<algorithm> node_queue;
      REQUIRE_NOTHROW( node_queue = mpitsk->get_node_queue() );
      REQUIRE_NOTHROW( node_queue->split_to_tasks() );
      REQUIRE_NOTHROW( node_queue->set_outer_as_synchronization_points() );
      std::vector<std::shared_ptr<task>> tsks;
      REQUIRE_NOTHROW( tsks = node_queue->get_tasks() );
      if (mpitsk->has_type_origin()) 
	CHECK( tsks.size()==omp_nprocs );
      else
	CHECK( tsks.size()==2*omp_nprocs );
      int s;
      REQUIRE_NOTHROW( node_queue->determine_locally_executable_tasks() );
      REQUIRE_NOTHROW( s = node_queue->get_has_synchronization_tasks() );
      CHECK( s==2 );
      int count=0;
      for ( auto t : tsks) {
	INFO( "omp task " << t->as_string() );
	if (t->has_type_origin()) { count++;
	  auto d = t->get_domain();
	  if (d.is_on_face(t->get_out_object())) {
	    CHECK( t->get_is_synchronization_point() );
	  } else {
	    CHECK( !t->get_is_synchronization_point() );
	  }
	} else {
	  CHECK( !t->get_is_synchronization_point() );
	}
      }
      CHECK( count==omp_nprocs );
    }
  }

  fmt::print("{}\n",queue->header_as_string());
  CHECK_NOTHROW( queue->execute() );

  {
    int i;
    CHECK_NOTHROW( ydata = yvector[nsteps-1]->get_data(mycoord) );
    CHECK( ydata!=nullptr );
    for (i=0; i<nlocal; i++) {
      INFO( "yvalue:" << i << ":" << ydata[i] );
      CHECK( ydata[i] == Approx( pow(2,nsteps)*pointfunc33(i,my_first) ) );
    }
  }
}

TEST_CASE( "Threepoint queue mod","[queue][execute][modulo][halo][101]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=4;
  product_distribution *block;
  REQUIRE_NOTHROW( block = new product_block_distribution(decomp,nlocal*mpi_nprocs) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  auto no_op =  ioperator("none") ;
  auto right_shift_mod =  ioperator(">>1") ;
  auto left_shift_mod  =  ioperator("<<1") ;

  double *xdata = new double[nlocal],*ydata;
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  for (int i=0; i<nlocal; i++)
    xdata[i] = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    REQUIRE_NOTHROW( yvector[iv] = std::shared_ptr<object>( new product_object(block) ) );
  }

  std::shared_ptr<algorithm> queue;
  CHECK_NOTHROW( queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) ) );
  {
    product_kernel *k;
    REQUIRE_NOTHROW( k = new product_kernel(xvector) );
    k->set_localexecutefn( &vecnoset );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    product_kernel *k;
    if (iv==0) {
      REQUIRE_NOTHROW( k = new product_kernel(xvector,yvector[0]) );
    } else {
      REQUIRE_NOTHROW( k = new product_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn( &threepointsummod );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK( queue->get_tasks().size()==(nsteps+1) );
  std::vector<std::shared_ptr<task>> predecessors,tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    CHECK_NOTHROW( predecessors = t->get_predecessors() );
    INFO( "step=" << t->get_step() );
    if (t->has_type_origin()) {
      CHECK( t->get_receive_messages().size()==0 );
      CHECK( predecessors.size()==0 );
    } else {
      CHECK( t->get_receive_messages().size()==3 );
      CHECK( predecessors.size()==3 );
    }
  }

  CHECK_NOTHROW( queue->execute() );
  return;
  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata[i] );
	CHECK( ydata[i] == Approx( pow(3,s+1) ) );
      }
    }
  }
}

TEST_CASE( "Threepoint queue","[queue][execute][halo][102]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=4;
  product_distribution *block;
  REQUIRE_NOTHROW( block = new product_block_distribution(decomp,nlocal*mpi_nprocs) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  auto no_op =  ioperator("none") ;
  auto right_shift_mod =  ioperator(">=1") ;
  auto left_shift_mod  =  ioperator("<=1") ;

  double *xdata = new double[nlocal],*ydata;
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  for (int i=0; i<nlocal; i++)
    xdata[i] = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    REQUIRE_NOTHROW( yvector[iv] = std::shared_ptr<object>( new product_object(block) ) );
  }

  std::shared_ptr<algorithm> queue;
  CHECK_NOTHROW( queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) ) );
  {
    product_kernel *k;
    REQUIRE_NOTHROW( k = new product_kernel(xvector) );
    k->set_localexecutefn( &vecnoset );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    product_kernel *k;
    if (iv==0) {
      REQUIRE_NOTHROW( k = new product_kernel(xvector,yvector[0]) );
    } else {
      REQUIRE_NOTHROW( k = new product_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn( &threepointsumbump );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK( queue->get_tasks().size()==(nsteps+1) );
  std::vector<std::shared_ptr<task>> predecessors,tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    CHECK_NOTHROW( predecessors = t->get_predecessors() );
    INFO( "step=" << t->get_step() );
    if (t->has_type_origin()) {
      CHECK( t->get_receive_messages().size()==0 );
      CHECK( predecessors.size()==0 );
    } else {
      auto d = t->get_domain();
      if (d.is_on_face(t->get_out_object())) { // (d==0 || d==mpi_nprocs-1) {
	CHECK( t->get_receive_messages().size()==2 );
	CHECK( predecessors.size()==2 );
      } else {
	CHECK( t->get_receive_messages().size()==3 );
	CHECK( predecessors.size()==3 );
      }
    }
  }

  CHECK_NOTHROW( queue->execute() );
  return;
  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata[i] );
	CHECK( ydata[i] == Approx( pow(3,s+1) ) );
      }
    }
  }
}

#if 0

TEST_CASE( "Shift from left kernel bump, execute","[task][kernel][halo][execute][modulo][38]" ) {

  INFO( "mytid=" << mytid << " out of " << mpi_nprocs );

  int nlocal=10;
  product_distribution *block = 
    new product_block_distribution(decomp,-1,nlocal*mpi_nprocs);
  int nglobal = block->global_volume();
  double *xdata = new double[nlocal];
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  index_int my_first = block->first_index_r(mycoord).coord(0);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *shift = new product_kernel(xvector,yvector);
  shift->set_name("38shift");
  shift->add_sigma_operator(  ioperator("none")  );
  shift->add_sigma_operator(  ioperator("<=1")  );
  shift->set_localexecutefn( &vecshiftrightbump );

  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  REQUIRE_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( shift_task = shift->get_tasks()[0] );
  CHECK_NOTHROW( shift_task->analyze_dependencies() );

  REQUIRE_NOTHROW( shift->execute() );
  CHECK_NOTHROW( halo_data = shift_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i;
    if (mytid==0) { // there is no halo to the left
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i,my_first)) );
      }
    } else { // there is a left halo, so hi==f(i-1)
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " hi=" << halo_data[i] );
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
  {
    int i;
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata[i] );
	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
      i = 0;
      CHECK_NOTHROW( ydata[0]==1 );
      //      CHECK( ydata[i] == Approx(pointfunc33(i,my_first)) );
    } else {
      for (i=0; i<nlocal; i++) {
	INFO( "i=" << i << " yi=" << ydata[i] );
	//	CHECK( ydata[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
}

TEST_CASE( "Add with left right modulo, execute","[task][kernel][halo][execute][modulo][39]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=10;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  int nglobal = block->global_volume();
  double *xdata = new double[nlocal];
  auto xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first);
  product_kernel
    *sum = new product_kernel(xvector,yvector);
  sum->set_name("39sum");
  sum->add_sigma_operator(  ioperator("none")  );
  sum->add_sigma_operator(  ioperator("<<1")  );
  sum->add_sigma_operator(  ioperator(">>1")  );
  sum->set_localexecutefn( &threepointsum );

  std::shared_ptr<task> sum_task;
  double *halo_data,*ydata;

  REQUIRE_NOTHROW( sum->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( sum->split_to_tasks() );
  CHECK_NOTHROW( sum_task = sum->get_tasks()[0] );
  CHECK_NOTHROW( sum_task->analyze_dependencies() );

  CHECK_NOTHROW( sum->execute() );
  CHECK_NOTHROW( halo_data = sum_task->get_beta_object(0)->get_data(mycoord) );
  CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
  {
    int i=0, lastv = pointfunc33(nlocal-1,block->first_index_r(mpi_nprocs-1));
    INFO( "i=" << i << " hi=" << halo_data[i] );
    if (mytid==0) {
      for (i=1; i<nlocal+2; i++) {
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
      i = 0;
      CHECK( halo_data[i] == Approx(lastv) );
    } else if (mytid==mpi_nprocs-1) {
      for (i=0; i<nlocal+1; i++) {
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
      i = nlocal+1;
      CHECK( halo_data[i] == Approx(pointfunc33(0,0)) );
    } else {
      for (i=0; i<nlocal+2; i++) {
	CHECK( halo_data[i] == Approx(pointfunc33(i-1,my_first)) );
      }
    }
  }
  {
    int i=0, lastv = pointfunc33(nlocal-1,block->first_index_r(mpi_nprocs-1));
    INFO( "i=" << i << " yi=" << ydata[i] );
    if (mytid==0) {
      for (i=1; i<nlocal; i++) {
	CHECK( ydata[i] == Approx(3*pointfunc33(i,my_first)) );
      }
      i = 0; // last+0+1
      CHECK( ydata[i] == Approx(pointfunc33(i,my_first)+pointfunc33(i+1,my_first)+lastv) );
    } else if (mytid==mpi_nprocs-1) {
      for (i=0; i<nlocal-1; i++) {
	CHECK( ydata[i] == Approx(3*pointfunc33(i,my_first)) );
      }
      i = nlocal-1; // last-1+last+0
      CHECK( ydata[i] == Approx(pointfunc33(i,my_first)+pointfunc33(i-1,my_first)+pointfunc33(0,0)) );
    } else {
      for (i=0; i<nlocal-1; i++) {
	CHECK( ydata[i] == Approx(3*pointfunc33(i,my_first)) );
      }
    }
  }
}

TEST_CASE( "Test explicit beta","[beta][distribution][40]" ) {

  INFO( "mytid=" << mytid );

  int localsize=15,gsize = localsize*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *block
    = new product_block_distribution(decomp,localsize,-1);
  index_int 
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  product_distribution
    *left,*right,*wide;
  left = (product_distribution*)block->operate
    (  ioperator("<<1")  );
  right = (product_distribution*)block->operate
    (  ioperator(">>1")  );
  REQUIRE_NOTHROW( wide = (product_distribution*)left->distr_union(right) );
  CHECK( wide->first_index_r(mycoord).coord(0)==my_first-1 ); // distributions can stick out
  CHECK( wide->last_index_r(mycoord).coord(0)==my_last+1 );

  auto in = std::shared_ptr<object>( new product_object(block) ),
    out = std::shared_ptr<object>( new product_object(block) );
  product_kernel *threepoint;
  REQUIRE_NOTHROW( threepoint = new product_kernel(in,out) );
  threepoint->set_localexecutefn( &threepointsum );
  threepoint->set_name("40threepoint");
  REQUIRE_NOTHROW( threepoint->set_explicit_beta_distribution
		   (wide->get_environment(),wide->get_processor_structure()) );
  REQUIRE_NOTHROW( threepoint->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = threepoint->get_tasks() );
  CHECK( tsks.size()==1 );
  std::shared_ptr<task> threetask;
  REQUIRE_NOTHROW( threetask = tsks[0] );
  std::vector<message*> rmsgs;
  REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
  CHECK( rmsgs.size()==3 );

  double *indata = in->get_data(mycoord);
  CHECK( in->volume(mycoord)==localsize );
  for (index_int i=0; i<localsize; i++)
    indata[i] = 2.;
  REQUIRE_NOTHROW( threepoint->execute() );
  double *outdata = out->get_data(mycoord);
  CHECK( out->volume(mycoord)==localsize );
  for (index_int i=0; i<localsize; i++) {
    INFO( "i=" << i << " data[i]=" << outdata[i] );
    CHECK( outdata[i] == Approx(6.) );
  }
}

TEST_CASE( "Beta from sparse matrix","[beta][sparse][41]" ) {

  int localsize=20,gsize=localsize*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = .5;
  }
  product_kernel *kern = new product_kernel(in_obj,out_obj);
  kern->set_name("sparse-stuff");
  kern->set_localexecutefn( &local_sparse_matrix_vector_multiply );
  product_sparse_matrix *pattern;

  SECTION( "connect right" ) {
    REQUIRE_NOTHROW( pattern = new product_sparse_matrix(block) );
    for (index_int i=block->first_index_r(mycoord).coord(0); i<=block->last_index_r(mycoord).coord(0); i++) {
      pattern->add_element(i,i);
      if (i+1<gsize) {
	CHECK_NOTHROW( pattern->add_element(i,i+1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i+1) );
      }
    }
    REQUIRE_NOTHROW( kern->set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = kern->get_tasks() );
    CHECK( tsks.size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tsks[0] );
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==mpi_nprocs-1) {
      CHECK( rmsgs.size()==1 );
    } else {
      CHECK( rmsgs.size()==2 );
    }
    for ( m : rmsgs ) { // (std::vector<message*>::iterator m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
      //      message *msg = (message*)(*m);
      if (msg->get_sender().equals(msg->get_receiver())) {
	CHECK( msg->size()==localsize );
      } else {
	CHECK( msg->size()==1 );
      }
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
    // double *outdata = out_obj->get_data(mycoord); index_int n = out_obj->volume(mycoord);
    // if (mytid==mpi_nprocs-1) {
    //   for (index_int i=0; i<n-1; i++)
    // 	CHECK( outdata[i] == Approx(1.) );
    //   CHECK( outdata[n-1] == Approx(.5) );
    // } else {
    //   for (index_int i=0; i<n; i++)
    // 	CHECK( outdata[i] == Approx(1.) );
    // }
  }

  SECTION( "connect left" ) {
    REQUIRE_NOTHROW( pattern = new product_sparse_matrix(block) );
    for (index_int i=block->first_index_r(mycoord).coord(0); i<=block->last_index_r(mycoord).coord(0); i++) {
      pattern->add_element(i,i);
      if (i-1>=0) {
	CHECK_NOTHROW( pattern->add_element(i,i-1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i-1) );
      }
    }
    REQUIRE_NOTHROW( kern->set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = kern->get_tasks() );
    CHECK( tsks.size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tsks[0] );
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==0) {
      CHECK( rmsgs.size()==1 );
    } else {
      CHECK( rmsgs.size()==2 );
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
  }

}

TEST_CASE( "Actual sparse matrix","[beta][sparse][42]" ) {
  REQUIRE(mpi_nprocs>1); // need at least two processors

  int localsize=20,gsize=localsize*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }
  product_kernel *spmvp = new product_kernel(in_obj,out_obj);
  spmvp->set_name("sparse-mvp");

  int ncols = 4;
  index_int mincol = block->first_index_r(mycoord).coord(0), maxcol = block->last_index_r(mycoord).coord(0);
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->global_ndomains()));

  // create a matrix with zero row sums
  product_sparse_matrix *mat = new product_sparse_matrix(block);
  index_int my_first = block->first_index_r(mycoord).coord(0),my_last = block->last_index_r(mycoord).coord(0);
  indexstruct *my_columns = new contiguous_indexstruct(my_first,my_last);
  for (index_int row=my_first; row<=my_last; row++) {
    for (index_int ic=0; ic<ncols; ic++) {
      //printf("columns so far: %s\n",my_columns->as_string());
      index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
      CHECK( xs>=0 );
      CHECK( xs<localsize );
      if (mytid<mpi_nprocs-1) col = my_last+1+xs;
      else               col = xs;
      REQUIRE( col>=0 );
      REQUIRE( col<gsize );
      if ( !((col<my_first) || ( col>my_last )) )
	printf("range error in row %d: %d [%d-%d]\n",row,col,my_first,my_last);
      REQUIRE( ((col<my_first) || ( col>my_last )) );
      if (col<mincol) mincol = col;
      if (col>maxcol) maxcol = col;
      REQUIRE_NOTHROW( mat->add_element(row,col,-1.) );
      indexstruct *col_struct = new contiguous_indexstruct(col,col);
      CHECK( col_struct->is_contiguous() );
      CHECK( col_struct->is_sorted() );
      CHECK( col_struct->volume()==1 );
      //printf(" adding %s\n",col_struct->as_string());
      REQUIRE_NOTHROW( my_columns->union_with(col_struct) );
    }
    REQUIRE_NOTHROW( mat->add_element(row,row,(double)ncols+1.5) );    
  }

  indexstruct *mstruct = mat->all_columns();
  // printf("matrix structure on %d: %s\nmy columns: %s",
  // 	 mytid,mstruct->as_string(),my_columns->as_string());
  CHECK( mstruct->equals(my_columns) );

  spmvp->set_index_pattern( mat );
  spmvp->set_localexecutefn( &local_sparse_matrix_vector_multiply );
  spmvp->set_localexecutectx( mat );
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = spmvp->get_tasks() );
  CHECK( tsks.size()==1 );
  std::shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = tsks[0] );
  std::vector<message*> rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for ( auto m : rmsgs ) {
    if (msg->get_sender()!=msg->get_receiver())
      if (mytid==mpi_nprocs-1)
	CHECK( msg->get_sender().coord(0)==0 );
      else 
	CHECK( msg->get_sender().coord(0)==mytid+1 );
  }

  {
    signature_function *beta;
    REQUIRE_NOTHROW( beta = spmvp->get_beta_definition() );
    distribution *structure;
    REQUIRE_NOTHROW( structure = beta->get_beta_distribution() );
    indexstruct *column_indices;
    REQUIRE_NOTHROW( column_indices = structure->get_processor_structure(mycoord) );
    //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
    CHECK( column_indices->is_sorted() ); 
    REQUIRE_NOTHROW( mat->remap( beta ) );
  }

  {
    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = spmvp->get_tasks() );
    CHECK( tsks.size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tsks[0] );
  }
  REQUIRE_NOTHROW( spmvp->execute() );

  {
    double *data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->volume(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data[i] == Approx(1.5) );
    }
  }
}
TEST_CASE( "Sparse matrix kernel","[beta][sparse][43]" ) {
  REQUIRE(mpi_nprocs>1); // need at least two processors

  int localsize=20,gsize=localsize*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }

  int ncols = 4;
  index_int mincol = block->first_index_r(mycoord).coord(0), maxcol = block->last_index_r(mycoord).coord(0);
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->global_ndomains()));

  // create a matrix with zero row sums
  product_sparse_matrix *mat = new product_sparse_matrix(block);
  index_int my_first = block->first_index_r(mycoord).coord(0),my_last = block->last_index_r(mycoord).coord(0);
  for (index_int row=my_first; row<=my_last; row++) {
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
      CHECK( xs>=0 );
      CHECK( xs<localsize );
      if (mytid<mpi_nprocs-1) col = my_last+1+xs;
      else               col = xs;
      if (col<mincol) mincol = col;
      if (col>maxcol) maxcol = col;
      mat->add_element(row,col,-1.); // off elt
    }
    mat->add_element(row,row,(double)ncols+1.5); // diag elt
  }

  product_kernel *spmvp = new product_spmvp_kernel(in_obj,out_obj,mat);
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = spmvp->get_tasks() );
  CHECK( tsks.size()==1 );
  std::shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = tsks[0] );
  std::vector<message*> rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for ( auto m : rmsgs ) {
    if (msg->get_sender()!=msg->get_receiver())
      if (mytid==mpi_nprocs-1)
	CHECK( msg->get_sender().coord(0)==0 );
      else 
	CHECK( msg->get_sender().coord(0)==mytid+1 );
  }

  {
    signature_function *beta;
    REQUIRE_NOTHROW( beta = spmvp->get_beta_definition() );
    distribution *structure;
    REQUIRE_NOTHROW( structure = beta->get_beta_distribution() );
    indexstruct *column_indices;
    REQUIRE_NOTHROW( column_indices = structure->get_processor_structure(mycoord) );
    //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
    CHECK( column_indices->is_sorted() ); 
    REQUIRE_NOTHROW( mat->remap( beta ) );
  }

  {
    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = spmvp->get_tasks() );
    CHECK( tsks.size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tsks[0] );
  }
  REQUIRE_NOTHROW( spmvp->execute() );

  {
    double *data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->volume(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data[i] == Approx(1.5) );
    }
  }
}

TEST_CASE( "Kernel data graph: test acyclicity","[kernel][50]" ) {

  int localsize = 20,gsize = localsize*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  std::shared_ptr<object> object1,object2,object3,object4;
  product_kernel *kernel1,*kernel2,*kernel3;
  std::shared_ptr<algorithm> queue;
  char *object_name,*test_name;

  SECTION( "unnamed objects" ) {
    object1 = std::shared_ptr<object>( new product_object(block) );
    object2 = std::shared_ptr<object>( new product_object(block) );
    object3 = std::shared_ptr<object>( new product_object(block) );
    object4 = std::shared_ptr<object>( new product_object(block) );

    kernel1 = new product_kernel(object1,object2);
    kernel1->set_name("50kernel1");
    kernel2 = new product_kernel(object1,object3);
    kernel2->set_name("50kernel2");
    kernel3 = new product_kernel(object2,object3);
    kernel3->set_name("50kernel3");

    queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel3) );

    const char *name;
    // kernel *def;
    // std::vector<kernel*> *use;

    // there are two kernels that use object1, none that generate it
    name = object1->get_name();
    // REQUIRE_NOTHROW( queue->get_data_relations(name,&def,&use) ); // VLE need unique names
    // CHECK( def==NULL );
    // CHECK( use->size()==2 );
    // // there are two kernels that generate object3; we do not allow that.
    // name = object3->get_name();
    // REQUIRE_THROWS( queue->get_data_relations(name,&def,&use) );
  }

  SECTION( "named objects" ) {
    object1 = std::shared_ptr<object>( new product_object(block) );
    object1->set_name("object1");
    object2 = std::shared_ptr<object>( new product_object(block) );
    object2->set_name("object2");
    object3 = std::shared_ptr<object>( new product_object(block) );
    object3->set_name("object3");
    object4 = std::shared_ptr<object>( new product_object(block) );
    object4->set_name("object4");

    kernel1 = new product_kernel(object1,object2);
    kernel1->set_name("make2");
    kernel2 = new product_kernel(object1,object3);
    kernel2->set_name("make3");
    kernel3 = new product_kernel(object2,object3);
    kernel3->set_name("make4");

    queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel3) );

    kernel<product_message> *def;
    std::vector<kernel<product_message>*> *use;
    // there are two kernels that use object1, none that generate it
    REQUIRE_NOTHROW( queue->get_data_relations("object1",&def,&use) );
    CHECK( def==NULL );
    CHECK( use->size()==2 );
    // there are two kernels that generate object3; we do not allow that.
    REQUIRE_THROWS( queue->get_data_relations("object3",&def,&use) );
  }

  SECTION( "kernel predecessors" ) {
    object1 = std::shared_ptr<object>( new product_object(block) );
    object1->set_name("object1");
    object2 = std::shared_ptr<object>( new product_object(block) );
    object2->set_name("object2");
    object3 = std::shared_ptr<object>( new product_object(block) );
    object3->set_name("object3");
    object4 = std::shared_ptr<object>( new product_object(block) );
    object4->set_name("object4");

    kernel1 = new product_kernel(object1,object2);
    kernel1->set_name("make2");
    kernel2 = new product_kernel(object2,object3);
    kernel2->set_name("make3");

    queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );

    kernel<product_message> *def;
    std::vector<kernel<product_message>*> *use;
    // there are two kernels that use object1, none that generate it
    REQUIRE_NOTHROW( queue->get_predecessors("make3",&use) );
    REQUIRE( use->size()==1 );
    // there are two kernels that generate object3; we do not allow that.
    CHECK( (*use)[0]->get_name()=="make2" );
  }
}

TEST_CASE( "Mapping between redundant distros","[redundant][70]" ) {

  product_distribution
    *din = new product_replicated_scalar(env),
    *dout = new product_replicated_scalar(env);
  auto scalar_in = std::shared_ptr<object>( new product_object( din ) ),
    scalar_out = std::shared_ptr<object>( new product_object( dout ) );
  double
    *indata = scalar_in->get_data(mycoord),
    *outdata = scalar_out->get_data(mycoord);
  *indata = 15.3;

  product_kernel
    *copy_kernel = new product_kernel( scalar_in,scalar_out );
  copy_kernel->set_explicit_beta_distribution
    ( dout->get_environment(),dout->get_processor_structure() );
  std::shared_ptr<task> copy_task;

  REQUIRE_NOTHROW( copy_kernel->last_dependency()->ensure_beta_distribution(scalar_out ) );
  CHECK_NOTHROW( copy_kernel->split_to_tasks() );
  CHECK_NOTHROW( copy_task = copy_kernel->get_tasks().at(0) );
  CHECK_NOTHROW( copy_task->analyze_dependencies() );
  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = copy_task->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_receiver().equals(mycoord) );
  CHECK( msgs.at(0)->get_sender().coord(0)==mytid );
}

TEST_CASE( "Interpolation by restriction", "[stretch][distribution][71]" ) {
  int nlocal=10;
  product_distribution
    *target_dist = new product_block_distribution(decomp,nlocal*mpi_nprocs),
    *source_dist = new product_block_distribution(decomp,2*nlocal*mpi_nprocs);
  index_int
    myfirst = target_dist->first_index_r(mycoord).coord(0), mylast = target_dist->last_index_r(mycoord).coord(0);
  auto target = std::shared_ptr<object>( new product_object(target_dist) ),
    source = std::shared_ptr<object>( new product_object(source_dist) );
  product_kernel
    *restrict = new product_kernel(source,target);
  
  CHECK_NOTHROW( restrict->add_sigma_oper
		 (  ioperator("*2")  ) );
  std::shared_ptr<task> t;
  REQUIRE_NOTHROW( restrict->last_dependency()->ensure_beta_distribution(target) );
  CHECK_NOTHROW( restrict->split_to_tasks() );
  CHECK( restrict->get_tasks().size()==1 );
  CHECK_NOTHROW( t = restrict->get_tasks()[0] );

  {
    std::vector<ioperator*> *ops;
    CHECK_NOTHROW( ops = restrict->beta_definition->get_operators() );
    ioperator *beta_op;
    CHECK_NOTHROW( beta_op = (ioperator*) *(ops->begin()) );
    CHECK( beta_op.is_restrict_op() );

    indexstruct
      *gamma_struct = target_dist->get_processor_structure(mycoord),
      *beta_block;

    // spell it out: operate_and_breakup
    CHECK_NOTHROW( beta_block = 
        gamma_struct->operate( beta_op,source_dist->get_enclosing_structure() ) );
    CHECK( beta_block->stride()==2 );
    std::vector<message*> msgs;

    SECTION( "carefully" ) {
      msgs = source_dist->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,beta_block );
      CHECK( msgs.size()==1 );
      auto mtmp = msgs.at(0);
      auto global_struct = mtmp->get_global_struct(),
	local_struct = mtmp->get_local_struct();
      CHECK( global_struct->volume()==nlocal );
      CHECK( global_struct->first_index_r()==2*myfirst ); // src in the big alpha vec
      CHECK( ( global_struct->last_index_r()-global_struct->first_index_r() )==2*nlocal-2 );
      CHECK( mtmp->get_local_struct()->first_index_r()==domain_coordinate_zero(1) ); // tar relative to the halo
    }

    SECTION( "same thing in one go" ) {
      CHECK_NOTHROW( msgs = source_dist->analyze_one_dependence
		     (mytid,0,beta_op,target_dist,beta_block) );
    }
  }
}

index_int dup(int p,index_int i) {
  return 2*(p/2)+i;
}

index_int hlf(int p,index_int i) {
  return p/2;
}

TEST_CASE( "Multistage tree collecting, explicit distributions","[redundant][72][hide]" ) {

  product_distribution *twoper,*unique,*redund;   // alpha/gamma distros
  product_distribution *duplic; product_distribution *twosies; // beta distros
  std::shared_ptr<object> bot,mid,top;
  product_kernel *gather1,*gather2;
  auto div2 =  ioperator("/2") ;
  auto ul2 =  ioperator("x2") ;

  INFO( "mytid=" << mytid );

  twoper = new product_block_distribution(decomp,2,-1);
  bot = std::shared_ptr<object>( new product_object(twoper) );
  CHECK( bot->volume(mycoord)==2 );
  CHECK( bot->first_index_r(mycoord).coord(0)==(2*mytid) );

  SECTION( "derive the unique distro" ) {
    SECTION( "twoper->unique, explicit" ) {
      unique = new product_block_distribution(decomp,1,-1);
    }
    SECTION( "twoper->unique, recursive" ) {
      REQUIRE_NOTHROW( unique = (product_distribution*)twoper->operate( div2 ) );
    }
    mid = std::shared_ptr<object>( new product_object(unique) );
    CHECK( mid->volume(mycoord)==1 );
    CHECK( mid->first_index_r(mycoord).coord(0)==mytid );
  }

  SECTION( "take it from the unique definition" ) {
    unique = (product_distribution*)twoper->operate( div2 );
    mid = std::shared_ptr<object>( new product_object(unique) );
    gather1 = new product_kernel(bot,mid);

    SECTION( "create the redundant distro" ) {
      SECTION( "redund, explicit" ) {
	redund = new product_distribution(decomp,&hlf,1);
      }
      SECTION( "redund, recursive" ) {
	REQUIRE_NOTHROW( redund = (product_distribution*)unique->operate( div2 ) );
      }
      CHECK( redund->volume(mycoord)==1 );
      CHECK( redund->first_index_r(mycoord).coord(0)==(mytid/2) );
    }

    SECTION( "take it from the redundant" ) {
      redund = (product_distribution*)unique->operate( div2 );

      SECTION( "beta for gather1" ) {
	SECTION( "it's really the bottom alpha" ) {
	  //REQUIRE_NOTHROW( twosies = new product_distribution( *twoper ) );
	  twosies = twoper;
	}
	SECTION( "but we can also do it from mid" ) {
	  product_distribution *twicegamma,*shiftgamma;
	  REQUIRE_NOTHROW( twicegamma = (product_distribution*)unique->operate(mul2) );
	  CHECK( twicegamma->first_index_r(mycoord).coord(0)==2*unique->first_index_r(mycoord).coord(0) );
	  CHECK( twicegamma->volume(mycoord)==unique->volume(mycoord) );
	  index_int s = unique->volume(mycoord);
	  REQUIRE_NOTHROW( shiftgamma = twicegamma->operate
			   (  ioperator("shift",s)  ) );
	  REQUIRE_NOTHROW( twosies = twicegamma->distr_union( shiftgamma ) );
	}
	CHECK( twosies->first_index_r(mycoord).coord(0)==(2*mytid) );
	CHECK( twosies->volume(mycoord)==2 );
      }
      SECTION( "use the gather1 beta" ) {
	twosies = new product_distribution( *twoper );
	throw("explicit beta syntax\n"); //gather1->set_explicit_beta_distribution( twoper );
	gather1->set_name("gather-local");
	CHECK_NOTHROW( gather1->analyze_dependencies() );
	
	redund = (product_distribution*)unique->operate( div2 ); // copied from above
	top = std::shared_ptr<object>( new product_object(redund) );
	CHECK( top->volume(mycoord)==1 );
	CHECK( top->first_index_r(mycoord).coord(0)==(mytid/2) );
	CHECK( top->last_index_r(mycoord).coord(0)==top->first_index_r(mycoord).coord(0) );
	gather2 = new product_kernel(mid,top);
	
	// check the function distributions
	// duplic is the beta distribution for the gather to redundant,
	// see 16.2.4 in the everything writeup
	
	SECTION( "duplicate, explicit" ) {
	  duplic = new product_distribution(decomp,&dup,2); // this is the tricky one
	}
	SECTION( "duplicate, recursively" ) {
	  product_distribution *twicegamma,*shiftgamma;
	  REQUIRE_NOTHROW( twicegamma = (product_distribution*)redund->operate(mul2) );
	  CHECK( twicegamma->first_index_r(mycoord).coord(0)==2*redund->first_index_r(mycoord).coord(0) );
	  CHECK( twicegamma->volume(mycoord)==redund->volume(mycoord) );
	  index_int s = redund->volume(mycoord);
	  REQUIRE_NOTHROW( shiftgamma = twicegamma->operate
			   (  ioperator("shift",s)  ) );
	  REQUIRE_NOTHROW( duplic = twicegamma->distr_union( shiftgamma ) );
	}
	{ // test duplic
	  int f = mytid-mytid%2; // test the duplic beta distro
	  INFO( duplic->get_processor_structure(mycoord)->as_string() );
	  CHECK( duplic->first_index_r(mycoord).coord(0)==f );
	  CHECK( duplic->volume(mycoord)==2 );
	  CHECK( duplic->last_index_r(mycoord).coord(0)==f+1 );
	}
	throw("explicit beta syntax\n"); //gather2->set_explicit_beta_distribution( duplic );
	gather2->set_name("gather-to-redundant");
	REQUIRE_NOTHROW( gather2->analyze_dependencies() );
      }
    }
  }
}

indexstruct *doubleinterval(index_int i) {
  return new contiguous_indexstruct(2*i,2*i+1);
}

void scansum(int step,int p,std::shared_ptr<object> invector,std::shared_ptr<object> outvector,void *ctx)
{
  double *indata = invector->get_data(mycoord);
  int insize = invector->volume(p);

  double *outdata = outvector->get_data(mycoord);
  int outsize = outvector->volume(p);

  if (2*outsize!=insize) {
    printf("in/out not compatible: %d %d\n",insize,outsize); throw(6);}

  for (index_int i=0; i<outsize; i++) {
    outdata[i] = indata[2*i]+indata[2*i+1];
  }
}

TEST_CASE( "multistage tree collecting, recursive","[distribution][redundant][73]" ) {

  REQUIRE( (mpi_nprocs%4)==0 ); // for now only only perfect bisection

  // start with a bottom distribution of two points per proc
  product_distribution *twoper = new product_block_distribution(decomp,2,-1);
  auto bot = std::shared_ptr<object>( new product_object(twoper) );
  {
    CHECK( bot->volume(mycoord)==2 );
    double *botdata = bot->get_data(mycoord); botdata[0] = botdata[1] = 1.;
  }

  // mid has one point per proc, redund is two-way redundant
  auto div2 =  ioperator("/2") ;
  auto mul2 =  ioperator("x2") ;
  product_distribution *unique,*redund;
  std::shared_ptr<object> mid,top;

  REQUIRE_NOTHROW( unique = (product_distribution*)twoper->operate( div2 ) );
  REQUIRE_NOTHROW( mid = std::shared_ptr<object>( new product_object(unique) ) );
  CHECK( mid->volume(mycoord)==1 );
  CHECK( mid->first_index_r(mycoord).coord(0)==mytid );

  REQUIRE_NOTHROW( redund = (product_distribution*)unique->operate( div2 ) );
  REQUIRE_NOTHROW( top = std::shared_ptr<object>( new product_object(redund) ) );
  CHECK( top->volume(mycoord)==1 );
  CHECK( top->first_index_r(mycoord).coord(0)==(mytid/2) );
  CHECK( top->last_index_r(mycoord).coord(0)==top->first_index_r(mycoord).coord(0) );

  std::vector<std::shared_ptr<task>> tsks;
  std::shared_ptr<task> tsk; message *msg;

  // gathering from bot to mid should be local
  product_kernel *gather1;
  REQUIRE_NOTHROW( gather1 = new product_kernel(bot,mid) );
  gather1->set_localexecutefn( &scansum );
  REQUIRE_NOTHROW( gather1->set_signature_function( &doubleinterval ) );
  //gather1->set_explicit_beta_distribution( twoper );
  gather1->set_name("gather-local");
  CHECK_NOTHROW( gather1->analyze_dependencies() );
  REQUIRE_NOTHROW( tsks = gather1->get_tasks() );
  CHECK( tsks.size()==1 );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  REQUIRE_NOTHROW( tsk->get_receive_messages() );
  CHECK( tsk->get_receive_messages().size()==1 );
  REQUIRE_NOTHROW( msg = tsk->get_receive_messages().at(0) );
  CHECK( msg->get_sender().coord(0)==mytid );
  CHECK( msg->get_receiver().equals(mycoord) );
  REQUIRE_NOTHROW( gather1->execute() );
  {
    double *data = mid->get_data(mycoord); 
    CHECK( data[0]==2. );
  }


  product_kernel *gather2;
  REQUIRE_NOTHROW( gather2 = new product_kernel(mid,top) );
  gather2->set_localexecutefn( &scansum );
  REQUIRE_NOTHROW( gather2->set_signature_function( &doubleinterval ) );
  //gather2->set_explicit_beta_distribution( duplic );
  gather2->set_name("gather-to-redundant");
  REQUIRE_NOTHROW( gather2->analyze_dependencies() );
  REQUIRE_NOTHROW( tsks = gather2->get_tasks() );
  CHECK( tsks.size()==1 );
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  REQUIRE_NOTHROW( tsk->get_receive_messages().size() );
  CHECK( tsk->get_receive_messages().size()==2 );
  for (int imsg=0; imsg<2; imsg++) {
    REQUIRE_NOTHROW( msg = tsk->get_receive_messages().at(imsg) );
    if (mytid%2==0) {
      CHECK( (msg->get_sender().coord(0)==mytid || msg->get_sender().coord(0)==mytid+1) );
    } else {
      CHECK( (msg->get_sender().coord(0)==mytid || msg->get_sender().coord(0)==mytid-1) );
    }
  }
  REQUIRE_NOTHROW( gather2->execute() );
  {
    double *data = top->get_data(mycoord); 
    CHECK( data[0]==4. );
  }

}

  // // unlike in [72] we derive the beta by operations
  // // gather2 is unique => redundant
  // product_distribution *duplic,*twicegamma,*shiftgamma;
  // REQUIRE_NOTHROW( twicegamma = (product_distribution*)redund->operate(mul2) );
  // CHECK( twicegamma->first_index_r(mycoord).coord(0)==2*redund->first_index_r(mycoord).coord(0) );
  // CHECK( twicegamma->volume(mycoord)==redund->volume(mycoord) );
  // index_int s = redund->volume(mycoord);
  // REQUIRE_NOTHROW( shiftgamma = (product_distribution*)twicegamma->operate( new ioperator("shift",s) ) );
  // REQUIRE_NOTHROW( duplic = (product_distribution*)twicegamma->distr_union( shiftgamma ) );
  // {
  //   int f = mytid-mytid%2;
  //   INFO( duplic->get_processor_structure(mycoord)->as_string() );
  //   CHECK( duplic->first_index_r(mycoord).coord(0)==f );
  //   CHECK( duplic->volume(mycoord)==2 );
  //   CHECK( duplic->last_index_r(mycoord).coord(0)==f+1 ); // VLE this is the true last because indexed.
  // }

TEST_CASE( "multistage tree collecting iterated","[distribution][redundant][74]" ) {

  //  REQUIRE( (mpi_nprocs%4)==0 ); // for now only only perfect bisection
  int points_per_proc = 4;
  index_int gsize = points_per_proc*mpi_nprocs;
  int twos,nlevels;
  std::vector<int> levels;
  for (nlevels=1,twos=1; twos<=gsize; nlevels++,twos*=2)
    levels.push_back(twos);
  twos /= 2; nlevels -= 1;
  REQUIRE( twos==gsize ); // perfect bisection only
  CHECK( nlevels==levels.size() );

  // create the distributions and objects
  int tsize = gsize, itest=0;
  //snippet dividedistributions
  auto div2 =  ioperator("/2") ;
  auto mul2 =  ioperator("x2") ;
  distribution *distributions[nlevels];
  std::shared_ptr<object> objects[nlevels];
  for (int nlevel=0; nlevel<nlevels; nlevel++) {
    if (nlevel==0) {
      distributions[0]
          = new product_block_distribution(decomp,points_per_proc,-1);
    } else {
      distributions[nlevel] = distributions[nlevel-1]->operate(div2);
    }
    INFO( "level: " << nlevel << "; g=" << distributions[nlevel]->global_volume() );
    objects[nlevel] = std::shared_ptr<object>( new product_object(distributions[nlevel]) );
    //snippet end
    if (nlevel>0) {
      auto cur = objects[nlevel], prv = objects[nlevel-1];
      index_int csize = cur->volume(mycoord),psize = prv->volume(mycoord);
      if (itest==0) // test for the trivial levels:
	CHECK( (2*csize)==psize ); // doing local combines
      else { // test for the redundant levels
	INFO( "(redundant level)" );
	CHECK( csize==1 ); // one point, not necessarily unique
      }
      if (csize==1) itest++;
    }
    if (nlevel<nlevels-1)
      tsize /= 2;
  }
  CHECK( tsize==1 );

  // create kernels
  //snippet dividekernels
  product_kernel *kernels[nlevels-1];
  for (int nlevel=0; nlevel<nlevels-1; nlevel++) {
    INFO( "level: " << nlevel );
    char name[20];
    sprintf(name,"gather-%d",nlevel);
    kernels[nlevel] = new product_kernel(objects[nlevel],objects[nlevel+1]);
    kernels[nlevel]->set_name( name );
    kernels[nlevel]->set_signature_function( &doubleinterval );
    kernels[nlevel]->set_localexecutefn( &scansum );
  }
  //snippet end

  // does this work?
  double *data;
  int n;
  CHECK_NOTHROW( n = objects[0]->volume(mycoord) );
  CHECK( n==points_per_proc );
  CHECK_NOTHROW( data = objects[0]->get_data(mycoord) );
  for (int i=0; i<n; i++) 
    data[i] = 1.;
  int should=1.;
  for (int nlevel=0; nlevel<nlevels-1; nlevel++) {
    INFO( "level: " << nlevel );
    CHECK_NOTHROW( kernels[nlevel]->execute() );
    should *= 2;
    CHECK_NOTHROW( n = objects[nlevel+1]->volume(mycoord) );
    CHECK_NOTHROW( data = objects[nlevel+1]->get_data(mycoord) );
    //printf("Are we equal to %d?\n",should);
    for (int i=0; i<n; i++) {
      CHECK( data[i] == Approx(should) );
    }
  }
}

TEST_CASE( "Scale queue with gen kernel","[queue][execute][103]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=22;
  auto no_op =  ioperator("none") ;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  std::shared_ptr<algorithm> queue;
  CHECK_NOTHROW( queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) ) );

  double *xdata,*ydata;
  auto xvector = std::shared_ptr<object>( new product_object(block) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  product_kernel
    *gen_kernel = new product_kernel(xvector),
    *mult_kernel = new product_kernel(xvector,yvector);
  gen_kernel->set_localexecutefn( &vecset );
  mult_kernel->add_sigma_operator(  ioperator("none")  );
  mult_kernel->set_localexecutefn( &vecscalebytwo );

  CHECK_NOTHROW( queue->add_kernel(gen_kernel) );
  CHECK_NOTHROW( queue->add_kernel(mult_kernel) );
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK_NOTHROW( queue->execute() );

}

TEST_CASE( "Threepoint queue with gen kernel","[queue][execute][halo][modulo][104]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=17,nsteps=4;
  product_distribution *block = new product_block_distribution(decomp,nlocal*mpi_nprocs);
  index_int
    my_first = block->first_index_r(mycoord).coord(0), my_last = block->last_index_r(mycoord).coord(0);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  auto no_op =  ioperator("none") ;
  auto right_shift_mod =  ioperator(">>1") ;
  auto left_shift_mod  =  ioperator("<<1") ;

  auto xvector = std::shared_ptr<object>( new product_object(block) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = std::shared_ptr<object>( new product_object(block) );
  }

  std::shared_ptr<algorithm> queue;
  CHECK_NOTHROW( queue = std::shared_ptr<algorithm>( new product_algorithm(decomp) ) );
  {
    product_kernel *k = new product_kernel(xvector);
    k->set_localexecutefn( &vecset );
    CHECK_NOTHROW( queue->add_kernel( k ) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    product_kernel *k;
    if (iv==0) {
      k = new product_kernel(xvector,yvector[0]);
    } else {
      k = new product_kernel(yvector[iv-1],yvector[iv]);
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn( &threepointsum );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK( queue->get_tasks().size()==(nsteps+1) );
  std::vector<int> *predecessors;
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    CHECK_NOTHROW( predecessors = t->get_predecessors() );
    if (t->get_step()==0 ) {
      CHECK( predecessors.size()==0 );
    } else {
      CHECK( predecessors.size()==3 );
    }
  }

  SECTION( "jit" ) {
    CHECK_NOTHROW( queue->execute() );
  }

  SECTION( "asap" ) {
    CHECK_NOTHROW( queue->optimize() );
    CHECK_NOTHROW( queue->execute() );
  }

  {
    double *ydata;
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_data(mycoord) );
      for (i=0; i<nlocal; i++) {
	INFO( "step " << s << ", yvalue-" << i << ":" << ydata[i] );
	CHECK( ydata[i] == Approx( pow(3,s+1) ) );
      }
    }
  }
}

#endif
