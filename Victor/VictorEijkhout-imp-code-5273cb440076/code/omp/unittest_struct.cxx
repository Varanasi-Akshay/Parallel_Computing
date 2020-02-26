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
 **** unit tests for communication structure
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_static_vars.h"
#include "omp_ops.h"
#include "unittest_functions.h"
#include "imp_functions.h"

TEST_CASE( "Environment is proper","[environment][init][1]" ) {
  REQUIRE( env->get_architecture()->nprocs() > 0 );
  SECTION( "static vars" ) {
    CHECK( ntids==env->get_architecture()->nprocs() );
  }
  SECTION( "on the fly" ) {
    int ntids;
#pragma omp parallel
    ntids = omp_get_num_threads();
    CHECK( ntids==env->get_architecture()->nprocs() );
  }
  CHECK( env->get_architecture()->nprocs()==ntids );
}

TEST_CASE( "checking on index offset calculation","[index][07]" ) {

  int dim;
  SECTION( "1" ) { dim = 1; }
  //  SECTION( "2" ) { dim = 2; }
  //  SECTION( "3" ) { dim = 3; }

  INFO( "dim=" << dim );
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  decomposition *decomp; REQUIRE_NOTHROW( decomp  = new omp_decomposition(arch,layout) );
  CHECK( decomp->get_dimensionality()==dim );

  std::vector<index_int> domain;
  for (int id=0; id<dim; id++) {
    int pi;
    REQUIRE_NOTHROW( pi = layout->coord(id) );
    CHECK( pi>=1 );
    domain.push_back(10*pi);
  }
  distribution *d; std::shared_ptr<object> v;
  REQUIRE_NOTHROW( d = new omp_block_distribution(decomp,domain) );
  CHECK( d->get_dimensionality()==dim );
  REQUIRE_NOTHROW( v = std::shared_ptr<object>( new omp_object(d) ) );

  kernel *make;
  REQUIRE_NOTHROW( make = new omp_origin_kernel(v) );
  REQUIRE_NOTHROW( make->set_localexecutefn(&vecsetconstantp) );
  algorithm *maker = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( maker->add_kernel(make) );
  REQUIRE_NOTHROW( maker->analyze_dependencies() );
  REQUIRE_NOTHROW( maker->execute() );

  double *data; index_int dsize;
  REQUIRE_NOTHROW( data = v->get_raw_data() );
  REQUIRE_NOTHROW( dsize = v->get_raw_size() );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    std::shared_ptr<multi_indexstruct> pstruct,nstruct,gstruct;
    REQUIRE_NOTHROW( pstruct = v->get_processor_structure(mycoord) );
    REQUIRE_NOTHROW( nstruct = v->get_numa_structure() );
    REQUIRE_NOTHROW( gstruct = v->get_enclosing_structure() );

    index_int offset;
    REQUIRE_NOTHROW( offset = pstruct->linear_location_in(nstruct) );
    CHECK( offset<dsize );
    INFO( fmt::format("mycoord={} has struct {},\nlocated at {} in numa={}",
		      mycoord.as_string(),pstruct->as_string(),offset,nstruct->as_string()) );
    CHECK( data[offset]==Approx( (double)mytid ) );
  }
}

TEST_CASE( "Analyze single interval","[index][structure][12]" ) {
  int localsize = 5;
  omp_distribution *dist = new omp_distribution(decomp); 
  parallel_structure *pstruct;
  REQUIRE_NOTHROW( pstruct = dynamic_cast<parallel_structure*>(dist) );
  CHECK_NOTHROW( pstruct->create_from_global_size(localsize*ntids) );
  std::vector<message*> mm;
  message *m; std::shared_ptr<multi_indexstruct> s; std::shared_ptr<multi_indexstruct> segment;

  SECTION( "find an exact processor interval" ) {
    processor_coordinate *zero = new processor_coordinate_zero(1);
    REQUIRE( ntids==dist->domains_volume() );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      segment = std::shared_ptr<multi_indexstruct>
	( new multi_indexstruct
	  ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,localsize-1) ) ) );
      CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
      CHECK( mm.size()==1 );
      m = mm.at(0);
      CHECK( m->get_sender().equals( zero ) );
      CHECK( m->get_receiver().equals(mycoord) );
      REQUIRE_NOTHROW( s = m->get_global_struct() );
      CHECK( s->first_index_r()[0]==0 );
      CHECK( s->last_index_r()[0]==localsize-1 );
    }
  }

  SECTION( "find a sub interval" ) {
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      segment = std::shared_ptr<multi_indexstruct>
	( new multi_indexstruct
	  ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(1,localsize-2) ) ));
	CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
      CHECK( mm.size()==1 );
      m = mm.at(0);
      CHECK( m->get_sender().equals( new processor_coordinate_zero(1) ) );
      CHECK( m->get_receiver().equals(mycoord) );
      s = m->get_global_struct();
      CHECK( s->first_index_r()[0]==1 );
      CHECK( s->last_index_r()[0]==localsize-2 );
    }
  }

  SECTION( "something with nontrivial intersection" ) {
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      if (ntids>1) {
	std::shared_ptr<multi_indexstruct> range;
	segment = std::shared_ptr<multi_indexstruct>
	  ( new multi_indexstruct
	    ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,localsize) ) ));
	CHECK_NOTHROW( mm = dist->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
	CHECK( mm.size()==2 );
	for (int im=0; im<2; im++) { // we don't insist on ordering
	  m = mm.at(im);
	  CHECK( m->get_receiver().equals(mycoord) );
	  int snd = m->get_sender().coord(0);
	  INFO( "sender: " << snd );
	  CHECK( ( snd==0 || snd==1 ) );
	  range = m->get_global_struct();
	  if (m->get_sender().equals( new processor_coordinate_zero(1) )) {
	    CHECK( range->first_index_r()[0]==0 );
	    CHECK( range->last_index_r()[0]==localsize-1 );
	  } else {
	    CHECK( range->first_index_r()[0]==localsize );
	    CHECK( range->last_index_r()[0]==localsize );
	  }
	}
      }
    }
  }
}

TEST_CASE( "OMP distributions","[omp][distribution][replicated][13]" ) {
  
  index_int s = 100*ntids; int dim=1;
  omp_distribution *d1;

  SECTION( "block" ) {
    REQUIRE_NOTHROW( d1 = new omp_block_distribution(decomp,s) );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

      CHECK( d1->volume(mycoord)==100 );
      int f=100*mytid, l=100*(mytid+1)-1;
      CHECK( d1->first_index_r(mycoord)[0]==f );
      auto cf = domain_coordinate(std::vector<index_int>{f});
      CHECK( d1->contains_element(mycoord,cf) );
      auto f1 = domain_coordinate(std::vector<index_int>{f-1});
      CHECK( ( !d1->is_valid_index(f1) || !d1->contains_element(mycoord,f1) ) );
    
      CHECK( d1->last_index_r(mycoord)[0]==l );
      auto lm1 = domain_coordinate(std::vector<index_int>{l-1});
      CHECK( d1->contains_element(mycoord,lm1) );
      auto ds = domain_coordinate(std::vector<index_int>{s});
      CHECK( !d1->contains_element(mycoord,ds) );
      auto l1 = domain_coordinate(std::vector<index_int>{l+1});
      CHECK( ( !d1->is_valid_index(l1) || !d1->contains_element(mycoord,l1) ) );
      auto m1 = domain_coordinate(std::vector<index_int>{-1});
      CHECK( !d1->contains_element(mycoord,m1) );
    }
  }

  SECTION( "replicated" ) {
    REQUIRE_NOTHROW( d1 = new omp_replicated_distribution(decomp) );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      CHECK( d1->volume(mycoord)==1 );
      CHECK( d1->first_index_r(mycoord)==domain_coordinate_zero(dim) );
      CHECK( d1->last_index_r(mycoord)==domain_coordinate_zero(dim) );
    }
    std::shared_ptr<object> scalar;
    REQUIRE_NOTHROW( scalar = std::shared_ptr<object>( new omp_object(d1) ) );
    REQUIRE_NOTHROW( scalar->allocate() );
    double *d,*d0;
    REQUIRE_NOTHROW( d = scalar->get_data(new processor_coordinate_zero(1)) );
    d0 = d;
    for (int i=0; i<ntids; i++) {
      REQUIRE_NOTHROW( d[i] = i+1 );
    }
    INFO( "replicated storage starts at " << (long)d0 );
    for (index_int i=0; i<ntids; i++) {
      processor_coordinate c = decomp->coordinate_from_linear(i);
      REQUIRE_NOTHROW( d = scalar->get_data(c) );
      INFO( "i=" << i << "; data at " << (long)d );
      CHECK( ((long)d)==((long)(d0+i)) );
      CHECK( d[0]==Approx(i+1) );
    }
  }

  SECTION( "shifted" ) {
    index_int bottom = 37,top=37+s-1, f,l=bottom-1;
    auto domain = std::shared_ptr<indexstruct>( new contiguous_indexstruct(bottom,top) );
    parallel_structure *parallel = new parallel_structure(decomp);
    REQUIRE_NOTHROW( parallel->create_from_indexstruct(domain) );
    REQUIRE_NOTHROW( d1 = new omp_distribution(/*decomp,*/parallel) );
    CHECK( d1->global_volume()==s );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      REQUIRE_NOTHROW( f = d1->first_index_r(mycoord)[0] );
      CHECK( f==l+1 );
      REQUIRE_NOTHROW( l = d1->last_index_r(mycoord)[0] );
    }
    CHECK( l==top );
  }
}

TEST_CASE( "Analyze one dependency","[operate][dependence][modulo][16]") {
  index_int localsize = 100,gsize = localsize*ntids;
  omp_distribution *alpha = 
    new omp_block_distribution(decomp,localsize*ntids);
  ioperator shiftop;
  std::shared_ptr<multi_indexstruct> alpha_block, segment,halo;
  //  std::vector<indexstruct*> *beta_blocks;
  std::vector<message*> mm; message *m;
  index_int my_first_index,my_last_index;


  SECTION( "right bump" ) {
    shiftop = ioperator(">=1" );
    CHECK( shiftop.is_right_shift_op() );
    CHECK( !shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      int my_first = mytid*localsize,my_last = (mytid+1)*localsize;

      REQUIRE_NOTHROW( alpha_block = alpha->get_processor_structure(mycoord) );
      CHECK( alpha_block->first_index_r()[0]==my_first );
      REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure()) );
      CHECK( segment->first_index_r()[0]==my_first+1 );
      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
      if (mytid<ntids-1) {
	CHECK( mm.size()==2 );
      } else {
	CHECK( mm.size()==1 );
      }
      m = mm.at(0); 
      CHECK( m->get_sender().coord(0)==mytid );
      CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
      CHECK( m->get_global_struct()->first_index_r()[0]==my_first+1 );
      if (mytid<ntids-1) {
	m = mm.at(1);
	CHECK( m->get_sender().coord(0)==mytid+1 );
	CHECK( m->get_global_struct()->local_size(0)==1 );
	CHECK( m->get_global_struct()->first_index_r()[0]==my_last );
      }
    }
  }

  SECTION( "left bump" ) {
    shiftop = ioperator("<=1" );
    CHECK( shiftop.is_left_shift_op() );
    CHECK( !shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      int my_first = mytid*localsize,my_last = (mytid+1)*localsize;

      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->local_size(0)==localsize );
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
	    CHECK( m->get_global_struct()->local_size(0)==1 );
	  } else if ( m->get_sender().coord(0)==mytid ) {
	    CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
	  } else {
	    CHECK( 1==0 );
	  }
	}
      } else {
	m = mm.at(0);
	CHECK( m->get_sender().coord(0)==mytid );
	CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
      }
    }
  }

  SECTION( "right modulo" ) {
    if (ntids<3) { printf("Test 16 modulo needs at least 3 procs\n"); return; }
    shiftop = ioperator(">>1" );
    CHECK( shiftop.is_right_shift_op() );
    CHECK( shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      index_int
	my_first = alpha->first_index_r(mycoord)[0],
	my_last = alpha->last_index_r(mycoord)[0];
      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->local_size(0)==localsize );
      CHECK( alpha_block->first_index_r()[0]==my_first );
      CHECK( alpha_block->last_index_r()[0]==my_last );
      segment = alpha_block->operate(shiftop);
      halo = alpha_block->struct_union(segment);
      CHECK( segment->first_index_r()[0]==my_first+1 );
      CHECK( segment->last_index_r()[0]==my_last+1 );
      CHECK( halo->first_index_r()[0]==my_first );
      CHECK( halo->last_index_r()[0]==my_last+1 );
      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,halo ) );
      CHECK( mm.size()==2 );
      for ( auto m : mm ) { //int imsg=0; imsg<2; imsg++) {
	//message *m = (*mm)[imsg];
	INFO( m->get_sender().as_string() << " sends " <<
	      m->get_global_struct()->first_index_r()[0] << "--" << 
	      m->get_global_struct()->last_index_r()[0] << " to " <<
	      m->get_receiver().as_string() );
	if (m->get_sender().coord(0)==mytid) {
	  CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
	  CHECK( m->get_global_struct()->first_index_r()[0]==my_first+1 );
	  CHECK( m->get_local_struct()->first_index_r()[0]==my_first+1 );
	  //	  CHECK( m->get_local_struct()->first_index_r()[0]==1 );
	} else {
	  CHECK( m->get_sender().coord(0)==MOD(mytid+1,ntids) );
	  if (mytid==ntids-1) CHECK( m->get_sender().equals( new processor_coordinate_zero(1) ) );
	  CHECK( m->get_global_struct()->local_size(0)==1 );
	  CHECK( m->get_global_struct()->first_index_r()[0]==MOD(my_last+1,gsize) );
	  CHECK( m->get_local_struct()->first_index_r()[0]==MOD(my_last+1,gsize) );
	  //	  CHECK( m->get_local_struct()->first_index_r()[0]==localsize );
	}
      }
    }
  }

  // // this example does truncating modulo, which I'm not sure is ever needed.
  // SECTION( "left modulo" ) {
  //   shiftop = ioperator("<<1" );
  //   CHECK( shiftop.is_left_shift_op() );
  //   CHECK( shiftop.is_modulo_op() );
  //   for (int mytid=0; mytid<ntids; mytid++) {
      // processor_coordinate mycoord;
      // REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
  //     index_int my_first = alpha->first_index_r(mycoord)[0],my_last = alpha->last_index_r(mycoord)[0];

  //     alpha_block = alpha->get_processor_structure(mycoord);
  //     CHECK( alpha_block->local_size(0)==localsize );
  //     REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop,alpha->get_get_processor_structure()) );
  //     REQUIRE_NOTHROW( halo = alpha_block->struct_union(segment) );
  //     CHECK( halo->first_index_r()[0]==my_first-1 );
  //     CHECK( halo->last_index_r()[0]==my_last );
  //     CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,halo ) );
  //     CHECK( mm->size()==2 );
  //     for (int imsg=0; imsg<2; imsg++) {
  // 	message *m = (*mm)[imsg];
  // 	if (m->get_sender()==mytid) {
  // 	  CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
  // 	  CHECK( m->get_global_struct()->first_index_r()[0]==my_first );
  // 	  CHECK( m->get_local_struct()->first_index_r()[0]==1 );
  // 	} else {
  // 	  CHECK( m->get_sender()==MOD(mytid-1,ntids) );
  // 	  if (mytid==0) CHECK( m->get_sender()==(ntids-1) );
  // 	  CHECK( m->get_global_struct()->local_size(0)==1 );
  // 	  CHECK( m->get_global_struct()->first_index_r()[0]==MOD(my_first-1,gsize) );
  // 	  CHECK( m->get_local_struct()->first_index_r()[0]==0 );
  // 	}
  //     }
  //   }
  // }
}

TEST_CASE( "Analyze one dependency, shifted domain","[domain][operate][dependence][016]") {
  if (ntids<3) { printf("Test 016 needs 3 procs\n"); return; }

  index_int localsize = 100,gsize = localsize*ntids, bottom=13,top=bottom+gsize-1;
  auto domain = std::shared_ptr<indexstruct>( new contiguous_indexstruct(bottom,top) );
  parallel_structure *parallel = new parallel_structure(decomp);
  parallel->create_from_indexstruct(domain);
  omp_distribution *alpha;
  REQUIRE_NOTHROW( alpha = new omp_distribution(/*decomp,*/parallel) );
  
  ioperator shiftop;
  std::shared_ptr<multi_indexstruct> alpha_block;
  //  std::vector<indexstruct*> *beta_blocks;
  std::vector<message*> mm; message *m;
  index_int my_first_index,my_last_index;
  std::shared_ptr<multi_indexstruct> segment,halo;

  SECTION( "right bump" ) {
    shiftop = ioperator(">=1" );
    CHECK( shiftop.is_right_shift_op() );
    CHECK( !shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      INFO( "thread " << mytid );
      int my_first = bottom+mytid*localsize,my_last = bottom+(mytid+1)*localsize-1;

      alpha_block = alpha->get_processor_structure(mycoord); // should be from alpha/gamma
      CHECK( alpha_block->first_index_r()[0]==my_first );
      CHECK( alpha_block->last_index_r()[0]==my_last );
      segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure());
      CHECK( segment->first_index_r()[0]==my_first+1 );
      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,segment ) );
      if (mytid<ntids-1) {
	CHECK( mm.size()==2 );
      } else {
	CHECK( mm.size()==1 );
      }
      m = mm.at(0);
      CHECK( m->get_sender().coord(0)==mytid );
      CHECK( m->get_global_struct()->first_index_r()[0]==my_first+1 );
      CHECK( m->get_global_struct()->last_index_r()[0]==my_last );
      CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
      if (mytid<ntids-1) {
	m = mm.at(1);
	CHECK( m->get_sender().coord(0)==mytid+1 );
	CHECK( m->get_global_struct()->local_size(0)==1 );
	CHECK( m->get_global_struct()->first_index_r()[0]==my_last+1 );
      }
    }
  }

  SECTION( "left bump" ) {
    shiftop = ioperator("<=1" );
    CHECK( shiftop.is_left_shift_op() );
    CHECK( !shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      int my_first = bottom+mytid*localsize,my_last = bottom+(mytid+1)*localsize-1;

      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->local_size(0)==localsize );
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
  	    CHECK( m->get_global_struct()->local_size(0)==1 );
  	  } else if ( m->get_sender().coord(0)==mytid ) {
  	    CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
  	  } else {
  	    CHECK( 1==0 );
  	  }
  	}
      } else {
  	m = mm.at(0);
  	CHECK( m->get_sender().coord(0)==mytid );
  	CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
      }
    }
  }
}

TEST_CASE( "Analyze one dependency, shifted domain, modulo","[domain][operate][modulo][dependence][016]") {
  if (ntids<3) { printf("Test 016 needs 3 procs\n"); return; }

  index_int localsize = 100,gsize = localsize*ntids, bottom=13,top=bottom+gsize-1;
  auto domain = std::shared_ptr<indexstruct>( new contiguous_indexstruct(bottom,top) );
  parallel_structure *parallel = new parallel_structure(decomp);
  parallel->create_from_indexstruct(domain);
  omp_distribution *alpha;
  REQUIRE_NOTHROW( alpha = new omp_distribution(/*decomp,*/parallel) );
  
  ioperator shiftop;
  std::shared_ptr<multi_indexstruct> alpha_block;
  //  std::vector<indexstruct*> *beta_blocks;
  std::vector<message*> mm; message *m;
  index_int my_first_index,my_last_index;
  std::shared_ptr<multi_indexstruct> segment,halo;

  SECTION( "right modulo" ) {
    shiftop = ioperator(">>1" );
    CHECK( shiftop.is_right_shift_op() );
    CHECK( shiftop.is_modulo_op() );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      INFO( "thread " << mytid );
      index_int my_first = alpha->first_index_r(mycoord)[0],
	my_last = alpha->last_index_r(mycoord)[0];
      alpha_block = alpha->get_processor_structure(mycoord);
      CHECK( alpha_block->local_size(0)==localsize );
      CHECK( alpha_block->first_index_r()[0]==my_first );
      CHECK( alpha_block->last_index_r()[0]==my_last );

      segment = alpha_block->operate(shiftop);
      CHECK( segment->first_index_r()[0]==my_first+1 );
      CHECK( segment->last_index_r()[0]==my_last+1 );

      REQUIRE_NOTHROW( halo = alpha_block->struct_union(segment) );
      CHECK( halo->first_index_r()[0]==my_first );
      CHECK( halo->last_index_r()[0]==my_last+1 );

      CHECK_NOTHROW( mm = alpha->messages_for_segment( mycoord,self_treatment::INCLUDE,segment,halo ) );
      CHECK( mm.size()==2 );
      for ( auto m : mm ) { //int imsg=0; imsg<2; imsg++) {
  	//message *m = (*mm)[imsg];
  	INFO( m->get_sender().as_string() << " sends " <<
	      m->get_global_struct()->first_index_r()[0] << "--" << 
  	      m->get_global_struct()->last_index_r()[0] << " to " <<
	      m->get_receiver().as_string() );
  	if (m->get_sender().coord(0)==mytid) {
  	  CHECK( m->get_global_struct()->local_size(0)==localsize-1 );
  	  CHECK( m->get_global_struct()->first_index_r()[0]==my_first+1 );
  	  CHECK( m->get_local_struct()->first_index_r()[0]==m->get_global_struct()->first_index_r()[0] );
  	} else {
  	  CHECK( m->get_sender().coord(0)==MOD(mytid+1,ntids) );
  	  if (mytid==ntids-1)
	    CHECK( m->get_sender().equals( new processor_coordinate_zero(1) ) );
  	  CHECK( m->get_global_struct()->local_size(0)==1 );
  	  CHECK( m->get_global_struct()->first_index_r()[0]==MOD(my_last+1,gsize) );
  	  CHECK( m->get_local_struct()->first_index_r()[0]==m->get_global_struct()->first_index_r()[0] );
  	}
      }
    }
  }

}

TEST_CASE( "Analyze processor dependencies, right modulo, include self message",
	   "[index][message][modulo][17]") {
  if (ntids<3) { printf("Test 17 needs 3 procs\n"); return; }

  std::vector<message*> messages;
  message *m,*m0,*m1;
  index_int localsize = 10,gsize = ntids*localsize;
  auto 
    no_op       = ioperator("none" ),
    right_shift = ioperator(">>1" ),
    left_shift  = ioperator("<<1" );
  omp_distribution *d1 = 
    new omp_block_distribution(decomp,10*ntids);
  std::shared_ptr<multi_indexstruct> h,beta_block;

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    int myfirst = mytid*localsize,mylast = (mytid+1)*localsize-1;

    // send from self to self: should deliver only one message
    h = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast) ) ));
    beta_block = d1->get_processor_structure(mycoord)->operate(no_op);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,h) );
    //    CHECK_NOTHROW( messages = d1->analyze_one_dependence(mytid,0,no_op,d1,h) );
    CHECK( messages.size()==1 );
    m0 = messages.at(0);
    auto  global_struct = m0->get_global_struct(),
      local_struct = m0->get_local_struct();
    index_int f = global_struct->first_index_r()[0],l = global_struct->last_index_r()[0];
    CHECK( f==myfirst );
    CHECK( l==mylast );
    CHECK( m0->get_receiver().equals(mycoord) );
    CHECK( m0->get_sender().coord(0)==mytid );

    // // something non-trivial
    h = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1) ) ));
    beta_block = d1->get_processor_structure(mycoord)->operate(right_shift);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,h) );
    //CHECK_NOTHROW( messages = d1->analyze_one_dependence(mytid,0,right_shift,d1,h) );
    // // there are two messages because of the module shift
    CHECK( messages.size()==2 );
    for ( auto m : messages ) { //int imsg=0; imsg<2; imsg++) {
      //message *m = (*messages)[imsg];
      CHECK( m->get_receiver().equals(mycoord) );
      int s = m->get_sender().coord(0);
      if (mytid==ntids-1) {
	CHECK( ( s==0 || s==mytid ) );
      } else {
	CHECK( ( s==mytid || s==mytid+1 ) );
      }
      auto src = m->get_global_struct(), tar = m->get_local_struct();
      if (s==mytid) {
	// first message is shift of my local data
	CHECK( src->first_index_r()[0]==myfirst+1 );
	CHECK( src->last_index_r()[0]==mylast );
	CHECK( tar->first_index_r()[0]==src->first_index_r()[0] ); // no longer remapping omp halo
	CHECK( tar->last_index_r()[0]==src->last_index_r()[0] );
	// CHECK( tar->first_index_r()[0]==1 );
	// CHECK( tar->last_index_r()[0]==localsize-1 );
      } else {
	// second msg gets one element from my right neighbour
	CHECK( src->first_index_r()[0]==(mylast+1)%gsize );
	CHECK( src->last_index_r()[0]==(mylast+1)%gsize );
	CHECK( tar->first_index_r()[0]==src->first_index_r()[0] ); // no longer remapping omp halo
	CHECK( tar->last_index_r()[0]==src->last_index_r()[0] );
	// CHECK( tar->first_index_r()[0]==localsize );
	// CHECK( tar->last_index_r()[0]==localsize );
      }
    }
  }
}

TEST_CASE( "Analyze processor dependencies, right modulo, skip self",
	   "[index][message][modulo][18]") {
  if (ntids<3) { printf("Test 18 needs 3 procs\n"); return; }

  std::vector<message*> messages;
  message *m,*m0;
  index_int localsize = 10,gsize = ntids*localsize;
  auto 
    no_op       = ioperator("none" ),
    right_shift = ioperator(">>1" ),
    left_shift  = ioperator("<<1" );
  omp_distribution *d1 = 
    new omp_block_distribution(decomp,10*ntids);
  std::shared_ptr<multi_indexstruct> h,beta_block;

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    int myfirst = mytid*localsize,mylast = (mytid+1)*localsize-1;

    // send from self to self: should deliver only one message
    h = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast) ) ));
    beta_block = d1->get_processor_structure(mycoord)->operate(no_op);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::EXCLUDE,beta_block,h) );
    CHECK( messages.size()==0 ); // skip self: nothing remains
    beta_block = d1->get_processor_structure(mycoord)->operate(no_op);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,h) );
    CHECK( messages.size()==1 ); // this is the real case
    m0 = messages.at(0);
    auto global_struct = m0->get_global_struct(),
      local_struct = m0->get_local_struct();
    index_int f = global_struct->first_index_r()[0],l = global_struct->last_index_r()[0];
    CHECK( f==myfirst );
    CHECK( l==mylast );
    CHECK( m0->get_receiver().equals(mycoord) );
    CHECK( m0->get_sender().coord(0)==mytid );

    int s0;
    std::shared_ptr<multi_indexstruct> src0,tar0;

    // something non-trivial
    h = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1) ) ));
    beta_block = d1->get_processor_structure(mycoord)->operate(right_shift);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::EXCLUDE,beta_block,h) );
    // there is one message for everyone, since we're skipping self
    CHECK( messages.size()==1 );
    m0 = messages.at(0);
    CHECK( m0->get_sender().coord(0)==(mytid+1)%ntids );
    CHECK( m0->get_receiver().equals(mycoord) );
    src0 = m0->get_global_struct(); tar0 = m0->get_local_struct();
    // msg gets one element from my right neighbour, global coordinates, wrapped
    CHECK( src0->first_index_r()[0]==(mylast+1)%gsize );
    CHECK( src0->last_index_r()[0]==(mylast+1)%gsize );
    // winds up in the last location of the halo regardless
    CHECK( tar0->first_index_r()[0]==src0->first_index_r()[0] );
    CHECK( tar0->last_index_r()[0]==src0->last_index_r()[0] );
    // CHECK( tar0->first_index_r()[0]==localsize );
    // CHECK( tar0->last_index_r()[0]==localsize );

    // something non-trivial: left shift
    h = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct
	( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst-1,mylast) ) ));
    beta_block = d1->get_processor_structure(mycoord)->operate(left_shift);
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::EXCLUDE,beta_block,h) );
    // there is one message for everyone
    CHECK( messages.size()==1 );
    m0 = messages.at(0);
    CHECK( m0->get_sender().coord(0)==(mytid-1+ntids)%ntids );
    CHECK( m0->get_receiver().equals(mycoord) );
    src0 = m0->get_global_struct(); tar0 = m0->get_local_struct();
    // msg gets one element from my left neighbour
    CHECK( src0->first_index_r()[0]==(myfirst-1+gsize)%gsize );
    CHECK( src0->last_index_r()[0]==src0->first_index_r()[0] );
    // winds up in the first location of the halo regardless
    CHECK( tar0->first_index_r()[0]==src0->first_index_r()[0] ); // no longer remapping omp halo
    CHECK( tar0->last_index_r()[0]==src0->last_index_r()[0] );
    // CHECK( tar0->first_index_r()[0]==0 );
    // CHECK( tar0->last_index_r()[0]==0 );
  }
}

TEST_CASE( "Analyze processor dependencies, left bump","[index][message][19]") {
  std::vector<message*> messages;
  message *m0,*m1;
  auto 
    no_op       = ioperator("none" ),
    right_shift = ioperator(">=1" ),
    left_shift  = ioperator("<=1" );
  index_int localsize = 10;
  omp_distribution *d1;
  REQUIRE_NOTHROW( d1 = new omp_block_distribution(decomp,10*ntids) );
  std::shared_ptr<multi_indexstruct> h,beta_block;

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    int myfirst = mytid*localsize,mylast = (mytid+1)*localsize-1;

    { // slightly larger than the actual halo
      int f=myfirst,l=mylast;
      if (mytid>0) f--; if (mytid<ntids-1) l++;
      h = std::shared_ptr<multi_indexstruct>
	( new multi_indexstruct
	  ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) ) ));
    }
    CHECK_NOTHROW( beta_block = d1->get_processor_structure(mycoord)->operate
		       (left_shift,d1->get_enclosing_structure()) );
    CHECK_NOTHROW( messages = d1->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,h) );
    if (mytid==0) { // there is no message from the left
      CHECK( messages.size()==1 );
    } else { // one element from the left
      CHECK( messages.size()==2 );
    }

    if (mytid==0) { // only a shift in place to the right
      m0 = messages.at(0);
      int s0,s1;
      s0 = m0->get_sender().coord(0);
      std::shared_ptr<multi_indexstruct> src0,tar0;
      src0 = m0->get_global_struct();
      tar0 = m0->get_local_struct();

      CHECK( m0->get_receiver().equals(mycoord) );
      CHECK( s0==mytid );
      CHECK( src0->first_index_r()[0]==myfirst );
      CHECK( src0->last_index_r()[0]==mylast-1 );
      CHECK( tar0->first_index_r()[0]==0 );
      CHECK( tar0->last_index_r()[0]==localsize-2 );
    } else {
      for (int im=0; im<2; im++) {
	auto m = messages.at(im);
	auto src = m->get_global_struct(), tar = m->get_local_struct();

	CHECK( m->get_receiver().equals(mycoord) );
	//    s1 = m1->get_sender().coord(0);
	if ( m->get_sender().coord(0)==mytid-1 ) { // s0 case
	  // from left neighbour
	  CHECK( src->first_index_r()[0]==myfirst-1 );
	  CHECK( src->last_index_r()[0]==myfirst-1 );
	  CHECK( tar->first_index_r()[0]==src->first_index_r()[0] );
	  CHECK( tar->last_index_r()[0]==src->last_index_r()[0] );
	} else if (m->get_sender().coord(0)==mytid ) { // s1 case
	  // then shift of local
	  CHECK( src->first_index_r()[0]==myfirst );
	  CHECK( src->last_index_r()[0]==mylast-1 );
	  CHECK( tar->first_index_r()[0]==src->first_index_r()[0] );
	  CHECK( tar->last_index_r()[0]==src->last_index_r()[0] );
	} else {
	  CHECK( 1==0 );
	}      
      }
    }
  }
}

TEST_CASE( "Task dependencies threepoint bump","[task][message][object][21]" ) {

  // create distributions and objects for threepoint combination

  index_int g = 10*ntids;
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  omp_distribution *d1;
  REQUIRE_NOTHROW( d1 = new omp_block_distribution(decomp,g) );

  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d1) ) );
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d1) ) );
  CHECK( d1->is_known_globally() );
  CHECK( r1->is_known_globally() );

  // declare the beta vectors
  signature_function *sigma_opers;
  CHECK_NOTHROW( sigma_opers = new signature_function(/*d1*/) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( no_op ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( left_shift ) );
  CHECK_NOTHROW( sigma_opers->add_sigma_operator( right_shift ) );
  //CHECK_NOTHROW( sigma_opers->ensure_beta_distribution(d1,d1) );

  parallel_structure *preds;
  std::shared_ptr<multi_indexstruct> enclosing;
  REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
  REQUIRE_NOTHROW( preds = sigma_opers->derive_beta_structure(d1,enclosing) );
  omp_distribution *beta_dist;
  CHECK_NOTHROW( beta_dist = new omp_distribution(/*decomp,*/preds) );

  kernel *combine;
  std::shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine = new omp_kernel(o1,r1) );
  REQUIRE_NOTHROW( combine->split_to_tasks());
  CHECK_NOTHROW( combine->set_beta_distribution( 0,beta_dist ) );
  REQUIRE_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    std::shared_ptr<multi_indexstruct> dom;
    REQUIRE_NOTHROW( dom = beta_dist->get_visibility(mycoord) );
    CHECK( dom->first_index_r()[0]==0 );
    CHECK( dom->last_index_r()[0]==(g-1) );
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(mytid) );
    CHECK_NOTHROW( combine_task->derive_receive_messages(/*0,mytid*/) );
  }
}

TEST_CASE( "Task dependencies threepoint modulo","[task][message][object][modulo][22]" ) {

  // create distributions and objects for threepoint combination
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  index_int nlocal = 10, nglobal=nlocal*ntids;
  omp_distribution *d1 = 
    new omp_block_distribution(decomp,nlocal*ntids);
  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d1) ) );
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d1) ) );

  // declare the beta vectors
  signature_function *beta_opers = new signature_function(/*d1*/);
  beta_opers->add_sigma_operator( no_op );
  beta_opers->add_sigma_operator( left_shift );
  beta_opers->add_sigma_operator( right_shift );
 
  parallel_structure *preds;
  std::shared_ptr<multi_indexstruct> enclosing;
  REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
  REQUIRE_NOTHROW( preds = beta_opers->derive_beta_structure(d1,enclosing) );
  omp_distribution *beta_dist;
  CHECK_NOTHROW( beta_dist = new omp_distribution(/*decomp,*/preds) );
  CHECK( beta_dist->global_volume()==(nglobal+2) );
  CHECK( beta_dist->global_first_index()[0]==-1 );
  CHECK( beta_dist->global_last_index()[0]==nglobal );

  if( ntids<=2 ) { printf("test 22 needs >2 processors\n"); return; }

  kernel *combine;
  std::shared_ptr<task> combine_task;
  REQUIRE_NOTHROW( combine = new omp_kernel(o1,r1) );
  REQUIRE_NOTHROW( combine->split_to_tasks());
  REQUIRE_NOTHROW( combine->set_beta_distribution( 0,beta_dist ) );
  REQUIRE_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid: " << mytid );
    index_int my_first=d1->first_index_r(mycoord)[0],my_last=d1->last_index_r(mycoord)[0];
    REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(mytid) );
    REQUIRE_NOTHROW( combine_task->derive_receive_messages(/*0,mytid*/) );

    std::vector<message*> msgs;
    CHECK_NOTHROW( msgs = combine_task->get_receive_messages() );
    // one message from self, two neighbours
    CHECK( msgs.size()==3 );
    for ( auto m : msgs ) {
      //for (std::vector<message*>::iterator m=msgs->begin(); m!=msgs->end(); ++m) {
      int sender = m->get_sender().coord(0);
      auto glb = m->get_global_struct(),loc = m->get_local_struct();
      auto
	gf = glb->first_index_r(),gl = glb->last_index_r(),
	lf = loc->first_index_r(),ll = loc->last_index_r();
      // where is everything in the halo?
      if (sender==MOD(mytid+1,ntids)) { // right neighbour
	CHECK( gf[0]==MOD(my_last+1,nglobal) );
	auto gsize = gl-gf+1;
	CHECK( gsize[0]==1 );
	// we are no longer localizing OMP.....
	CHECK( lf==gf ); // (my_last+2) ); // my_last+1, shifted because of left halo
	auto lsize = ll-lf+1; 
	CHECK( lsize[0]==1 );
      } else if (sender==MOD(mytid-1,ntids)) {
	CHECK( gf[0]==MOD(my_first-1,nglobal) );
	auto gsize = gl-gf+1;
	CHECK( gsize[0]==1 );
	CHECK( lf==gf ); //(my_first) ); // my_first-1, shifted because of left halo
	auto lsize = ll-lf+1;
	CHECK( lsize[0]==1 );
      } else {
	CHECK( sender==mytid );
	CHECK( gf==my_first );
	auto gsize = gl-gf+1;
	CHECK( gsize[0]==nlocal );
	CHECK( lf==gf );
	// CHECK( lf==(my_first+1) );
	auto lsize = ll-lf+1;
	CHECK( lsize[0]==nlocal );
      }
      CHECK( d1->contains_element(m->get_sender(),gf) );
      CHECK( d1->contains_element(m->get_sender(),gl) );
    }
  }
}

TEST_CASE( "Task send structure threepoint bump","[task][message][object][23]" ) {

  // create distributions and objects for threepoint combination

  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  index_int nlocal=10,gsize=nlocal*ntids;
  omp_distribution *d1 = 
    new omp_block_distribution(decomp,gsize);

  std::shared_ptr<object> o1,r1;
  CHECK_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d1) ) );
  CHECK_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d1) ) );

  // declare the beta vectors
  signature_function *beta_opers = new signature_function(/*d1*/);

  SECTION( "no op only" ) {
    beta_opers->add_sigma_operator( no_op );
    parallel_structure *preds;
    omp_distribution *beta_dist;
    std::shared_ptr<multi_indexstruct> enclosing;
    REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
    REQUIRE_NOTHROW( preds = beta_opers->derive_beta_structure(d1,enclosing) );
    CHECK_NOTHROW( beta_dist = new omp_distribution(/*decomp,*/preds) );

    kernel *originate = new omp_kernel(o1,r1);
    std::shared_ptr<task> originate_task;
    CHECK_NOTHROW( originate->set_beta_distribution( 0,beta_dist ) );
    REQUIRE_NOTHROW( originate->last_dependency()->create_beta_vector(r1) );
    REQUIRE_NOTHROW( originate->split_to_tasks());

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      int my_first = d1->first_index_r(mycoord)[0],my_last = d1->last_index_r(mycoord)[0];
      // first a task with no predecessors
      REQUIRE_NOTHROW( originate_task = originate->get_tasks().at(mytid) );
      REQUIRE_NOTHROW( originate_task->derive_receive_messages() );
    }
  }

  SECTION( "non trivial stuff" ) {
    // add stuff to the beta definition
    beta_opers->add_sigma_operator( left_shift );
    beta_opers->add_sigma_operator( right_shift );
    beta_opers->add_sigma_operator( no_op );

    parallel_structure *preds;
    omp_distribution *beta_dist;
    std::shared_ptr<multi_indexstruct> enclosing;
    REQUIRE_NOTHROW( enclosing = o1->get_enclosing_structure() );
    REQUIRE_NOTHROW( preds = beta_opers->derive_beta_structure(d1,enclosing) );
    CHECK_NOTHROW( beta_dist = new omp_distribution(/*decomp,*/preds) );
    CHECK( beta_dist->global_volume()==gsize );
    CHECK( beta_dist->global_first_index()[0]==0 );
    CHECK( beta_dist->global_last_index()[0]==(gsize-1) );

    kernel *combine; 
    REQUIRE_NOTHROW( combine = new omp_kernel(o1,r1) );
    REQUIRE_NOTHROW( combine->split_to_tasks() );
    CHECK_NOTHROW( combine->set_beta_distribution( 0,beta_dist ) );
    REQUIRE_NOTHROW( combine->last_dependency()->create_beta_vector(r1) );

    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      int my_first,my_last;
      REQUIRE_NOTHROW( my_first = d1->first_index_r(mycoord)[0] );
      REQUIRE_NOTHROW( my_last = d1->last_index_r(mycoord)[0] );
      std::shared_ptr<task> combine_task;
      REQUIRE_NOTHROW( combine_task = combine->get_tasks().at(mytid) );
      REQUIRE_NOTHROW( combine_task->derive_receive_messages() );

      //      CHECK_THROWS( combine_task->get_nsends() ); // not for OMP?
      for ( auto m : combine_task->get_receive_messages()) {
	auto src = m->get_global_struct(),tar = m->get_local_struct();
	index_int
	  sf = src->first_index_r()[0], sl = src->last_index_r()[0],
	  tf = tar->first_index_r()[0], tl = tar->last_index_r()[0];
	auto csf = domain_coordinate(std::vector<index_int>{sf});
	CHECK( d1->contains_element(m->get_sender(),csf) );
	auto csl = domain_coordinate(std::vector<index_int>{sl});
	CHECK( d1->contains_element(m->get_sender(),csl) );
	// src coordinates are global
	if (m->get_sender().coord(0)==mytid-1) {
	  CHECK( sf==my_first-1 );
	  CHECK( sl==my_first-1 );
	} else if (m->get_sender().coord(0)==mytid+1) {
	  CHECK( sf==my_last+1 );
	  CHECK( sl==my_last+1 );
	} else {
	  CHECK( m->get_sender().coord(0)==mytid );
	  CHECK( sf==my_first );
	  CHECK( sl==my_last );
	}
	// tar coordinates are wrt the halo
	int hasleft = (mytid>0), hasright = (mytid<ntids-1);
	if (m->get_sender().coord(0)==mytid-1) { // from left
	  CHECK( tf==(my_first-1+!hasleft) );
	  CHECK( tl==tf );
	} else if (m->get_sender().coord(0)==mytid+1) {
	  CHECK( tf==(my_last+1-!hasright) );
	  CHECK( tl==tf );
	} else {
	  CHECK( tf==my_first );
	  CHECK( tl==my_last );
	}
      }
    }
  }
}

TEST_CASE( "Task execute on local data","[object][task][29]" ) {

  index_int s = 10;
  distribution *block = new omp_block_distribution(decomp,s*ntids);
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( block->volume(mycoord)==s );
  }
  auto vector = std::shared_ptr<object>( new omp_object(block) );
  REQUIRE_NOTHROW( vector->allocate() );
  REQUIRE( vector->has_data_status_allocated() );

  kernel *k = new omp_kernel(vector);
  REQUIRE_NOTHROW( k->split_to_tasks());
  REQUIRE( k->get_tasks().size()==ntids );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    std::shared_ptr<task> t;
    REQUIRE_NOTHROW( t = k->get_tasks().at(mytid) );
    CHECK( t->get_out_object()->volume(mycoord)==s );
    t->set_localexecutefn(&vector_gen);
    CHECK_NOTHROW( t->local_execute(vector) );
    double *data;
    REQUIRE_NOTHROW( data = vector->get_data(mycoord) );
    for (int i=block->first_index_r(mycoord)[0]; i<=block->last_index_r(mycoord)[0]; i++) {
      CHECK( data[i]==mytid+.5 );
    }
  }
}

TEST_CASE( "Task execute on external data","[object][task][30]" ) {

  // same test as before but now with externally allocated data
  index_int s = 13;
  auto xdata = new double[ntids*s];
  distribution *block = new omp_block_distribution(decomp,s*ntids);
  auto xvector = std::shared_ptr<object>( new omp_object(block,xdata) );

  kernel *xk = new omp_kernel(xvector);
  REQUIRE_NOTHROW( xk->split_to_tasks());
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    std::shared_ptr<task> xt;
    REQUIRE_NOTHROW( xt = xk->get_tasks().at(mytid) );
    xt->set_localexecutefn(&vector_gen);
    CHECK_NOTHROW( xt->local_execute(xvector) );
    CHECK( block->first_index_r(mycoord)[0]==mytid*s );
    CHECK( block->last_index_r(mycoord)[0]==(mytid+1)*s-1 );
    for (int i=block->first_index_r(mycoord)[0]; i<=block->last_index_r(mycoord)[0]; i++) {
      CHECK( xdata[i]==mytid+.5 );
    }
  }
}

TEST_CASE( "Execute local to local task","[task][halo][execute][31]" ) {

  index_int nlocal=15;
  auto no_op("none");
  omp_distribution *block = 
    new omp_block_distribution(decomp,nlocal*ntids);
  auto xdata = new double[nlocal];
  auto 
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<nlocal; i++)
    xdata[i] = 1.5+i;
  
  std::vector<std::shared_ptr<task>> tasks;
  std::shared_ptr<task> copy_task,scale_task;
  double *ydata;

  SECTION( "copy kernel" ) {
    omp_kernel *copy = new omp_kernel(xvector,yvector);
    copy->add_sigma_operator( no_op );
    copy->set_localexecutefn(&veccopy);
    CHECK_NOTHROW( copy->analyze_dependencies() );
    CHECK_NOTHROW( tasks = copy->get_tasks() );
    for ( auto copy_task : copy->get_tasks() ) {
      processor_coordinate mycoord; REQUIRE_NOTHROW( mycoord = copy_task->get_domain() );
      int mytid = mycoord[0];
      //CHECK_THROWS( mytid = copy_task->get_domain() );
      auto myfirst = block->first_index_r(mycoord),
	mylast = block->last_index_r(mycoord);

      INFO( "mytid=" << mytid );

      CHECK_NOTHROW( copy_task->analyze_dependencies() );
      // check that everything is in place
      CHECK_NOTHROW( copy_task->get_in_object(0) );
      std::shared_ptr<object> halo_object;
      CHECK_NOTHROW( halo_object = copy_task->get_beta_object(0) ); // we now have a halo...
      CHECK( copy_task->get_receive_messages().size()==1 );
      std::vector<message*> msgs;
      std::shared_ptr<multi_indexstruct> global_struct,local_struct;
      REQUIRE_NOTHROW( msgs = copy_task->get_receive_messages() );
      CHECK( msgs.size()==1 );
      CHECK_NOTHROW( global_struct = msgs[0]->get_global_struct() );
      CHECK_NOTHROW( local_struct = msgs[0]->get_local_struct() );
      CHECK( global_struct->first_index_r()==myfirst ); // global
      CHECK( global_struct->last_index_r()==mylast );
      CHECK( local_struct->first_index_r()==myfirst ); // local struct equals global
      CHECK( local_struct->last_index_r()==mylast );

      // // we can only execute an OMP task in the context of a kernel
      // CHECK_NOTHROW( copy_task->execute() );
      // CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
      // for (int i=0; i<nlocal; i++)
      // 	CHECK( ydata[i] == Approx( (double)(1.5+i) ) );
    }
  }

  SECTION( "scale kernel" ) {
    //  let's try the same with scaling
    omp_kernel *scale = new omp_kernel(xvector,yvector);
    scale->add_sigma_operator( no_op );
    scale->set_localexecutefn(&vecscalebytwo);

    CHECK_NOTHROW( scale->last_dependency()->ensure_beta_distribution(yvector) );
    CHECK_NOTHROW( scale->split_to_tasks() );
    CHECK_NOTHROW( tasks = scale->get_tasks() );
    CHECK_NOTHROW( scale_task = tasks.at(0) );
    CHECK_NOTHROW( scale_task->analyze_dependencies() );
    // CHECK_NOTHROW( scale_task->execute() );
    // CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
    // for (int i=0; i<nlocal; i++)
    //   CHECK( ydata[i] == Approx( 2*(1.5+i) ) );
  }
}

TEST_CASE( "Execute shift task","[task][halo][execute][32]" ) {

  index_int nlocal=15,gsize=nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,gsize);

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
    INFO( "mytid=" << mytid );

    CHECK( my_first==mytid*nlocal );
    CHECK( my_last==(mytid+1)*nlocal-1 );
  }

  auto xdata = new double[gsize];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<gsize; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel
    *shift = new omp_kernel(xvector,yvector);
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator(">=1") );
  shift->set_localexecutefn(&vecshiftleftbump);
  
  std::vector<std::shared_ptr<task>> tasks;
  double *halo_data,*ydata;

  CHECK_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK_NOTHROW( tasks = shift->get_tasks() );
  CHECK( tasks.size()>0 );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
    INFO( "mytid=" << mytid );

    std::shared_ptr<task> shift_task = tasks.at(mytid);

    CHECK_NOTHROW( shift_task->analyze_dependencies() );
    // check that everything is in place
    CHECK_NOTHROW( shift_task->get_in_object(0) );

    // No halo for OMP tasks!
    // CHECK( shift_task->get_beta_object(0)!=NULL );
    // if (mytid<ntids-1) {
    //   CHECK( shift_task->get_beta_object(0)->volume(mycoord)==nlocal+1 );
    // } else {
    //   CHECK( shift_task->get_beta_object(0)->volume(mycoord)==nlocal );
    // }

    // check recv messagegs
    std::shared_ptr<multi_indexstruct> global_struct,local_struct;
    if (mytid<ntids-1) {
      CHECK( shift_task->get_receive_messages().size()==2 );
    } else {
      CHECK( shift_task->get_receive_messages().size()==1 );
    }
    // VLE can we assume that messages are ordered like this?
    std::vector<message*> msgs;
    REQUIRE_NOTHROW( msgs = shift_task->get_receive_messages() );
    int nmsgs = msgs.size();
    if (mytid<ntids-1) 
      REQUIRE( nmsgs==2 );
    else
      REQUIRE( nmsgs==1 );
    for (int imsg=0; imsg<nmsgs; imsg++) {
      message *msg = msgs[imsg];
      CHECK_NOTHROW( global_struct = msg->get_global_struct() );
      CHECK_NOTHROW( local_struct = msg->get_local_struct() );
      if ( msg->get_sender().coord(0)==mytid) { // self message
	CHECK( global_struct->first_index_r()[0]==my_first );
	CHECK( global_struct->last_index_r()[0]==my_last );
	CHECK( local_struct->first_index_r()[0]==my_first );
	CHECK( local_struct->last_index_r()[0]==my_last );
      } else {
	//	if (mytid<ntids-1) { // the other message comes from the right
	  CHECK( msg->get_sender().coord(0)==mytid+1 );
	  CHECK( global_struct->first_index_r()[0]==my_last+1 );
	  CHECK( global_struct->last_index_r()[0]==my_last+1 );
	  CHECK( local_struct->first_index_r()[0]==my_last+1 );
	  CHECK( local_struct->last_index_r()[0]==my_last+1 );
	  // }
      }
    }
    // There are no send messages in OMP

  //   // can only execute omp tasks in kernel context
  //   CHECK_NOTHROW( shift_task->execute(xvector,yvector) );
  //   {
  //     int i;
  //     CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
  //     if (mytid==ntids-1) {
  //   	for (i=0; i<nlocal-1; i++) {
  //   	  INFO( "i=" << i << " yi=" << ydata[i] );
  //   	  CHECK( ydata[i+my_first] == Approx( pointfunc33(i+1,my_first) ) );
  //   	}
  //     } else {
  //   	for (i=0; i<nlocal; i++) {
  //   	  INFO( "i=" << i << " yi=" << ydata[i] );
  //   	  CHECK( ydata[i+my_first] == Approx( pointfunc33(i+1,my_first) ) );
  //   	}
  //     }
  //   }
  }
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  index_int nlocal=12,gsize=nlocal*ntids;
  ioperator no_op("none");
  omp_distribution *block = 
    new omp_block_distribution(decomp,gsize);
  auto xdata = new double[gsize];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<gsize; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel *scale = new omp_kernel(xvector,yvector);
  scale->add_sigma_operator( no_op );
  scale->set_localexecutefn(&vecscalebytwo);
  
  std::shared_ptr<task> scale_task;
  double *ydata;

  CHECK_NOTHROW( scale->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( scale->split_to_tasks() );
  CHECK_NOTHROW( scale->analyze_dependencies() );
  // VLE there is no kernel execute in OMP
  // CHECK_NOTHROW( scale->execute() );

  // CHECK( scale->get_tasks().size()==ntids );
 
  // for (int mytid=0; mytid<ntids; mytid++) {
      // processor_coordinate mycoord;
      // REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

  //   INFO( "mytid=" << mytid );

  //   index_int
  //     my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
  //   CHECK( my_first==mytid*nlocal );
  //   CHECK( my_last==(mytid+1)*nlocal-1 );
  //   CHECK_NOTHROW( scale_task = (omp_task*) ( *scale->get_tasks() )[mytid] );
  //   CHECK_NOTHROW( scale_task->get_beta_object(0)->get_data(new processor_coordinate_zero(1)) ); // we have a halo
  //   CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
  //   for (int i=0; i<nlocal; i++) {
  //     CHECK( ydata[i+my_first] == Approx( 2*pointfunc33(i,my_first) ) );
  //   }
  // }
}

TEST_CASE( "Shift kernel modulo","[task][kernel][modulo][execute][34]" ) {

  index_int nlocal=10,gsize=nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,nlocal*ntids);
  auto xdata = new double[gsize];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<gsize; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel
    *shift = new omp_kernel(xvector,yvector);
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator(">>1") );
  shift->set_localexecutefn(&vecshiftleftmodulo);
  
  std::vector<std::shared_ptr<task>> tasks;
  std::shared_ptr<task> shift_task;
  double *ydata;

  CHECK_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );
  CHECK( shift->get_tasks().size()==ntids );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

    INFO( "mytid=" << mytid );

    index_int
      my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
    CHECK( my_first==(mytid*nlocal) );
    CHECK( my_last==((mytid+1)*nlocal-1) );
    REQUIRE_NOTHROW( tasks = shift->get_tasks() );
    CHECK_NOTHROW( shift_task = tasks.at(mytid) );
    CHECK_NOTHROW( shift_task->analyze_dependencies() );
    // CHECK_NOTHROW( shift_task->execute() );
    // CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
    // REQUIRE( yvector->get_allocated_size()==gsize );
    // {
    //   int i;
    //   //INFO( "i:" << i <<", iglobal=" << i+my_first << " yi=" << ydata[i+my_first] );
    //   if (mytid==ntids-1) {
    //     for (i=0; i<nlocal-1; i++) {
    //       REQUIRE( (i+my_first)<gsize );
    //       CHECK( ydata[i+my_first] == Approx( pointfunc33(i+1,my_first) ) );
    //     }
    //     i = nlocal-1;
    //     REQUIRE( (i+my_first)<gsize );
    //     CHECK( ydata[i+my_first] == Approx( pointfunc33(0,0) ) );
    //   } else {
    //     for (i=0; i<nlocal; i++) {
    //       REQUIRE( (i+my_first)<gsize );
    //       CHECK( ydata[i+my_first] == Approx( pointfunc33(i+1,my_first) ) );
    //     }
    //   }
    // }
  }
}

TEST_CASE( "Shift from left kernel, message structure","[task][kernel][35]" ) {

  index_int nlocal=10,gsize = nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,gsize);
  auto xdata = new double[gsize];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<gsize; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel
    *shift = new omp_kernel(xvector,yvector);
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<=1") );
  //  shift->set_localexecutefn(&vecshiftrightbump);

  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  CHECK_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

    INFO( "mytid=" << mytid );

    index_int
      my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
    CHECK_NOTHROW( shift_task = shift->get_tasks()[mytid] );
    CHECK_NOTHROW( shift_task->analyze_dependencies() );

    // see if the halo is properly transmitted
    auto rmsgs = shift_task->get_receive_messages();
    if (mytid>0) { // everyone but the first receives from the left
      CHECK( rmsgs.size()==2 );
      for (int i=0; i<2; i++) {
	message *msg = rmsgs[i];
	auto rstruct = msg->get_local_struct(); // local wrt the halo
	if (msg->get_sender().coord(0)==mytid-1) { // msg to the right
	  CHECK( rstruct->first_index_r()[0]==(my_first-1) );
	  CHECK( rstruct->last_index_r()[0]==(my_first-1) );
	} else {
	  // CHECK( msg->get_sender().coord(0)==mytid);
	  // CHECK( rstruct->first_index_r()[0]==1 );
	  // CHECK( rstruct->last_index_r()[0]==nlocal+1 );
	}
      }
    } else {
      CHECK( rmsgs.size()==1 );
    }
  }
}

TEST_CASE( "Shift from left kernel modulo, execute","[task][kernel][execute][modulo][36]" ) {
  index_int nlocal=10,gsize=nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,gsize);
  auto xdata = new double[gsize];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int i=0; i<gsize; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel
    *shift = new omp_kernel(xvector,yvector);
  shift->add_sigma_operator( ioperator("none") );
  shift->add_sigma_operator( ioperator("<<1") );
  shift->set_localexecutefn(&vecshiftrightbump);
  
  std::shared_ptr<task> shift_task;
  double *halo_data,*ydata;

  CHECK_NOTHROW( shift->last_dependency()->ensure_beta_distribution(yvector) );
  CHECK_NOTHROW( shift->split_to_tasks() );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid=" << mytid );
  
    index_int my_first = block->first_index_r(mycoord)[0];

    CHECK_NOTHROW( shift_task = shift->get_tasks().at(mytid) );
    CHECK_NOTHROW( shift_task->analyze_dependencies() );

    // CHECK_NOTHROW( shift_task->execute() );
    // CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
    // {
    //   int i;
    //   if (mytid==0) {
    // 	i = 0;
    // 	CHECK( ydata[i+my_first] == Approx(pointfunc33(MOD(i-1,gsize),my_first)) );
    // 	for (i=1; i<nlocal; i++) {
    // 	  INFO( "i=" << i << " yi=" << ydata[i+my_first] );
    // 	  CHECK( ydata[i+my_first] == Approx(pointfunc33(i-1,my_first)) );
    // 	}
    //   } else {
    // 	for (i=0; i<nlocal; i++) {
    // 	  INFO( "i=" << i << " yi=" << ydata[i+my_first] );
    // 	  CHECK( ydata[i+my_first] == Approx(pointfunc33(i-1+my_first,0)) );
    // 	}
    //   }
    // }
  }
}

/*
  We set the beta equal the alpha, which leads to only one message, which is fine for omp.
  For MPI the beta should be truly non-disjoint, but that doesn't apply to OpenMP.
  Is this clever or is this abuse?
*/
TEST_CASE( "Test explicit beta bump","[beta][distribution][40]" ) {

  int localsize=15,gsize = localsize*ntids;
  omp_distribution *block = new omp_block_distribution(decomp,gsize);
  int
    iscont = block->has_type(distribution_type::CONTIGUOUS),
    isblok = block->has_type(distribution_type::BLOCKED);
  CHECK( ( iscont || isblok ) );
  std::shared_ptr<multi_indexstruct> bigblock;
  REQUIRE_NOTHROW( bigblock = block->get_enclosing_structure() );
  distribution *left,*right,*wide;

  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");

  // left
  REQUIRE_NOTHROW( left = block->operate_trunc(left_shift,bigblock ) );
  CHECK( left->get_is_orthogonal() );
  CHECK( left->has_type_locally_contiguous() );
  CHECK( !left->has_type(distribution_type::UNDEFINED) );
  // right
  REQUIRE_NOTHROW( right = block->operate_trunc(right_shift,bigblock ) );
  CHECK( !right->has_type(distribution_type::UNDEFINED) );
  // union
  REQUIRE_NOTHROW( wide = block->operate(no_op) );

  auto
    in = std::shared_ptr<object>( new omp_object(block) ),
    out = std::shared_ptr<object>( new omp_object(block) );
  in->set_name("in40"); out->set_name("out40");
  REQUIRE_NOTHROW( in->allocate() );
  REQUIRE( in->has_data_status_allocated() );
  double *indata; REQUIRE_NOTHROW( indata = in->get_raw_data() );
  for (index_int i=0; i<gsize; i++) indata[i] = 2.;

  omp_algorithm *summing; REQUIRE_NOTHROW( summing = new omp_algorithm(decomp) );
  REQUIRE_NOTHROW( summing->add_kernel( new omp_origin_kernel(in) ) );

  omp_kernel *threepoint;
  REQUIRE_NOTHROW( threepoint = new omp_kernel(in,out) );
  threepoint->set_localexecutefn(  &threepointsumbump );
  threepoint->set_exec_trace_level();
  threepoint->set_name("40threepoint");
  REQUIRE_NOTHROW( threepoint->set_explicit_beta_distribution(wide) );
  REQUIRE_NOTHROW( summing->add_kernel(threepoint) );
  REQUIRE_NOTHROW( summing->analyze_dependencies() );
  REQUIRE_NOTHROW( summing->execute() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = threepoint->get_tasks() );
  CHECK( tasks.size()==ntids );
  for ( auto threetask : tasks ) {
    std::vector<message*> rmsgs;
    processor_coordinate dom(1); REQUIRE_NOTHROW( dom = threetask->get_domain() );
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    // if (dom[0]==0 || dom[0]==ntids-1)
    //   CHECK( rmsgs->size()==2 );
    // else
    CHECK( rmsgs.size()==1 );
  }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

    // check halo
    std::shared_ptr<object> halo; double *halodata;
    REQUIRE_NOTHROW( halo = threepoint->last_dependency()->get_beta_object() );
    REQUIRE_NOTHROW( halodata = halo->get_data(mycoord) );
    index_int
      hf = out->first_index_r(mycoord)[0],
      hl = out->last_index_r(mycoord)[0];
    for (index_int i=hf; i<=hl; i++) {
      INFO( "i=" << i << " data[i]=" << halodata[i] );
      CHECK( halodata[i] == Approx(2.) );
    }
    // check output
    CHECK( out->volume(mycoord)==localsize );
    double *outdata = out->get_data(mycoord);
    index_int
      f = out->first_index_r(mycoord)[0],
      l = out->last_index_r(mycoord)[0];
    // for (index_int i=f; i<=l; i++) {
    //   INFO( "i=" << i << " data[i]=" << outdata[i] );
    //   CHECK( outdata[i] == Approx(6.) );
    // }
  }
}

TEST_CASE( "Test explicit beta mod","[beta][distribution][modulo][41]" ) {

  int localsize=15,gsize = localsize*ntids;
  omp_distribution *block = new omp_block_distribution(decomp,gsize);
  int
    iscont = block->has_type(distribution_type::CONTIGUOUS),
    isblok = block->has_type(distribution_type::BLOCKED);
  CHECK( ( iscont || isblok ) );
  distribution *left,*right,*wide;

  // left
  REQUIRE_NOTHROW( left = block->operate( ioperator("<<1") ) );
  CHECK( left->get_is_orthogonal() );
  CHECK( left->has_type_locally_contiguous() );
  CHECK( !left->has_type(distribution_type::UNDEFINED) );
  // right
  REQUIRE_NOTHROW( right = block->operate( ioperator(">>1") ) );
  CHECK( !right->has_type(distribution_type::UNDEFINED) );
  // union
  REQUIRE_NOTHROW( wide = left->distr_union(right) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int 
      my_first = block->first_index_r(mycoord)[0],
      my_last = block->last_index_r(mycoord)[0];
    CHECK( block->first_index_r(mycoord)[0]==my_first );
    CHECK( left->first_index_r(mycoord)[0]==my_first-1 );
    CHECK( block->first_index_r(mycoord)[0]==my_first );
    CHECK( right->first_index_r(mycoord)[0]==my_first+1 );
    CHECK( wide->first_index_r(mycoord)[0]==my_first-1 ); // distributions can stick out
    CHECK( wide->last_index_r(mycoord)[0]==my_last+1 );
  }

  auto
    in = std::shared_ptr<object>( new omp_object(block) ),
    out = std::shared_ptr<object>( new omp_object(block) );
  in->set_name("in40"); out->set_name("out40");
  REQUIRE_NOTHROW( in->allocate() );
  REQUIRE( in->has_data_status_allocated() );
  auto indata = in->get_raw_data();
  for (index_int i=0; i<gsize; i++) indata[i] = 2.;

  omp_algorithm *summing = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( summing->add_kernel( new omp_origin_kernel(in) ) );

  omp_kernel *threepoint;
  REQUIRE_NOTHROW( threepoint = new omp_kernel(in,out) );
  threepoint->set_localexecutefn(  &threepointsummod );
  threepoint->set_name("40threepoint");
  REQUIRE_NOTHROW( threepoint->set_explicit_beta_distribution(wide) );
  REQUIRE_NOTHROW( summing->add_kernel(threepoint) );
  REQUIRE_NOTHROW( summing->analyze_dependencies() );
  REQUIRE_NOTHROW( summing->execute() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = threepoint->get_tasks() );
  CHECK( tasks.size()==ntids );
  for ( auto threetask : tasks ) {
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    CHECK( rmsgs.size()==3 );
  }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );

    // check halo
    std::shared_ptr<object> halo; double *halodata;
    REQUIRE_NOTHROW( halo = threepoint->last_dependency()->get_beta_object() );
    REQUIRE_NOTHROW( halodata = halo->get_data(mycoord) );
    index_int
      hf = out->first_index_r(mycoord)[0],
      hl = out->last_index_r(mycoord)[0];
    for (index_int i=hf; i<=hl; i++) {
      INFO( "i=" << i << " data[i]=" << halodata[i] );
      CHECK( halodata[i] == Approx(2.) );
    }
    // check output
    CHECK( out->volume(mycoord)==localsize );
    double *outdata = out->get_data(mycoord);
    index_int
      f = out->first_index_r(mycoord)[0],
      l = out->last_index_r(mycoord)[0];
    // for (index_int i=f; i<=l; i++) {
    //   INFO( "i=" << i << " data[i]=" << outdata[i] );
    //   CHECK( outdata[i] == Approx(6.) );
    // }
  }
}

#if 0
TEST_CASE( "special matrices","[kernel][spmvp][46]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, g = ntids*nlocal;
  distribution *blocked =
    new omp_block_distribution(decomp,g);
  index_int my_first = blocked->first_index_r(mycoord)[0], my_last = blocked->last_index_r(mycoord)[0];
  omp_sparse_matrix *A;

  SECTION( "lower diagonal" ) {
    REQUIRE_NOTHROW( A = new omp_lowerbidiagonal_matrix( blocked, 1,0 ) );
    if (mytid==0)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto
	x = std::shared_ptr<object>( new omp_object(blocked) ),
	y = std::shared_ptr<object>( new omp_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new omp_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto *tasks = k->get_tasks();
      CHECK( tasks->size()==1 );
      for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies()->size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  if (mytid==ntids-1)
	    CHECK( send->size()==1 );
	  else
	    CHECK( send->size()==2 );
	  auto recv = (*t)->get_receive_messages();
	  if (mytid==0)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data;
      REQUIRE_NOTHROW( data = objs[0]->get_raw_data() ); //decomp->coord(0)) );
      for (index_int i=0; i<nlocal; i++) data[i] = 0.;
      if (mytid==0) data[0] = 1.;
      algorithm *queue = new omp_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(objs[0]) ) );
      for (int istep=1; istep<g; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_raw_data() );//decomp->coord(0)) );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==istep) CHECK( data[i-my_first]==Approx(1.) );
	  else          CHECK( data[i-my_first]==Approx(0.) );
	}
      }
    }
  }

  SECTION( "upper diagonal" ) {
    REQUIRE_NOTHROW( A = new omp_upperbidiagonal_matrix( blocked, 0,1 ) );
    if (mytid==ntids-1)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto
	x = std::shared_ptr<object>( new omp_object(blocked) ),
	y = std::shared_ptr<object>( new omp_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new omp_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto *tasks = k->get_tasks();
      CHECK( tasks->size()==1 );
      for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies()->size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  if (mytid==0)
	    CHECK( send->size()==1 );
	  else
	    CHECK( send->size()==2 );
	  auto recv = (*t)->get_receive_messages();
	  if (mytid==ntids-1)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_raw_data() );
      for (index_int i=0; i<nlocal; i++) data[i] = 0.;
      if (mytid==ntids-1) data[nlocal-1] = 1.;
      algorithm *queue = new omp_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(objs[0]) ) );
      for (int istep=1; istep<g; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_raw_data() );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==g-1-istep) CHECK( data[i-my_first]==Approx(1.) );
	  else            CHECK( data[i-my_first]==Approx(0.) );
	}
      }
    }
  }

  SECTION( "toeplitz" ) {
    REQUIRE_NOTHROW( A = new omp_toeplitz3_matrix( blocked, 0,2,0 ) );
    CHECK( A->nnzeros()==3*nlocal-(mytid==0)-(mytid==ntids-1) );

    SECTION( "kernel analysis" ) {
      auto
	x = std::shared_ptr<object>( new omp_object(blocked) ),
	y = std::shared_ptr<object>( new omp_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new omp_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto *tasks = k->get_tasks();
      CHECK( tasks->size()==1 );
      for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies()->size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  CHECK( send->size()==3-(mytid==0)-(mytid==ntids-1) );
	  auto recv = (*t)->get_receive_messages();
	  CHECK( send.size()==3-(mytid==0)-(mytid==ntids-1) );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_raw_data() );
      for (index_int i=0; i<nlocal; i++) data[i] = 1.;
      //      if (mytid==ntids-1) data[nlocal-1] = 1.;
      algorithm *queue = new omp_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(objs[0]) ) );
      int nsteps = 5;
      for (int istep=1; istep<nsteps; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );

      REQUIRE_NOTHROW( data = objs[nsteps-1]->get_raw_data() );
      for (index_int i=my_first; i<=my_last; i++) {
	INFO( "index " << i );
	CHECK( data[i-my_first]==Approx(pow(2,nsteps-1)) );
      }
    }
  }
}
#endif

TEST_CASE( "Kernel data graph: test acyclicity","[kernel][50]" ) {

  int localsize = 20,gsize = localsize*ntids;
  distribution *block = new omp_block_distribution(decomp,gsize);
  std::shared_ptr<object> object1,object2,object3,object4;
  omp_kernel *kernel1,*kernel2,*kernel3;
  omp_algorithm *queue;
  char *object_name,*test_name;

  SECTION( "unnamed objects" ) {
    object1 = std::shared_ptr<object>( new omp_object(block) );
    object2 = std::shared_ptr<object>( new omp_object(block) );
    object3 = std::shared_ptr<object>( new omp_object(block) );
    object4 = std::shared_ptr<object>( new omp_object(block) );

    kernel1 = new omp_kernel(object1,object2);
    kernel1->set_name("50kernel1");
    kernel2 = new omp_kernel(object1,object3);
    kernel2->set_name("50kernel2");
    kernel3 = new omp_kernel(object2,object3);
    kernel3->set_name("50kernel3");

    queue = new omp_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel3) );

  }

  SECTION( "named objects" ) {
    object1 = std::shared_ptr<object>( new omp_object(block) );
    object1->set_name("object1");
    object2 = std::shared_ptr<object>( new omp_object(block) );
    object2->set_name("object2");
    object3 = std::shared_ptr<object>( new omp_object(block) );
    object3->set_name("object3");
    object4 = std::shared_ptr<object>( new omp_object(block) );
    object4->set_name("object4");

    kernel1 = new omp_kernel(object1,object2);
    kernel1->set_name("make2");
    kernel2 = new omp_kernel(object1,object3);
    kernel2->set_name("make3");
    kernel3 = new omp_kernel(object2,object3);
    kernel3->set_name("make4");

    queue = new omp_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel3) );

    kernel *def;
    std::vector<kernel*> *use;
    // there are two kernels that use object1, none that generate it
    REQUIRE_NOTHROW( queue->get_data_relations("object1",&def,&use) );
    CHECK( def==NULL );
    CHECK( use->size()==2 );
    // there are two kernels that generate object3; we do not allow that.
    REQUIRE_THROWS( queue->get_data_relations("object3",&def,&use) );
  }

  SECTION( "kernel predecessors" ) {
    object1 = std::shared_ptr<object>( new omp_object(block) );
    object1->set_name("object1");
    object2 = std::shared_ptr<object>( new omp_object(block) );
    object2->set_name("object2");
    object3 = std::shared_ptr<object>( new omp_object(block) );
    object3->set_name("object3");
    object4 = std::shared_ptr<object>( new omp_object(block) );
    object4->set_name("object4");

    kernel1 = new omp_kernel(object1,object2);
    kernel1->last_dependency()->set_type_local();
    kernel1->set_name("make2");
    kernel2 = new omp_kernel(object2,object3);
    kernel2->set_name("make3");
    kernel2->last_dependency()->set_type_local();

    queue = new omp_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel(kernel1) );
    REQUIRE_NOTHROW( queue->add_kernel(kernel2) );

    // kernel *def;
    // std::vector<std::shared_ptr<task>> *ktasks;
    // REQUIRE_NOTHROW( queue->analyze_dependencies() ); // this makes pred connections
    // // CHECK_NOTHROW( kernel1->analyze_dependencies() );
    // // CHECK_NOTHROW( kernel2->analyze_dependencies() );
    // REQUIRE_NOTHROW( ktasks = kernel2->get_tasks() );
    // for (std::vector<std::shared_ptr<task>>::iterator t=ktasks->begin(); t!=ktasks->end(); ++t) {
    //   std::vector<int> *preds;
    //   REQUIRE_NOTHROW( preds = (*t)->get_predecessor_numbers() );
    //   REQUIRE( preds->size()==1 );
    //   std::shared_ptr<task> pt;
    //   REQUIRE_NOTHROW( pt = queue->get_tasks().at( preds->at(0) ) );
    //   const char *n = pt->get_name();
    //   CHECK( strlen(n)>=5 );
    //   CHECK( !strncmp( n, "make2",5 ) );
    // }
  }
}

TEST_CASE( "Mapping between redundant distros","[redundant][70]" ) {

  omp_distribution
    *din = new omp_replicated_distribution(decomp),
    *dout = new omp_replicated_distribution(decomp);
  auto
    scalar_in = std::shared_ptr<object>( new omp_object( din ) ),
    scalar_out = std::shared_ptr<object>( new omp_object( dout ) );
  auto
    indata = scalar_in->get_raw_data(),
    outdata = scalar_out->get_raw_data();
  *indata = 15.3;

  omp_kernel
    *copy_kernel = new omp_kernel( scalar_in,scalar_out );
  copy_kernel->set_explicit_beta_distribution( dout );

  CHECK_NOTHROW( copy_kernel->split_to_tasks() );
  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = copy_kernel->get_tasks() );
  for ( auto copy_task : tasks ) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = copy_task->get_domain() );
    int mytid = mycoord[0];
    CHECK_NOTHROW( copy_task->analyze_dependencies() );
    CHECK( copy_task->get_receive_messages().size()==1 );
    CHECK( copy_task->get_receive_messages()[0]->get_receiver().equals(mycoord) );
    CHECK( copy_task->get_receive_messages()[0]->get_sender().equals(mycoord) );
  }
}

TEST_CASE( "Interpolation by restriction", "[stretch][distribution][71][hide]" ) {
  index_int nlocal=10;
  omp_distribution
    *target_dist = new omp_block_distribution(decomp,nlocal*ntids),
    *source_dist = new omp_block_distribution(decomp,2*nlocal*ntids);
  auto
    target = std::shared_ptr<object>( new omp_object(target_dist) ),
    source = std::shared_ptr<object>( new omp_object(source_dist) );
  omp_kernel
    *restrict = new omp_kernel(source,target);
  
  CHECK_NOTHROW( restrict->add_sigma_operator( ioperator("*2") ) );

  std::vector<multi_ioperator*> ops;
  CHECK_NOTHROW( ops = restrict->last_dependency()->get_operators() );
  multi_ioperator *beta_op;
  CHECK_NOTHROW( beta_op = /* (multi_ioperator*) */ *(ops.begin()) );
  CHECK( beta_op->is_restrict_op() );

  std::vector<std::shared_ptr<task>> tasks;
  CHECK_NOTHROW( restrict->split_to_tasks() );
  CHECK( restrict->get_tasks().size()==ntids );
  REQUIRE_NOTHROW( tasks = restrict->get_tasks() );
  for ( auto restrict_task : tasks ) {
    processor_coordinate mycoord = restrict_task->get_domain();
    int mytid = mycoord[0];
    auto
      myfirst = target_dist->first_index_r(mycoord),
      mylast = target_dist->last_index_r(mycoord);

    auto gamma_struct = target_dist->get_processor_structure(mycoord);
    std::shared_ptr<multi_indexstruct> beta_block;

    // spell it out: operate_and_breakup
    CHECK_NOTHROW( beta_block = 
        gamma_struct->operate( beta_op,source_dist->get_enclosing_structure() ) );
    CHECK( beta_block->stride()->coord(0)==2 );
    auto msgs = 
      source_dist->messages_for_segment( mycoord,self_treatment::INCLUDE,beta_block,beta_block );
    CHECK( msgs.size()==1 );
    auto mtmp = msgs.at(0);
    auto global_struct = mtmp->get_global_struct(),
      local_struct = mtmp->get_local_struct();
    CHECK( global_struct->first_index_r()==myfirst*2 ); // src in the big alpha vec
    // local size counts the number of actual elements
    CHECK( global_struct->local_size(0)==nlocal );
    // but they span 2*nlocal-1
    CHECK( ( global_struct->last_index_r()[0]-global_struct->first_index_r()[0]+1 )==2*nlocal-1 );
    // VLE ??? CHECK( local_struct->first_index()->at(0)==myfirst ); // tar relative to the halo

    // // same thing in one go
    // CHECK_NOTHROW( msgs = source_dist->analyze_one_dependence
    //     		   (mytid,0,beta_op,target_dist,beta_block) );
  }
}

TEST_CASE( "Queue relationships","[queue][100]" ) {

  int nlocal=20, nglobal=nlocal*ntids;
  omp_distribution *block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,nglobal) );
  omp_algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );
  auto
    o1 = std::shared_ptr<object>( new omp_object(block) ),
    o2 = std::shared_ptr<object>( new omp_object(block) ),
    o3 = std::shared_ptr<object>( new omp_object(block) );

  SECTION( "short queue relationships" ) {
    kernel *origin,*process,*final;
    REQUIRE_NOTHROW( origin = new omp_kernel(o1) );
    REQUIRE_NOTHROW( process = new omp_kernel(o1,o2) );
    CHECK_NOTHROW( process->last_dependency()->set_type_local() );
    REQUIRE_NOTHROW( final = new omp_kernel(o2,o3) );
    CHECK_NOTHROW( final->last_dependency()->set_type_local() );

    SECTION( "adding kernels in the right order" ) {
      REQUIRE_NOTHROW( queue->add_kernel(origin) );
      REQUIRE_NOTHROW( queue->add_kernel(process) );
      REQUIRE_NOTHROW( queue->add_kernel(final) );
    }
    SECTION( "adding kernels in reverse order" ) {
      REQUIRE_NOTHROW( queue->add_kernel(final) );
      REQUIRE_NOTHROW( queue->add_kernel(process) );
      REQUIRE_NOTHROW( queue->add_kernel(origin) );
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );

    int k1,k2,k3;
    CHECK_NOTHROW( k1 = o1->get_object_number() );
    CHECK_NOTHROW( k2 = o2->get_object_number() );
    CHECK_NOTHROW( k3 = o3->get_object_number() );

    std::vector<std::shared_ptr<task>> tsks;
    CHECK_NOTHROW( tsks = origin->get_tasks() );
    for ( auto t : tsks ) {
      int np;
      std::vector<task_id*> preds;
      CHECK_NOTHROW( preds = t->get_predecessor_coordinates() );
      CHECK( preds.size()==0 );
    }

    CHECK_NOTHROW( tsks = process->get_tasks() );
    for ( auto t : tsks ) {
      int np,mytid;
      CHECK_NOTHROW( mytid = t->get_domain().coord(0) );
      std::vector<task_id*> preds;
      CHECK_NOTHROW( preds = t->get_predecessor_coordinates() );
      CHECK( preds.size()==1 );
      CHECK( preds.at(0)->get_domain().coord(0)==mytid );
      CHECK( preds.at(0)->get_step()==k1 );
    }

    CHECK_NOTHROW( tsks = final->get_tasks() );
    for ( auto t : tsks ) {
      int np,mytid;
      CHECK_NOTHROW( mytid = t->get_domain().coord(0) );
      std::vector<task_id*> preds;
      CHECK_NOTHROW( preds = t->get_predecessor_coordinates() );
      CHECK( preds.size()==1 );
      CHECK( preds.at(0)->get_domain().coord(0)==mytid );
      CHECK( preds.at(0)->get_step()==k2 );
    }
  }
}

TEST_CASE( "Scale queue","[queue][execute][105]" ) {

  //object_data::set_trace_create_data();
  int nlocal=17,nsteps=1,nglobal=nlocal*ntids;
  REQUIRE( nglobal>0 );
  ioperator no_op("none");
  omp_distribution *block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,nglobal) );
  algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );

  for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid=" << mytid );
    index_int
      my_first = block->first_index_r(mycoord)[0],
      my_last = block->last_index_r(mycoord)[0];
    CHECK( my_first==mytid*nlocal );
    CHECK( my_last==(mytid+1)*nlocal-1 );
  }
  
  auto xdata = new double[nglobal];
  std::shared_ptr<object> xvector;
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  REQUIRE_NOTHROW( xvector = std::shared_ptr<object>( new omp_object(block,xdata) ) );
  REQUIRE_NOTHROW( xvector->set_name("xvector") );
  CHECK( xvector->has_data_status_allocated() );
  for (int i=0; i<nglobal; i++)
    xdata[i] = pointfunc33(i,0/*my_first*/);
  for (int iv=0; iv<nsteps; iv++) {
    INFO( iv );
    REQUIRE_NOTHROW( yvector[iv] = std::shared_ptr<object>( new omp_object(block) ) );
    REQUIRE_NOTHROW( yvector[iv]->set_name(fmt::format("vector{}",iv)) );
    REQUIRE_NOTHROW( yvector[iv]->allocate() ); printf("lose this allocate!\n");
  }

  omp_kernel *k = new omp_kernel(xvector);
  k->set_name( "generate" );
  k->set_localexecutefn(&vecnoset);
  CHECK_NOTHROW( queue->add_kernel(k) );

  for (int iv=0; iv<nsteps; iv++) {
    omp_kernel *k; char name[20];
    INFO( "step: " << iv );
    if (iv==0) {
      REQUIRE_NOTHROW( k = new omp_kernel(xvector,yvector[0]) );
    } else {
      REQUIRE_NOTHROW( k = new omp_kernel(yvector[iv-1],yvector[iv]) );
    }
    sprintf(name,"update-%d",iv);
    k->set_name( name );
    k->set_localexecutefn(&vecscalebytwo);
    k->add_sigma_operator( no_op );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }

  CHECK_NOTHROW( queue->analyze_dependencies() );
  std::vector<std::shared_ptr<task>> exits;
  REQUIRE_NOTHROW( exits = queue->get_exit_tasks() );
  CHECK( exits.size()==ntids );

  std::vector<std::shared_ptr<task>> predecessors;
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  CHECK( tsks.size()==ntids*(nsteps+1) );
  for ( auto tsk : tsks ) {
    std::vector<message*> rmsgs,smsgs;
    int step = tsk->get_step();
    CHECK( step>=0 );
    //    CHECK( step<=nsteps );
    CHECK_NOTHROW( predecessors = tsk->get_predecessors() );
    INFO( "step: " << step );
    CHECK_NOTHROW( rmsgs = tsk->get_receive_messages() );
    CHECK_NOTHROW( smsgs = tsk->get_send_messages() );
    if (tsk->has_type_origin()) {
      //CHECK( tsk->get_n_outstanding_requests()==0 );
      CHECK( predecessors.size()==0 );
      CHECK( rmsgs.size()==0 );
      CHECK( smsgs.size()==0 );
    } else {
      //CHECK( tsk->get_n_outstanding_requests()==1 );
      CHECK( predecessors.size()==1 );
      CHECK( rmsgs.size()==1 );
      CHECK( smsgs.size()==1 );
    }
  }

  CHECK_NOTHROW( queue->execute() );
  CHECK( queue->get_all_tasks_executed() );

  // let's inspect some halos
  int found=0;
  for ( auto tsk : queue->get_tasks() ) {
    if (tsk->get_step()==yvector[0]->get_object_number()) {
      found++;
      std::shared_ptr<object> h;
      REQUIRE_NOTHROW( h=tsk->get_beta_object(0) );
      for (int mytid=0; mytid<ntids; mytid++) {
	processor_coordinate mycoord;
	REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
	CHECK( h->volume(mycoord)==nlocal );
      }
    }
  }
  REQUIRE( found==ntids );
  
  {
    double *ydata;
    CHECK_NOTHROW( ydata = yvector[nsteps-1]->get_raw_data() );
    CHECK( ydata!=nullptr );
    for (int i=0; i<nglobal; i++) {
      INFO( "yvalue[" << i << "] = " << ydata[i] );
      CHECK( ydata[i] == Approx( pow(2,nsteps)*pointfunc33(i,0/*my_first*/) ) );
    }
  }
}

TEST_CASE( "Threepoint queue bump","[queue][execute][halo][106]" ) {

  int nlocal=17,nsteps=1,nglobal=nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,nlocal*ntids);

  ioperator no_op("none"), right_shift_bmp(">=1"), left_shift_bmp("<=1");

  auto xdata = new double[nglobal];
  double *ydata;
  auto xvector = std::shared_ptr<object>( new omp_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  for (int i=0; i<nglobal; i++)
    xdata[i] = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = std::shared_ptr<object>( new omp_object(block) );
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector[nsteps-1]->get_object_number() );

  algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );
  {
    omp_kernel *k = new omp_kernel(xvector);
    k->set_localexecutefn(  &vecnoset );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    omp_kernel *k;
    if (iv==0) {
      k = new omp_kernel(xvector,yvector[0]);
    } else {
      k = new omp_kernel(yvector[iv-1],yvector[iv]);
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_bmp );
    k->add_sigma_operator( right_shift_bmp );
    k->set_localexecutefn(  &threepointsumbump );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  { // do all kernels have a proper step number?
    std::vector<kernel*> *kernels;
    REQUIRE_NOTHROW( kernels = queue->get_kernels() );
    for (std::vector<kernel*>::iterator k=kernels->begin(); k!=kernels->end(); ++k) {
      int kn,kv;
      {
	REQUIRE_NOTHROW( kv = (*k)->get_out_object()->get_object_number() );
	REQUIRE_NOTHROW( kn = (*k)->get_step() );
	CHECK( kn==kv );
      }
    }
  }

  const char *path;
  SECTION( "analysis unoptimized" ) { path = "analysis unoptimized";

    SECTION( "single analysis call" ) {
      CHECK_NOTHROW( queue->analyze_dependencies() );
      std::vector<std::shared_ptr<task>> exits;
      REQUIRE_NOTHROW( exits = queue->get_exit_tasks() );
      CHECK( exits.size()==ntids );
    }

    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks.size()==(ntids*(nsteps+1)) );
    std::vector<std::shared_ptr<task>> predecessors;
    for ( auto t : tsks ) {
      CHECK_NOTHROW( predecessors = t->get_predecessors() );
      INFO( "step=" << t->get_step() );
      auto rmsgs = t->get_receive_messages(), smsgs = t->get_send_messages();
      if (t->has_type_origin()) {
	CHECK( predecessors.size()==0 );
	CHECK( rmsgs.size()==0 );
	CHECK( smsgs.size()==0 );
      } else {
	auto d = t->get_domain();
	INFO( "domain: " << d.as_string() );
	std::shared_ptr<object> invector;
	REQUIRE_NOTHROW( invector = t->get_in_object(0) );
	REQUIRE_NOTHROW( invector->get_numa_structure()->equals
			 (invector->get_enclosing_structure()) );
	if (d.is_on_face(decomp)) {
	  CHECK( predecessors.size()==2 );
	  CHECK( rmsgs.size()==2 );
	  CHECK( smsgs.size()==2 );
	} else {
	  CHECK( predecessors.size()==3 );
	  CHECK( rmsgs.size()==3 );
	  CHECK( smsgs.size()==3 );
	}
	for ( auto rmsg : rmsgs ) {
	  domain_coordinate i(1), f = block->first_index_r(d), l = block->last_index_r(d);
	  if (rmsg->volume()==1) {
	    REQUIRE_NOTHROW( i = rmsg->get_local_struct()->first_index_r() );
	    INFO( "msg received into " << i.as_string() );
	    CHECK( ( ( i==f-1 ) || (i==l+1) ) ); // receive our left-1 or right+1
	  } else {
	    REQUIRE( rmsg->get_global_struct()->equals( rmsg->get_local_struct() ) );
	  }
	}
	for ( auto smsg : smsgs ) {
	  domain_coordinate i(1), f = block->first_index_r(d), l = block->last_index_r(d);
	  if (smsg->volume()==1) {
	    REQUIRE_NOTHROW( i = smsg->get_local_struct()->first_index_r() );
	    INFO( "msg received into " << i.as_string() );
	    CHECK( ( ( i==f ) || (i==l) ) );
	  } else {
	    INFO( "send globally " << smsg->get_global_struct()->as_string()
		  << " is locally " <<  smsg->get_local_struct()->as_string() );
	    REQUIRE( smsg->get_global_struct()->equals( smsg->get_local_struct() ) );
	  }
	}
      }
    }

  }
  
  // SECTION( "analysis with optimization" ) {
    
  //   CHECK_NOTHROW( queue->analyze_dependencies() );
  //   CHECK_NOTHROW( queue->optimize() );

  //   std::vector<std::shared_ptr<task>> *tsks;
  //   REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  //   for ( auto t : *tsks ) {
  //     INFO( "step=" << t->get_step() );
  //     CHECK( t->get_send_messages()->size()==0 );
  //     CHECK( t->get_receive_messages()->size()==0 );
  //     if (t->get_out_object()->get_object_number()==last_object_number) {
  // 	CHECK( t->get_post_messages()->size()==0 );
  // 	CHECK( t->get_xpct_messages()->size()==0 );
  //     } else {
  // 	CHECK( t->get_post_messages()->size()==3 );
  // 	CHECK( t->get_xpct_messages()->size()==3 );
  //     }
  //   }

  // }
  
  INFO( "analysis: " << path );

  CHECK_NOTHROW( queue->execute() );
  CHECK( queue->get_all_tasks_executed() );

  // let's inspect some halos
  int found=0;
  for (auto t : queue->get_tasks() ) {
    if (t->get_step()==yvector[0]->get_object_number()) {
      found++;
      std::shared_ptr<object> h; double *data;
      REQUIRE_NOTHROW( h = t->get_beta_object(0) );
      REQUIRE_NOTHROW( data = h->get_raw_data() );
      CHECK( h->global_volume()==ntids*nlocal );
      for (index_int i=0; i<h->global_volume(); i++) {
	INFO( "index in raw halo data: " << i );
	CHECK( data[i]==Approx(1.) );
      }
      for (int mytid=0; mytid<ntids; mytid++) {
	INFO( "p=" << mytid );
	processor_coordinate mycoord;
	REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
	index_int
	  start_own = mytid*nlocal, last_own = (mytid+1)*nlocal-1,
	  f = h->first_index_r(mycoord)[0], l = h->last_index_r(mycoord)[0];
	if (mytid==0) {
	  index_int ls = nlocal+1;
	  CHECK( h->volume(mycoord)==ls );
	  CHECK( f==start_own );
	  CHECK( l==last_own+1 );
	  CHECK( data[f]==Approx(1.) );
	  CHECK( data[l]==Approx(1.) );
	} else if (mytid==ntids-1) {
	  index_int ls = nlocal+1;
	  CHECK( h->volume(mycoord)==ls );
	  CHECK( f==(start_own-1) );
	  CHECK( l==last_own );
	  CHECK( data[f]==Approx(1.) );
	  CHECK( data[l]==Approx(1.) );
	} else {
	  index_int ls = nlocal+2;
	  CHECK( h->volume(mycoord)==ls );
	  CHECK( f==(start_own-1) );
	  CHECK( l==last_own+1 );
	  CHECK( data[f]==Approx(1.) );
	  CHECK( data[l]==Approx(1.) );
	}
      }
    }
  }
  REQUIRE( found==ntids );
  
  // investigate output:
  // the edges are increasingly not a threefold sum.
  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_raw_data() );
      //      printf("found data at %ld\n",(long)ydata);
      for (i=s+1; i<nglobal-s-1; i++) {
  	INFO( "step " << s << ", yvalue[" << i << "]=" << ydata[i] );
  	CHECK( ydata[i] == Approx( pow(3,s+1) ) );
      }
    }
  }

}

TEST_CASE( "Threepoint queue modulo","[queue][execute][halo][modulo][107]" ) {

  int nlocal=20,nsteps=1,nglobal=nlocal*ntids;
  omp_distribution *block = 
    new omp_block_distribution(decomp,nlocal*ntids);

  ioperator no_op("none"), right_shift_mod(">>1"), left_shift_mod("<<1");

  auto xdata = new double[nglobal];
  double *ydata;
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) );
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  //  printf("origin object %d\n",xvector->get_object_number());
  for (int i=0; i<nglobal; i++)
    xdata[i] = 1.;
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = std::shared_ptr<object>( new omp_object(block) );
  }
  int last_object_number;
  REQUIRE_NOTHROW( last_object_number = yvector[nsteps-1]->get_object_number() );

  algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );
  {
    omp_kernel *k = new omp_kernel(xvector);
    k->set_localexecutefn(  &vecnoset );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    omp_kernel *k;
    if (iv==0) {
      k = new omp_kernel(xvector,yvector[0]);
    } else {
      k = new omp_kernel(yvector[iv-1],yvector[iv]);
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn(  &threepointsummod );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  { // do all kernels have a proper step number?
    std::vector<kernel*> *kernels;
    REQUIRE_NOTHROW( kernels = queue->get_kernels() );
    for (std::vector<kernel*>::iterator k=kernels->begin(); k!=kernels->end(); ++k) {
      int kn,kv;
      {
	REQUIRE_NOTHROW( kv = (*k)->get_out_object()->get_object_number() );
	REQUIRE_NOTHROW( kn = (*k)->get_step() );
	CHECK( kn==kv );
      }
    }
  }

  SECTION( "analysis unoptimized" ) {

    SECTION( "single analysis call" ) {
      CHECK_NOTHROW( queue->analyze_dependencies() );
      std::vector<std::shared_ptr<task>> exits;
      REQUIRE_NOTHROW( exits = queue->get_exit_tasks() );
      CHECK( exits.size()==ntids );
    }

    std::vector<std::shared_ptr<task>> tsks;
    REQUIRE_NOTHROW( tsks = queue->get_tasks() );
    CHECK( tsks.size()==(ntids*(nsteps+1)) );
    std::vector<std::shared_ptr<task>> predecessors;
    for ( auto t : tsks ) { // (auto t=tsks->begin(); t!=tsks->end(); ++t) {
      CHECK_NOTHROW( predecessors = t->get_predecessors() );
      INFO( "step=" << t->get_step() );
      if (t->has_type_origin()) {
	CHECK( predecessors.size()==0 );
	CHECK( t->get_receive_messages().size()==0 );
      } else {
	CHECK( predecessors.size()==3 );
	CHECK( t->get_receive_messages().size()==3 );
      }
    }

  }
  
  // SECTION( "analysis with optimization" ) {
    
  //   CHECK_NOTHROW( queue->analyze_dependencies() );
  //   CHECK_NOTHROW( queue->optimize() );

  //   std::vector<std::shared_ptr<task>> *tsks;
  //   REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  //   for (std::vector<std::shared_ptr<task>>::iterator t=tsks->begin(); t!=tsks->end(); ++t) {
  //     INFO( "step=" << (*t)->get_step() );
  //     CHECK( (*t)->get_send_messages()->size()==0 );
  //     CHECK( (*t)->get_receive_messages()->size()==0 );
  //     if ((*t)->get_out_object()->get_object_number()==last_object_number) {
  // 	CHECK( (*t)->get_post_messages()->size()==0 );
  // 	CHECK( (*t)->get_xpct_messages()->size()==0 );
  //     } else {
  // 	CHECK( (*t)->get_post_messages()->size()==3 );
  // 	CHECK( (*t)->get_xpct_messages()->size()==3 );
  //     }
  //   }

  // }
  
  CHECK_NOTHROW( queue->execute() );
  CHECK( queue->get_all_tasks_executed() );

  // let's inspect some halos
  printf("[107] some halo tests disabled\n");
  int found=0;
  for (auto t=queue->get_tasks().begin(); t!=queue->get_tasks().end(); ++t) {
    if ((*t)->get_step()==yvector[0]->get_object_number()) {
      found++;
      std::shared_ptr<object> h; double *data; int f0;
      REQUIRE_NOTHROW( h=(*t)->get_beta_object(0) );
      CHECK( h->outer_size()==(nglobal+2) );
      REQUIRE_NOTHROW( data = h->get_raw_data() );
      //for (int i=0; i<nglobal+2; i++) printf("%e ",data[i]); printf("\n");
      for (int mytid=0; mytid<ntids; mytid++) {
	processor_coordinate mycoord;
	REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
	index_int start_own = mytid*nlocal, f,l;
	l = h->volume(mycoord);
	CHECK( l==(nlocal+2) );
	f = h->first_index_r(mycoord)[0];
	CHECK( f==(start_own-1) );
	// if (mytid==0) f0 = f; // this was never correct.....
	// for (index_int i=f; i<f+l; i++) {
	//   INFO( "task " << mytid << " @" << i );
	//   CHECK( data[i-f]==Approx(1.) );
	// }
      }
      break;
    }
  }
  REQUIRE( found>0 );

  {
    for (int s=0; s<nsteps; s++) {
      int i;
      CHECK_NOTHROW( ydata = yvector[s]->get_raw_data() );
      for (i=0; i<nglobal; i++) {
	if (i>s && i<nglobal-1-s) {
	  INFO( "step " << s << ", y[" << i << "]= " << ydata[i] );
	  CHECK( ydata[i] == Approx( pow(3,s+1) ) );
	}
      }
    }
  }
}

TEST_CASE( "Scale queue analysis with embedding","[queue][embed][execute][110]" ) {

  architecture *aa; omp_decomposition *dd; int power; const char *path;
  REQUIRE_NOTHROW( aa = env->make_architecture() );

  SECTION( "safe mode" ) { path = "safe";
    power = 0;
  }
  SECTION( "power mode" ) { path = "power";
    power = 1;
    aa->set_power_mode();
  }
  INFO( "using " << path << " mode" );

  REQUIRE_NOTHROW( dd = new omp_decomposition( dynamic_cast<omp_architecture*>(aa) ) );
  int nlocal=22;
  ioperator no_op("none");
  omp_distribution *block = 
    new omp_block_distribution(dd,nlocal*ntids);

  omp_algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(dd) );

  double *xdata,*ydata;
  auto
    xvector = std::shared_ptr<object>( new omp_object(block) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  if ( power==0 ) {
    CHECK( !block->get_can_embed_in_beta() );
    CHECK( xvector->has_data_status_allocated() );
    CHECK( yvector->has_data_status_allocated() );
  } else {
    CHECK( aa->get_can_embed_in_beta() );
    CHECK( block->get_can_embed_in_beta() );
    CHECK( !xvector->has_data_status_allocated() );
    CHECK( !yvector->has_data_status_allocated() );
  }
  omp_kernel *gen_kernel,*mult_kernel;
  REQUIRE_NOTHROW( gen_kernel = new omp_origin_kernel(xvector) );
  REQUIRE_NOTHROW( mult_kernel = new omp_kernel(xvector,yvector) );
  gen_kernel->set_localexecutefn(  &vecset );
  mult_kernel->add_sigma_operator( ioperator("none") );
  mult_kernel->set_localexecutefn(  &vecscalebytwo );

  CHECK_NOTHROW( queue->add_kernel(gen_kernel) );
  CHECK_NOTHROW( queue->add_kernel(mult_kernel) );
  
  //  CHECK( ( !aa->get_can_embed_in_beta() || !xvector->has_data_status_allocated() ) );
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK( xvector->has_data_status_allocated() );
  CHECK_NOTHROW( queue->execute() );
  // OMP only receives, so one msg per proc; also msg==synchronization,
  // so embedding doesn't eliminate this
  printf("reinstate tests for messages sent\n");
  //  CHECK( queue->get_nmessages_sent()==aa->nprocs() );
}

TEST_CASE( "test embedding by memory counting","[kernel][halo][embed][120]" ) {
  double allocated = env->get_allocated_space();
  INFO( "allocated at unittest 120 start: " << allocated );
  index_int nlocal = 100, nglobal = ntids*nlocal;
  int do_embed = 0;
  distribution *block; std::shared_ptr<object> xvector,yvector; algorithm *queue;
  SECTION( "allocate at once" ) { do_embed = 0;
    CHECK( !arch->get_can_embed_in_beta() );
    block = new omp_block_distribution(decomp,nlocal,-1);
    queue = new omp_algorithm(decomp);
  }
  SECTION( "allocate later" ) { do_embed = 1;
    omp_architecture *aa; REQUIRE_NOTHROW( aa = new omp_architecture(*arch) );
    REQUIRE_NOTHROW( aa->set_can_embed_in_beta() );
    CHECK( aa->get_can_embed_in_beta() );
    omp_decomposition *dd; REQUIRE_NOTHROW( dd = new omp_decomposition(aa) );
    block = new omp_block_distribution(dd,nlocal,-1);
    queue = new omp_algorithm(dd);
  }
  xvector = std::shared_ptr<object>( new omp_object(block) );
  if (do_embed) {
    CHECK( !xvector->has_data_status_allocated() );
    CHECK( env->get_allocated_space()==Approx(allocated) );
  } else {
    CHECK( xvector->has_data_status_allocated() );
    CHECK( env->get_allocated_space()==Approx(allocated+nglobal) );
  }
  yvector = std::shared_ptr<object>( new omp_object(block) );
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(xvector) ) );
  REQUIRE_NOTHROW( queue->add_kernel( new omp_copy_kernel(xvector,yvector) ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  if (do_embed)
    CHECK( env->get_allocated_space()==Approx(allocated+2*nglobal) );
  else
    CHECK( env->get_allocated_space()==Approx(allocated+3*nglobal) );

}

TEST_CASE( "test local executability for scaling","[queue][sync][150]" ) {

  // copied from [105]
  int nlocal=17,nsteps=5,nglobal=nlocal*ntids;
  REQUIRE( nglobal>0 );
  ioperator no_op("none");
  omp_distribution *block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,nglobal) );
  omp_algorithm *queue; // we will use omp-specific routines.
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );

  auto xdata = new double[nglobal];
  std::shared_ptr<object> xvector;
  auto yvector = std::vector<std::shared_ptr<object>>(nsteps);
  //auto yvector = new omp_object*[nsteps];
  REQUIRE_NOTHROW( xvector = std::shared_ptr<object>( new omp_object(block,xdata) ) );
  for (int i=0; i<nglobal; i++)
    xdata[i] = pointfunc33(i,0/*my_first*/);
  for (int iv=0; iv<nsteps; iv++) {
    INFO( iv );
    REQUIRE_NOTHROW( yvector[iv] = std::shared_ptr<object>( new omp_object(block) ) );
  }

  // make origin kernel
  CHECK_NOTHROW( queue->add_kernel( new omp_origin_kernel(xvector) ) );

  // make bunch of scale kernels
  for (int iv=0; iv<nsteps; iv++) {
    omp_kernel *k; char name[20];
    INFO( "step: " << iv );
    if (iv==0) {
      REQUIRE_NOTHROW( k = new omp_kernel(xvector,yvector[0]) );
    } else {
      REQUIRE_NOTHROW( k = new omp_kernel(yvector[iv-1],yvector[iv]) );
    }
    k->set_name( fmt::format("update-{}",iv) );
    k->set_localexecutefn(&vecscalebytwo);
    k->add_sigma_operator( no_op );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }

  CHECK_NOTHROW( queue->split_to_tasks() );
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  CHECK( tsks.size()==ntids*(nsteps+1) );

  SECTION( "traditional" ) {
    CHECK_NOTHROW( queue->determine_locally_executable_tasks() );
    CHECK( !queue->get_has_synchronization_tasks() );
    for ( auto t : tsks ) {
      CHECK( t->get_local_executability()==task_local_executability::YES );
    }
  }
  SECTION( "synchronized" ) {
    CHECK_NOTHROW( queue->set_outer_as_synchronization_points() );
    int nsync = 0;
    for ( auto t : tsks ) {
      auto d = t->get_domain();
      INFO( "task " << t->get_name() );
      if ( t->has_type_origin() && d.is_on_face(decomp) ) {
	CHECK( t->get_is_synchronization_point() ); nsync++;
      } else {
	CHECK( !t->get_is_synchronization_point() );
      }
    }
    CHECK( nsync==2 );
    CHECK_NOTHROW( queue->analyze_kernel_dependencies() );
    CHECK_NOTHROW( queue->find_predecessors() );
    CHECK_NOTHROW( queue->determine_locally_executable_tasks() );
    //    CHECK_NOTHROW( queue->analyze_dependencies() );
    int st;
    REQUIRE_NOTHROW( st = queue->get_has_synchronization_tasks() );
    CHECK( st>0 );

    for ( auto t : tsks ) {
      auto d = t->get_domain();
      INFO( "task " << t->get_name() );
      if (d.is_on_face(decomp))
	CHECK( t->get_local_executability()==task_local_executability::NO );
      else
	CHECK( t->get_local_executability()==task_local_executability::YES );
    }
  }
}

TEST_CASE( "test local executability; threepoint with sync corners","[queue][sync][151]" ) {

  // copied from [105]
  int nlocal=20,nsteps=5,nglobal=nlocal*ntids;
  REQUIRE( nglobal>0 );
  ioperator no_op("none"), right_shift(">=1"), left_shift("<=1");
  omp_distribution *block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,nglobal) );
  omp_algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );

  auto xdata = new double[nglobal];
  std::shared_ptr<object> xvector, yvector;
  REQUIRE_NOTHROW( xvector = std::shared_ptr<object>( new omp_object(block,xdata) ) );
  for (int i=0; i<nglobal; i++)
    xdata[i] = pointfunc33(i,0/*my_first*/);
  REQUIRE_NOTHROW( yvector = std::shared_ptr<object>( new omp_object(block) ) );

  CHECK_NOTHROW( queue->add_kernel( new omp_origin_kernel(xvector) ) );
  kernel *k;
  REQUIRE_NOTHROW( k = new omp_kernel(xvector,yvector) );
  k->add_sigma_operator( no_op );
  k->add_sigma_operator( right_shift );
  k->add_sigma_operator( left_shift );
  k->set_localexecutefn( &threepointsumbump );
  CHECK_NOTHROW( queue->add_kernel(k) );

  std::vector<kernel*> *kerns; REQUIRE_NOTHROW( kerns = queue->get_kernels() );
  for (auto k=kerns->begin(); k!=kerns->end(); ++k) {
    CHECK_NOTHROW( (*k)->split_to_tasks() );
    std::vector<std::shared_ptr<task>> tsks; REQUIRE_NOTHROW( tsks = (*k)->get_tasks() );
    for ( auto t : tsks ) {
      if (t->has_type_origin()) {
	auto d = t->get_domain();
	if (d.is_on_face(decomp)) {
	  t->set_is_synchronization_point();
	}
      }
    }
  }

  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK_NOTHROW( queue->determine_locally_executable_tasks() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( auto t : tsks ) {
    auto d = t->get_domain();
    INFO( "task at domain " << d.as_string() );
    if (t->has_type_origin()) {
      if (d.is_on_face(decomp))
	CHECK( t->get_local_executability()==task_local_executability::NO );
      else
	CHECK( t->get_local_executability()==task_local_executability::YES );
    } else {
      if (d.coord(0)<=1 || d.coord(0)>=ntids-2)
	CHECK( t->get_local_executability()==task_local_executability::NO );
      else
	CHECK( t->get_local_executability()==task_local_executability::YES );
    }
  }

  REQUIRE_NOTHROW( queue->execute() );
  CHECK( queue->get_all_tasks_executed() );

  double *ydata;
  REQUIRE_NOTHROW( ydata = yvector->get_raw_data() );
  for (index_int i=0; i<nglobal; i++) {
    double s = pointfunc33(i,0);
    if (i>0) s += pointfunc33(i-1,0);
    if (i<nglobal-1) s += pointfunc33(i+1,0);
    INFO( "yvalue[" << i << "] should be " << s << ", not " << ydata[i] );
    CHECK( ydata[i]==Approx(s) );
  }
}

TEST_CASE( "multi-dimensional kernels","[multi][kernel][200]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }
  int dim = 2;
  int ntids_i,ntids_j;
  for (int n=sqrt(ntids); n>=1; n--)
    if (ntids%n==0) {
      ntids_i = ntids/n; ntids_j = n; break; }
  if (ntids_i==1 || ntids_j==1) { printf("Could not split processor grid\n"); return; }

  omp_decomposition *mdecomp;
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  CHECK( (*layout)[0]==ntids_i );
  CHECK( (*layout)[1]==ntids_j );
  REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );
  processor_coordinate mycoord;
  for (int mytid=0; mytid<ntids; mytid++) {
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    int mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
    {
      processor_coordinate chkcoord;
      REQUIRE_NOTHROW( chkcoord = processor_coordinate(mytid,*mdecomp) );
      CHECK( chkcoord.coord(0)==mytid_i );
      CHECK( chkcoord.coord(1)==mytid_j );
    }
  }
  int nlocal = 10; indexstruct *idx; index_int g;
  std::vector<index_int> domain;
  g = ntids_i*(nlocal+1); domain.push_back(g);
  g = ntids_j*(nlocal+2); domain.push_back(g);

  omp_distribution *d;
  REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain) );

  std::shared_ptr<multi_indexstruct> local_domain; // we test this in struct[100]
  for (int mytid=0; mytid<ntids; mytid++) {
    int mytid_j = mytid%ntids_j, mytid_i = mytid/ntids_j;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( local_domain = d->get_processor_structure(mycoord) );
    for (int id=0; id<dim; id++)
      CHECK( local_domain->local_size(id)==nlocal+id+1 );
  }

  std::shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d) ) );
  REQUIRE_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d) ) );

  { // various tools for making multi kernels
    parallel_structure *pidx;
    REQUIRE_NOTHROW( pidx = new parallel_structure(d) );
  }
  kernel *copy;
  REQUIRE_NOTHROW( copy = new omp_kernel(o1,r1) );
  const char *sigma;
  SECTION( "explicit beta" ) { sigma = "explicit beta";
    REQUIRE_NOTHROW( copy->last_dependency()->set_explicit_beta_distribution(d) );
  }
  printf("restore [200] sigma function\n");
  // SECTION( "sigma function id" ) { sigma = "id function";
  //   REQUIRE_NOTHROW( copy->last_dependency()->set_signature_function_function
  // 		     ( new multi_sigma_operator
  // 		       ( dim, [] (std::shared_ptr<multi_indexstruct> i) -> multi_indexstruct* {
  // 			 return i->make_clone(); } ) ) );
  //   //	   ( [] (index_int i) -> indexstruct* { return new contiguous_indexstruct(i); } ) );
  // }
  INFO( "sigma specified by: " << sigma );
  REQUIRE_NOTHROW( copy->set_localexecutefn( &veccopy ) );

  algorithm *copy_queue;
  REQUIRE_NOTHROW( copy_queue = new omp_algorithm(mdecomp) );

  kernel *make;
  REQUIRE_NOTHROW( make = new omp_origin_kernel(o1) );
  REQUIRE_NOTHROW( make->set_localexecutefn(&vecsetconstantp) );
  REQUIRE_NOTHROW( copy_queue->add_kernel(make) );

  REQUIRE_NOTHROW( copy_queue->add_kernel( new omp_origin_kernel(o1) ) );
  REQUIRE_NOTHROW( copy_queue->add_kernel(copy) );

  REQUIRE_NOTHROW( copy_queue->analyze_dependencies() );
  {
    std::shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = copy->get_tasks().at(0) );
    std::vector<message*> msgs;
    REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
    CHECK( msgs.size()==1 );
  }

  REQUIRE_NOTHROW( copy_queue->execute() );

  double *data; index_int l;
  // inspect the input
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = make->get_tasks() );
  CHECK( tsks.size()==ntids );
  for ( auto t : tsks ) {
    auto dom = t->get_domain();
    INFO( "input on " << dom.as_string() );
    REQUIRE_NOTHROW( data = o1->get_data(dom) );
    REQUIRE_NOTHROW( l = o1->location_of_first_index(*o1.get(),dom) );
    INFO( " origin first index at " << l << "; data = " << data[l] );
    REQUIRE( data[l]==Approx( (double)(dom.linearize(mdecomp)) ) );
  }

  // inspect the halo
  tsks = copy->get_tasks(); CHECK( tsks.size()==ntids );
  auto 
    halo = copy->get_beta_object(0), copy_in = copy->get_in_object(0);
  for ( auto t : tsks ) {
    auto dom = t->get_domain();
    INFO( "input on " << dom.as_string() );
    REQUIRE_NOTHROW( data = copy_in->get_data(dom) );
    REQUIRE_NOTHROW( l = copy_in->location_of_first_index(*copy_in,dom) );
    INFO( " input first index at " << l << "; data = " << data[l] );
    REQUIRE( data[l]==Approx( (double)(dom.linearize(mdecomp)) ) );
  }
  for ( auto t : tsks ) {
    auto dom = t->get_domain();
    INFO( "halo on " << dom.as_string() );
    REQUIRE_NOTHROW( data = halo->get_data(dom) );
    REQUIRE_NOTHROW( l = halo->location_of_first_index(*halo.get(),dom) );
    INFO( " halo first index at " << l << "; data = " << data[l] );
    REQUIRE( data[l]==Approx( (double)(dom.linearize(mdecomp)) ) );
  }

  // inspect output
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "output on " << mytid );
    processor_coordinate mycoord; double *data;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );

    REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
    REQUIRE_NOTHROW( l = halo->location_of_first_index(*halo.get(),mycoord) );
    INFO( " output first index at " << l << "; data = " << data[l] );
    REQUIRE( data[l]==Approx( (double)mytid ) );
  }
}

TEST_CASE( "multi-dimensional reverse","[multi][kernel][201]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  int dim = 2;
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  omp_decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );
  int
    ntids_i = layout->coord(0), ntids_j = layout->coord(1);

  int nlocal = 10; indexstruct *idx;
  //index_int g;
  std::vector<index_int> domain{ntids_i*(nlocal+1),ntids_j*(nlocal+2)};

  omp_distribution *d;
  REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain) );

  std::shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d) ) );
  REQUIRE_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d) ) );

  // make the structure of the beta distribution
  parallel_structure *transstruct;
  REQUIRE_NOTHROW( transstruct = new parallel_structure(mdecomp) );
  CHECK( transstruct->get_dimensionality()==dim );
  transstruct->set_name("transposed structure");

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    int
      mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
  
    std::vector<int> flip_coords{
      layout->coord(0)-mycoord.coord(0)-1,layout->coord(1)-mycoord.coord(1)-1 };
    REQUIRE_NOTHROW( flipcoord = processor_coordinate( flip_coords ) );
    int fliptid; REQUIRE_NOTHROW( fliptid = flipcoord.linearize(layout) );
    INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	  "; flipped " << fliptid << "=" << flipcoord.as_string() );

    processor_coordinate pflip;
    REQUIRE_NOTHROW( pflip = processor_coordinate
		     ( std::vector<int>{layout->coord(0)-mycoord.coord(0)-1,
			 layout->coord(1)-mycoord.coord(1)-1} ) );
    REQUIRE_NOTHROW
      ( transstruct->set_processor_structure(mycoord,d->get_processor_structure(pflip)) );
    // fmt::print("coord {} from {}: <<{}>>\n",
    // 	       mycoord.as_string(),pflip->as_string(),
    // 	       transstruct->get_processor_structure(mycoord)->as_string()
    // 	       );
  }
  REQUIRE_NOTHROW( transstruct->set_type(distribution_type::BLOCKED) );
  CHECK( transstruct->has_type_blocked() );

  // // just checking
  // printf("Getting transstruct\n");
  // for (int ip=0; ip<ntids; ip++) {
  //   processor_coordinate pcoord = mdecomp->coordinate_from_linear(ip);
  //   std::shared_ptr<multi_indexstruct> str = transstruct->get_processor_structure(pcoord);
  //   fmt::print("{} struct : {}\n",pcoord.as_string(),str->as_string());
  // }

  omp_distribution *transdist;
  REQUIRE_NOTHROW( transdist = new omp_distribution(transstruct) );
  CHECK( transdist->get_dimensionality()==dim );
  {
    distribution_type transtype;
    REQUIRE_NOTHROW( transtype = transdist->get_type() );
    INFO( "transdist type " << (int)(transtype) <<
    	  ",  s/b " << (int)(distribution_type::BLOCKED) );
    CHECK( transdist->has_type_blocked() );
  }

  kernel *flip,*make;
  algorithm *reverse;
  REQUIRE_NOTHROW( reverse = new omp_algorithm(mdecomp) );
  double *data;

  REQUIRE_NOTHROW( make = new omp_origin_kernel(o1) );
  REQUIRE_NOTHROW( make->set_localexecutefn(&vecsetconstantp) );
  REQUIRE_NOTHROW( reverse->add_kernel(make) );

  REQUIRE_NOTHROW( flip = new omp_kernel(o1,r1) );
  REQUIRE_NOTHROW( flip->last_dependency()->set_explicit_beta_distribution(transdist) );
  REQUIRE_NOTHROW( flip->set_localexecutefn( &veccopy ) );
  REQUIRE_NOTHROW( reverse->add_kernel(flip) );

  REQUIRE_NOTHROW( reverse->analyze_dependencies() );
  REQUIRE_NOTHROW( reverse->execute() );

  // check input
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    INFO( "proc " << mycoord.as_string() );
    INFO( "transdist local: " << transstruct->get_processor_structure(mycoord)->as_string() );
    REQUIRE_NOTHROW( data = o1->get_data(mycoord) );
    domain_coordinate
      my_first = o1->first_index_r(mycoord), my_last = o1->last_index_r(mycoord),
      global_first = o1->get_numa_structure()->first_index_r(),
      numsize = o1->numa_size_r();
    for (int i=my_first[0]; i<=my_last[0]; i++) {
      for (int j=my_first[1]; j<=my_last[1]; j++) {
	index_int loc = INDEX2D( i,j,global_first,numsize );
	INFO( "index " << i << "," << j << " at linear loc " << loc );
	CHECK( data[loc]==Approx((double)mytid) );
      }
    }
  }

  auto tsks = flip->get_tasks(); REQUIRE( tsks.size()==ntids );
  for ( auto tsk : tsks ) {
    //for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = tsk->get_domain() );
    std::vector<int> flip_coords{
      layout->coord(0)-mycoord.coord(0)-1,layout->coord(1)-mycoord.coord(1)-1 };
    REQUIRE_NOTHROW( flipcoord = processor_coordinate( flip_coords ) );

    std::vector<message*> msgs;
    REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
    CHECK( msgs.size()==1 );
    {
      auto msg = msgs.at(0);
      CHECK( msg->get_receiver().equals(mycoord) );
      CHECK( msg->get_sender().equals(flipcoord) );
      INFO( "recv message " << msg->get_sender().as_string() << "->"
	    << mycoord.as_string() );
      { INFO( "should come from " << flipcoord.as_string() );
	CHECK( msg->get_sender().equals(flipcoord) );
      }
      {
	auto recv_structure = msg->get_global_struct();
	INFO( "receive structure " << msg->get_global_struct()->as_string() ); 
	CHECK( recv_structure->equals( r1->get_processor_structure(flipcoord) ) );
      }
    }
    REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
    CHECK( msgs.size()==1 );
    {
      auto msg = msgs.at(0);
      CHECK( msg->get_sender().equals(mycoord) );
      CHECK( msg->get_receiver().equals(flipcoord) );
      INFO( "send message  " << mycoord.as_string() << "->" 
	    << msg->get_receiver().as_string() );
      CHECK( msg->get_receiver().equals(flipcoord) );
      INFO( "recv message " << msg->get_sender().as_string() << "->"
	    << flipcoord.as_string() );
      {
	auto send_structure = msg->get_global_struct();
	INFO( "send structure " << msg->get_global_struct()->as_string() ); 
	CHECK( send_structure->equals( r1->get_processor_structure(mycoord) ) );
      }
    }
  }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    INFO( "proc " << mycoord.as_string() );

    REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
    domain_coordinate
      my_first = r1->first_index_r(mycoord), my_last = r1->last_index_r(mycoord),
      global_first = r1->get_numa_structure()->first_index_r(),
      numsize = r1->numa_size_r();
    for (int i=my_first[0]; i<=my_last[0]; i++) {
      for (int j=my_first[1]; j<=my_last[1]; j++) {
	index_int loc = INDEX2D( i,j,global_first,numsize );
	INFO( "index " << i << "," << j << " at linear loc " << loc );
	CHECK( data[loc]==Approx((double)(ntids-mytid-1) ) );
      }
    }
  }
}

TEST_CASE( "two-dimensional transpose","[multi][kernel][202]" ) {
  // this initial code comes from distribution[100]
  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  int dim = 2;
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  omp_decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );
  int
    ntids_i = layout->coord(0), ntids_j = layout->coord(1);
  if (ntids_i!=ntids_j) {
    printf("Square processor array for now\n"); return; }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    int
      mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
    CHECK( mytid==mytid_j+ntids_j*mytid_i );
    int fliptid = mytid_i+ntids_i*mytid_j;
    REQUIRE_NOTHROW( flipcoord = new processor_coordinate( std::vector<int>{mytid_j,mytid_i} ) );
    INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	  "; flipped " << fliptid << "=" << flipcoord.as_string() );
  }

  int nlocal = 10; indexstruct *idx; index_int g;
  std::vector<index_int> domain;
  g = ntids_i*(nlocal+1); domain.push_back(g);
  g = ntids_j*(nlocal+2); domain.push_back(g);

  omp_distribution *d;
  REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain) );

  std::shared_ptr<object> o1,r1;
  REQUIRE_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d) ) );
  REQUIRE_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d) ) );

  // make the structure of the beta distribution
  parallel_structure *transstruct;
  REQUIRE_NOTHROW( transstruct = new parallel_structure(mdecomp) );
  CHECK( transstruct->get_dimensionality()==dim );
  transstruct->set_name("transposed structure");
  for (int p=0; p<ntids; p++) {
  //for (int p=ntids-1; p>=0; p--) {
    fmt::print("\n");
    processor_coordinate pcoord;
    REQUIRE_NOTHROW( pcoord = mdecomp->coordinate_from_linear(p) );
    processor_coordinate *pflip;
    REQUIRE_NOTHROW( pflip = new processor_coordinate
		     ( std::vector<int>{pcoord.coord(1),pcoord.coord(0)} ) );
    // fmt::print("Trans struct in {} comes from {}: {}\n",
    // 	       pcoord.as_string(),pflip->as_string(),
    // 	       d->get_processor_structure(pflip)->as_string());
    REQUIRE_NOTHROW
      ( transstruct->set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
  }
  REQUIRE_NOTHROW( transstruct->set_type(distribution_type::BLOCKED) );
  printf("Getting transstruct\n");
  for (int ip=0; ip<ntids; ip++) { // this loop is pointless, right?
    processor_coordinate ic = mdecomp->coordinate_from_linear(ip);
    auto str = transstruct->get_processor_structure(ic);
    //    fmt::print("{} struct @{} : {}\n",ic->as_string(),(long)str,str->as_string());
    INFO( "transdist local: " << transstruct->get_processor_structure(ic)->as_string() );
  }
  printf("early return\n"); return;
  CHECK( transstruct->has_type_blocked() );
  omp_distribution *transdist;
  REQUIRE_NOTHROW( transdist = new omp_distribution(transstruct) );
  CHECK( transdist->get_dimensionality()==dim );
  CHECK( transdist->has_type_blocked() );

  kernel *flip,*make;
  double *data;

  REQUIRE_NOTHROW( make = new omp_origin_kernel(o1) );
  printf("write a better stepwise constant function\n");
  // REQUIRE_NOTHROW( make->set_localexecutefn
  // ( [mytid] (kernel_function_args) -> void {
  //   vecsetconstant(kernel_function_call,(double)mytid); } ) );
  REQUIRE_NOTHROW( make->analyze_dependencies() );
  REQUIRE_NOTHROW( make->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );

    REQUIRE_NOTHROW( data = o1->get_data(mycoord) );
    for (int i=0; i<o1->volume(mycoord); i++)
      CHECK( data[i]==Approx((double)mytid) );
  }
  
  REQUIRE_NOTHROW( flip = new omp_kernel(o1,r1) );
  REQUIRE_NOTHROW( flip->last_dependency()->set_explicit_beta_distribution(transdist) );
  REQUIRE_NOTHROW( flip->set_localexecutefn( &veccopy ) );
  REQUIRE_NOTHROW( flip->analyze_dependencies() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );

    auto tsks = flip->get_tasks(); REQUIRE( tsks.size()==1 );
    auto tsk = tsks.at(0);
    auto msgs = tsk->get_receive_messages();
    CHECK( msgs.size()==1 );
    {
      auto msg = msgs.at(0);
      CHECK( msg->get_receiver().equals(mycoord) );
      INFO( "message from " << msg->get_sender().as_string() );
      CHECK( msg->get_sender().equals(flipcoord) );
    }
  }

  REQUIRE_NOTHROW( flip->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,flipcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );

    int
      mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
    CHECK( mytid==mytid_j+ntids_j*mytid_i );
    int fliptid = mytid_i+ntids_i*mytid_j;

    REQUIRE_NOTHROW( data = r1->get_data(mycoord) );
    for (int i=0; i<o1->volume(mycoord); i++)
      CHECK( data[i]==Approx(fliptid) );
  }
}

TEST_CASE( "column rotating","[multi][kernel][220]" ) {
  int dim = 2;
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  omp_decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );
  int
    ntids_i = layout->coord(0), ntids_j = layout->coord(1);
  if (ntids_i!=ntids_j) {
    printf("Square processor array for Cannon\n"); return; }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord,rotcoord;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    int
      mytid_i = mycoord.coord(0), mytid_j = mycoord.coord(1);
    CHECK( mytid==mytid_j+ntids_j*mytid_i );
    int new_i = (mytid_i+mytid_j)%ntids_i;
    int rottid = new_i*ntids_j + mytid_j;
    REQUIRE_NOTHROW( rotcoord = new processor_coordinate( std::vector<int>{new_i,mytid_j} ) );
    INFO( "proc " << mytid << "=" << mycoord.as_string() <<
	  "; rotated " << rottid << "=" << rotcoord.as_string() );
  }

  // int nlocal = 10; indexstruct *idx; index_int g;
  // std::vector<index_int> domain;
  // g = ntids_i*(nlocal+1); domain.push_back(g);
  // g = ntids_j*(nlocal+2); domain.push_back(g);

  // omp_distribution *d;
  // REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain) );

  // std::shared_ptr<object> o1,r1;
  // REQUIRE_NOTHROW( o1 = std::shared_ptr<object>( new omp_object(d) ) );
  // REQUIRE_NOTHROW( r1 = std::shared_ptr<object>( new omp_object(d) ) );

  // // make the structure of the beta distribution
  // parallel_structure *transstruct;
  // REQUIRE_NOTHROW( transstruct = new parallel_structure(mdecomp) );
  // CHECK( transstruct->get_dimensionality()==dim );
  // transstruct->set_name("transposed structure");
  // for (int p=0; p<ntids; p++) {
  // //for (int p=ntids-1; p>=0; p--) {
  //   fmt::print("\n");
  //   processor_coordinate pcoord;
  //   REQUIRE_NOTHROW( pcoord = mdecomp->coordinate_from_linear(p) );
  //   processor_coordinate *pflip;
  //   REQUIRE_NOTHROW( pflip = new processor_coordinate
  // 		     ( std::vector<int>{pcoord.coord(1),pcoord.coord(0)} ) );
  //   // fmt::print("Trans struct in {} comes from {}: {}\n",
  //   // 	       pcoord.as_string(),pflip->as_string(),
  //   // 	       d->get_processor_structure(pflip)->as_string());
  //   REQUIRE_NOTHROW
  //     ( transstruct->set_processor_structure(pcoord,d->get_processor_structure(pflip)) );
  // }
  // REQUIRE_NOTHROW( transstruct->set_type(distribution_type::BLOCKED) );
}

