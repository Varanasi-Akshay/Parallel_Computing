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
 **** unit tests for over-decomposed mpi
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"

TEST_CASE( "test overdecomposition","[mpi][05]" ) {
  int over;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }

  INFO( "using over factor " << over );
  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  mpi_decomposition *decomp;
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
  CHECK( decomp->local_ndomains()==over );
  CHECK( decomp->domains_volume()==over*ntids );
}

TEST_CASE( "MPI distributions, local stuff","[mpi][distribution][13]" ) {
  
  int nlocal = 100, over; //= arch->get_over_factor(), 
  domain_coordinate s(1);
  mpi_decomposition *decomp; distribution *d1;
  SECTION( "local size" ) {
    SECTION( "regular" ) { over = 1;
      REQUIRE_NOTHROW( arch->set_over_factor(1) );
    }      
    SECTION( "over" ) { over = 2;
      REQUIRE_NOTHROW( arch->set_over_factor(over) );
    }      
    INFO( "over = " << over );
    REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
    CHECK( decomp->domains_volume()==over*ntids );
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,nlocal,-1) );
    REQUIRE_NOTHROW( s = d1->global_size() );
  }
  SECTION( "global size" ) {
    index_int s;
    SECTION( "regular" ) { over = 1;
      REQUIRE_NOTHROW( arch->set_over_factor(over) );
    }
    SECTION( "over" ) { over = 2;
      REQUIRE_NOTHROW( arch->set_over_factor(over) );
    }
    INFO( "over = " << over );
    s = nlocal*ntids*over;
    REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
    CHECK( decomp->domains_volume()==over*ntids );
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,s) );
  }
  INFO( "over = " << over );
  CHECK( d1->has_defined_type() );

  distribution *d2;
  REQUIRE_NOTHROW( d2 = new mpi_distribution(d1) );

  for (auto dom : decomp->get_domains()) {
    INFO( d1->as_string() << " on domain: " << dom.as_string() );
    CHECK( d1->volume(dom)==nlocal );
    auto ndom = dom.operate(mult_operator(nlocal)),
      ndom1 = dom.operate(mult_operator(nlocal))+nlocal-1;
    domain_coordinate f(ndom.data()), l(ndom1.data());
    INFO( "f=" << f.as_string() << ", l=" << l.as_string() );
    CHECK( d1->first_index_r(dom)==f );
    CHECK( d1->contains_element(dom,f) );
    {
      domain_coordinate f1 = f-1;
      CHECK( !d1->contains_element(dom,f-1) );
    }
    
    domain_coordinate d2s(1);
    REQUIRE_NOTHROW( d2s = d2->local_size_r(dom) );
    CHECK( d2->volume(dom)==nlocal );
    REQUIRE_NOTHROW( d2s = d2->first_index_r(dom) );
    CHECK( d2s==f );
    CHECK( d2->contains_element(dom,f) );
    {
      domain_coordinate f1 = f-1;
      CHECK( !d2->contains_element(dom,f1) );
    }
    
    CHECK( d1->last_index_r(dom)==l );
    CHECK( d1->contains_element(dom,l) );
    INFO( s.as_string() << " should be outside " << d1->first_index_r(dom).as_string()
	  << "-" << d1->last_index_r(dom).as_string() );
    CHECK_NOTHROW( !d1->contains_element(dom,s) );
    {
      domain_coordinate l1 = l+1;
      CHECK( !d1->contains_element(dom,l1) );
    }
  }

  distribution *d3 = new mpi_replicated_distribution(decomp);
  for (auto dom : decomp->get_domains()) {
    CHECK( d3->volume(dom)==1 );
    CHECK( d3->first_index_r(dom)==domain_coordinate_zero(1) );
    CHECK( d3->last_index_r(dom)==domain_coordinate_zero(1) );
  }
}

#if 0
TEST_CASE( "Operated distributions with modulo","[mpi][distribution][modulo][14]" ) {

  int nlocal=10, over; index_int s,gsize;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }
  INFO( "over: " << over );

  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  s = nlocal*ntids*over, gsize = nlocal*ntids*over;
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
  distribution *d1;
  REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,nlocal,-1) );

  distribution *d1shift = 
    new mpi_block_distribution(decomp,-1,gsize);
  ioperator shift_op(">>1");
  CHECK( shift_op.is_modulo_op() );
  d1shift->operate( shift_op );

  for (auto dom : decomp->get_domains()) {
    INFO( "dom=" << dom );

    // record information for the original distribution
    auto first = d1->first_index_r(dom),last = d1->last_index_r(dom);
    auto localsize = d1->volume(dom);

    // the unshifted distribution
    CHECK( d1->volume(dom)==localsize );
    CHECK( d1->contains_element(dom,first) );
    CHECK( d1->contains_element(dom,last) );
  
    // now check information for the shifted distribution, modulo
    int fshift=MOD(first+1,gsize),lshift=MOD(last+1,gsize);
    CHECK( d1shift->volume(dom)==localsize );
    CHECK( d1shift->contains_element(dom->coord(0),fshift) );
  }
}
#endif

TEST_CASE( "Overdecomposed objects","[object][over][20]" ) {
  mpi_architecture *earch = new mpi_architecture( *arch );
  REQUIRE_NOTHROW( earch->set_can_embed_in_beta() );
  int over = earch->get_over_factor();
  decomposition *edecomp; REQUIRE_NOTHROW( edecomp = new mpi_decomposition(earch) );
  index_int nlocal = 10;
  mpi_distribution *d;
  REQUIRE_NOTHROW( d = new mpi_block_distribution(edecomp,nlocal,-1) );
  mpi_object *o;
  REQUIRE_NOTHROW( o = new mpi_object(d) );
  CHECK( !o->has_data_status_allocated() );
  REQUIRE_NOTHROW( o->allocate() );
  CHECK( o->has_data_status_allocated() );

  CHECK( o->get_allocated_space()==Approx(nlocal*over) );

  double *dp; std::vector<processor_coordinate> domains;
  REQUIRE_NOTHROW( domains = d->get_domains() );
  REQUIRE( domains.size()==over );
  REQUIRE( d->domains_volume()==over*ntids );
  auto d0 = domains.at(0);
  REQUIRE_NOTHROW( dp = o->get_data(d0) );
  for ( auto dom : domains ) {
    INFO( "proc " << mytid << ", dom " << dom.as_string() );
    double *ptr; 
    REQUIRE_NOTHROW( ptr = o->get_data(dom) );
    // if (dom!=d0)
    //   CHECK( ((long)dp)!=((long)ptr) );
  }
}

TEST_CASE( "Analyze single interval, exact processor interval","[index][structure][21]" ) {
  if (decomp->domains_volume()<2) return;

  int localsize = 5, over;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }
  INFO( "over: " << over );

  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );

  mpi_distribution *dist = new mpi_distribution(decomp); 
  auto pstruct = dist->get_dimension_structure(0);
  CHECK_NOTHROW( pstruct->create_from_global_size(over*localsize*ntids) );
  std::vector<message*> mm;
  message *m; std::shared_ptr<multi_indexstruct> s,segment;

  // each processor claims to want the zero interval...
  segment = std::shared_ptr<multi_indexstruct>
    (new multi_indexstruct
     ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,localsize-1) ) ));
  for (auto dom : decomp->get_domains()) {
    CHECK_NOTHROW( mm = dist->messages_for_segment( dom,self_treatment::INCLUDE,segment,segment ) );
    CHECK( mm.size()==1 );
    m = mm.at(0);
    CHECK( m->get_sender().coord(0)==0 );
    CHECK( m->get_receiver().equals(dom) );
    s = m->get_global_struct();
    REQUIRE( s!=nullptr );
    CHECK( s->first_index_r()[0]==0 );
    CHECK( s->last_index_r()[0]==localsize-1 );
  }
}

TEST_CASE( "Analyze single interval, sub interval","[index][structure][21]" ) {
  if (decomp->domains_volume()<2) return;

  int localsize = 5, over;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }
  INFO( "over: " << over );

  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );

  mpi_distribution *dist = new mpi_distribution(decomp); 
  auto pstruct = dist->get_dimension_structure(0);
  CHECK_NOTHROW( pstruct->create_from_global_size(over*localsize*ntids) );
  std::vector<message*> mm;
  message *m; std::shared_ptr<multi_indexstruct> s,segment;

  segment = std::shared_ptr<multi_indexstruct>
    (new multi_indexstruct
     ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(1,localsize-2) ) ));
  for (auto dom : decomp->get_domains()) {
    CHECK_NOTHROW( mm = dist->messages_for_segment( dom,self_treatment::INCLUDE,segment,segment ) );
    CHECK( mm.size()==1 );
    m = mm.at(0);
    CHECK( m->get_sender().coord(0)==0 );
    CHECK( m->get_receiver().equals(dom) );
    s = m->get_global_struct();
    REQUIRE( s!=nullptr );
    CHECK( s->first_index_r()[0]==1 );
    CHECK( s->last_index_r()[0]==localsize-2 );
  }
}

TEST_CASE( "Analyze one dependency, right bump","[operate][dependence][22]") {

  int localsize = 100;

  int over;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }
  INFO( "over: " << over );

  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
  int ndoms = decomp->domains_volume();

  mpi_distribution *alpha = 
    new mpi_block_distribution(decomp,-1,over*localsize*ntids);
  std::shared_ptr<multi_indexstruct> alpha_block,segment;
  std::vector<message*> mm; message *m;

  auto shiftop = ioperator(">=1");
  CHECK( shiftop.is_right_shift_op() );
  CHECK( !shiftop.is_modulo_op() );

  for (auto dom : decomp->get_domains()) {
    auto
      my_first = alpha->first_index_r(dom),
      my_last = alpha->last_index_r(dom);
    alpha_block = alpha->get_processor_structure(dom); // should be from alpha/gamma
    CHECK( alpha_block->volume()==localsize );
    CHECK( alpha_block->first_index_r()==my_first );

    REQUIRE_NOTHROW( segment = alpha_block->operate
		     (shiftop,alpha->get_enclosing_structure()) );
    INFO( "messages for segment " << segment->as_string()
	  << " based on gamma block " << alpha_block->as_string() );
    CHECK( segment->first_index_r()==my_first+1 );
    CHECK_NOTHROW( mm = alpha->messages_for_segment
		   ( dom,self_treatment::INCLUDE,segment,segment ) );
    if (dom.coord(0)<ndoms-1) {
      CHECK( mm.size()==2 );
    } else {
      CHECK( mm.size()==1 );
    }
    m = mm.at(0); 
    CHECK( m->get_sender()==dom );
    CHECK( m->get_global_struct()->volume()==localsize-1 );
    CHECK( m->get_global_struct()->first_index_r()==my_first+1 );
    if (dom.coord(0)<ndoms-1) {
      m = mm.at(1);
      CHECK( m->get_sender()==dom+1 );
      CHECK( m->get_global_struct()->volume()==1 );
      CHECK( m->get_global_struct()->first_index_r()==my_last+1 );
    }
  }
}

TEST_CASE( "Analyze one dependency, left bump","[operate][dependence][23]") {

  int localsize = 100;

  int over;
  SECTION( "regular" ) { over = 1; }
  SECTION( "over" ) { over = 2; }
  INFO( "over: " << over );

  REQUIRE_NOTHROW( arch->set_over_factor(over) );
  REQUIRE_NOTHROW( decomp = new mpi_decomposition(arch) );
  int ndoms = decomp->domains_volume();

  mpi_distribution *alpha = 
    new mpi_block_distribution(decomp,-1,over*localsize*ntids);
  std::shared_ptr<multi_indexstruct> alpha_block,segment;
  std::vector<message*> mm; message *m;

  auto shiftop = ioperator("<=1");
  CHECK( shiftop.is_left_shift_op() );
  CHECK( !shiftop.is_modulo_op() );
  for (auto dom : decomp->get_domains()) {
    auto
      my_first = alpha->first_index_r(dom),
      my_last = alpha->last_index_r(dom);
    alpha_block = alpha->get_processor_structure(dom);
    CHECK( alpha_block->volume()==localsize );
    REQUIRE_NOTHROW( segment = alpha_block->operate(shiftop,alpha->get_enclosing_structure()) );
    CHECK_NOTHROW( mm = alpha->messages_for_segment
		   ( dom,self_treatment::INCLUDE,segment,segment ) );
    if (*dom>0) {
      CHECK( mm.size()==2 );
    } else {
      CHECK( mm.size()==1 );
    }
    if (*dom>0) {
      for (int im=0; im<2; im++) {
	m = mm.at(im);
	if ( m->get_sender()==dom-1 ) {
	  CHECK( m->get_global_struct()->volume()==1 );
	} else if ( m->get_sender()==dom ) {
	  CHECK( m->get_global_struct()->volume()==localsize-1 );
	} else {
	  CHECK( 1==0 );
	}
      }
    } else {
      m = mm.at(0);
      CHECK( m->get_sender()==dom );
      CHECK( m->get_global_struct()->volume()==localsize-1 );
    }
  }
}

#if 0
TEST_CASE( "Inserting new domain","[over][domain][30]" ) {
  int nlocal = 10,over = arch->get_over_factor(), nglobal = nlocal*ntids*over;
  mpi_distribution *block,*blockplus;
  REQUIRE_NOTHROW( block = new mpi_block_distribution(decomp,nlocal,-1) );
  CHECK( block->global_size()==nglobal );
  REQUIRE_NOTHROW( blockplus = new mpi_block_distribution( block ) );
  for ( int dom : decomp->get_domains() ) {
    INFO( "dom: " << dom );
    CHECK( block->first_index_r(dom)==blockplus->first_index_r(dom) );
  }
}
#endif
