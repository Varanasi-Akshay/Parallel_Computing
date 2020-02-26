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
 **** unit tests for OpenMP-based distributions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_ops.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"

TEST_CASE( "test presence of numa structure","[omp][numa][04]" ) {
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1;
  REQUIRE_NOTHROW( d1 = new omp_block_distribution(decomp,s) );
  indexstruct *numa;
  REQUIRE_NOTHROW( numa = d1->get_numa_structure(0) );
  CHECK( !numa->is_empty() );
  CHECK( numa->first_index()==0 );
  CHECK( numa->local_size()==s );
}

TEST_CASE( "OMP distributions, local stuff","[omp][distribution][13]" ) {
  
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1 = 
    new omp_block_distribution(decomp,s);
  CHECK( d1->has_defined_type() );
  INFO( "block distribution: " << d1->as_string() );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "coordinate " << mytid << "=" << mycoord.as_string() );
    auto
      f = domain_coordinate( std::vector<index_int>{nlocal*mytid} ),
      l = domain_coordinate( std::vector<index_int>{nlocal*(mytid+1)-1} );
    //int f=nlocal*mytid, l=nlocal*(mytid+1)-1;
    CHECK( d1->volume(mycoord)==nlocal );
    CHECK( d1->first_index_r(mycoord)==f );
    CHECK( d1->contains_element(mycoord,d1->first_index_r(mycoord)) );
    domain_coordinate f1 = f-1;
    CHECK( ( !d1->is_valid_index(f-11) || !d1->contains_element(mycoord,f1) ) );
  }
  
  distribution *d2;
  REQUIRE_NOTHROW( d2 = new omp_distribution(d1) );
  domain_coordinate d2c(0); index_int d2s;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    auto
      f = domain_coordinate( std::vector<index_int>{nlocal*mytid} ),
      l = domain_coordinate( std::vector<index_int>{nlocal*(mytid+1)-1} );
    //int f=nlocal*mytid, l=nlocal*(mytid+1)-1;
    REQUIRE_NOTHROW( d2s = d2->volume(mycoord) );
    CHECK( d2s==nlocal );
    REQUIRE_NOTHROW( d2c = d2->first_index_r(mycoord) );
    CHECK( d2c==f );
    CHECK( d2->contains_element(mycoord,f) );
    domain_coordinate f1 = f-1;
    CHECK( ( !d2->is_valid_index(f-1) || !d2->contains_element(mycoord,f1) ) );
  }
  
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    auto
      f = domain_coordinate( std::vector<index_int>{nlocal*mytid} ),
      l = domain_coordinate( std::vector<index_int>{nlocal*(mytid+1)-1} );
    //int f=nlocal*mytid, l=nlocal*(mytid+1)-1;
    CHECK( d1->last_index_r(mycoord)==l );
    CHECK( d1->contains_element(mycoord,l) );
    domain_coordinate cs = domain_coordinate(std::vector<index_int>{s});
    CHECK( !d1->contains_element(mycoord,cs) );
    domain_coordinate l1 = l+1;
    CHECK( ( !d1->is_valid_index(l+11) || !d1->contains_element(mycoord,l1) ) );
    domain_coordinate m1 = domain_coordinate(std::vector<index_int>{-1});
    CHECK( !d1->contains_element(mycoord,m1) );
  }
  
  distribution *d3 = new omp_replicated_distribution(decomp);
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( d3->volume(mycoord)==1 );
    CHECK( d3->first_index_r(mycoord)==domain_coordinate_zero(1) );
    CHECK( d3->last_index_r(mycoord)==domain_coordinate_zero(1) );
  }
}

TEST_CASE( "OMP distributions at nonzero offset, local stuff","[omp][distribution][15]" ) {
  
  int nlocal = 100, s = nlocal*ntids, base; const char *path;

  SECTION( "zero base" ) {
    base = 0; path = "zero base";
  }
  SECTION( "positive base" ) {
    base = nlocal/2; path = "positive base";
  }
  SECTION( "negative base" ) {
    base = -nlocal/2; path = "negative base";
  }
  INFO( "base: " << path );
  parallel_structure *pstr;
  REQUIRE_NOTHROW( pstr = new parallel_structure(decomp) );
  REQUIRE_NOTHROW( pstr->create_from_indexstruct
		   ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(base,base+s-1) ) ) );

  distribution *d1;
  REQUIRE_NOTHROW( d1 = new omp_distribution(pstr) );
  CHECK( d1->has_defined_type() );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    int f=base+nlocal*mytid, l=base+nlocal*(mytid+1)-1;
    CHECK( d1->volume(mycoord)==nlocal );
    CHECK( d1->first_index_r(mycoord).coord(0)==f );
    auto cf = domain_coordinate(std::vector<index_int>{f});
    CHECK( d1->contains_element(mycoord,cf) );
    auto f1 = domain_coordinate(std::vector<index_int>{f-1});
    CHECK( ( !d1->is_valid_index(f1) || !d1->contains_element(mycoord,f1) ) );
  }

  distribution *d2;
  REQUIRE_NOTHROW( d2 = new omp_distribution(d1) );
  index_int d2s,d2f;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord; std::shared_ptr<multi_indexstruct> struc;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( struc = d2->get_processor_structure(mycoord) );
    INFO( fmt::format("copied distributin on p={}={}: {}",
		      mytid,mycoord.as_string(),struc->as_string() ) );
    int f=base+nlocal*mytid, l=base+nlocal*(mytid+1)-1;
    INFO( fmt::format(".. computing first/last as {}-{}",f,l) );
    REQUIRE_NOTHROW( d2s = d2->volume(mycoord) );
    CHECK( d2s==nlocal );
    REQUIRE_NOTHROW( d2f = d2->first_index_r(mycoord).coord(0) );
    CHECK( d2f==f );
    auto cf = domain_coordinate(std::vector<index_int>{f});
    CHECK( struc->contains_element(cf) );
    CHECK( d2->contains_element(mycoord,cf) );
    auto f1 = domain_coordinate(std::vector<index_int>{f-1});
    CHECK( ( !d2->is_valid_index(f1) || !d2->contains_element(mycoord,f1) ) );
  }
  
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    int f=base+nlocal*mytid, l=base+nlocal*(mytid+1)-1;
    CHECK( d1->last_index_r(mycoord).coord(0)==l );
    auto cl = domain_coordinate(std::vector<index_int>{l});
    CHECK( d1->contains_element(mycoord,cl) );
    auto bs = domain_coordinate(std::vector<index_int>{base+s});
    CHECK( !d1->contains_element(mycoord,bs) );
    auto l1 = domain_coordinate(std::vector<index_int>{l+1});
    CHECK( ( !d1->is_valid_index(l1) || !d1->contains_element(mycoord,l1) ) );
    auto b1 = domain_coordinate(std::vector<index_int>{base-1});
    CHECK( !d1->contains_element(mycoord,b1) );
  }
}

TEST_CASE( "Operated distributions with modulo","[omp][distribution][modulo][24]" ) {

  int gsize = 10*ntids;
  distribution *d1 = 
    new omp_block_distribution(decomp,-1,gsize);
  // record information for the original distribution
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    domain_coordinate first = d1->first_index_r(mycoord),last = d1->last_index_r(mycoord);
      index_int localsize = d1->volume(mycoord);
    // the unshifted distribution
    CHECK( d1->volume(mycoord)==localsize );
    CHECK( d1->contains_element(mycoord,first) );
    CHECK( d1->contains_element(mycoord,last) );
  }

  // now check information for the shifted distribution, modulo
  distribution *d1shift;
  //multi_ioperator *shift_op = new multi_ioperator( new ioperator(">>1") );
  auto shift_op = ioperator(">>1");
  CHECK( shift_op.is_modulo_op() );
  d1shift = d1->operate( shift_op );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      first = d1->first_index_r(mycoord).coord(0),
      last = d1->last_index_r(mycoord).coord(0);
    index_int localsize = d1->volume(mycoord);
    int fshift=MOD(first+1,gsize),lshift=MOD(last+1,gsize);
    CHECK( d1shift->volume(mycoord)==localsize );
    auto cfs = domain_coordinate(std::vector<index_int>{fshift});
    CHECK( d1shift->contains_element(mycoord,cfs) );
  }
}

#if 0
TEST_CASE( "Masked distribution creation","[distribution][mask][70]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5;
  processor_mask *mask;

  fmt::MemoryWriter path;
  SECTION( "create mask by adding" ) { path.write("adding odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
    for (int p=1; p<ntids; p+=2 ) {
      processor_coordinate *c = new processor_coordinate(1); c->set(0,p);
      REQUIRE_NOTHROW( mask->add(c) );
    }
  }
  SECTION( "create mask by subtracting" ) { path.write("subtracting odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp,ntids) );
    for (int p=0; p<ntids; p+=2) {
      processor_coordinate *c = new processor_coordinate(1); c->set(0,p);
      REQUIRE_NOTHROW( mask->remove(c) );
    }
  }
  INFO( "masked created by: " << path.str() );

  omp_distribution
    *block = new omp_block_distribution(decomp,localsize,-1),
    *masked_block = new omp_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto
    whole_vector = std::shared_ptr<object>( new omp_object(block) ),
    masked_vector = std::shared_ptr<object>( new omp_object(masked_block) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK( whole_vector->get_allocated_space()==localsize*ntids );
  CHECK( masked_vector->get_allocated_space()==localsize*(ntids/2) );
  double *data;
  CHECK_NOTHROW( data = whole_vector->get_data(new processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    CHECK( block->lives_on(mytid) );
    if (mytid%2==1) {
      CHECK( masked_block->lives_on(mytid) );
      REQUIRE_NOTHROW( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    } else {
      CHECK( !masked_block->lives_on(mytid) );
      REQUIRE_THROWS( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  }
}

TEST_CASE( "Masked distribution on output","[distribution][mask][71]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5; int alive=1;
  processor_mask *mask;

  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  processor_coordinate *c = new processor_coordinate(1); c->set(0,alive);
  REQUIRE_NOTHROW( mask->add(c) );

  omp_distribution
    *block = new omp_block_distribution(decomp,localsize,-1),
    *masked_block = new omp_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto
    whole_vector = std::shared_ptr<object>( new omp_object(block) ),
    masked_vector = std::shared_ptr<object>( new omp_object(masked_block) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  CHECK( whole_vector->global_allocation()==ntids*localsize );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK( masked_vector->global_allocation()==localsize );

  double *indata,*outdata;
  REQUIRE_NOTHROW( indata = whole_vector->get_data(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<whole_vector->global_volume(); i++) indata[i] = 1.;
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    if (masked_vector->lives_on(mytid)) {
      REQUIRE_NOTHROW( outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
      CHECK( outdata!=nullptr );
      index_int f = masked_vector->location_of_first_index(*masked_vector.get(),mycoord),
	n = masked_vector->volume(mycoord);
      CHECK( f==0 );
      CHECK( whole_vector->location_of_first_index(*whole_vector.get(),mycoord)==localsize );
      for (index_int i=f; i<f+n; i++) outdata[i] = 2.;
    } else {
      REQUIRE_THROWS( outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  } 

  omp_kernel *copy;
  REQUIRE_NOTHROW
    ( copy = new omp_kernel(whole_vector,masked_vector,fmt::format("copy-to-masked")) );
  copy->last_dependency()->set_explicit_beta_distribution(masked_vector);
  copy->set_localexecutefn( &veccopy );
  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(whole_vector) ) );
  REQUIRE_NOTHROW( queue->add_kernel( copy ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  object *halo;
  REQUIRE_NOTHROW( halo = copy->get_dependency(0)->get_halo_object() );
  for (int mytid=0; mytid<ntids; mytid++)
    if (mytid==alive) {
      CHECK( halo->lives_on(mytid) );
      CHECK( halo->location_of_first_index(*halo.get(),mycoord)==0 );
    } else
      CHECK( !halo->lives_on(mytid) );
  std::vector<task*> *tsks; REQUIRE_NOTHROW( tsks = copy->get_tasks() );

  REQUIRE_NOTHROW( queue->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid=" << mytid );
    if (mytid==alive) {
      outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid));
      index_int f = masked_vector->location_of_first_index(*masked_vector.get(),mycoord),
	n = masked_vector->volume(mycoord);
      for (index_int i=f; i<f+n; i++) {
	INFO( ".. checking on location i=" << i << ", with base located at " << f );
	CHECK( outdata[i] == 1. );
      }
    }
  }
}

TEST_CASE( "Masked distribution on input","[distribution][mask][72]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5;
  processor_mask *mask;

  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  processor_coordinate *c = new processor_coordinate(1); c->set(0,0);
  REQUIRE_NOTHROW( mask->add(c) );

  omp_distribution
    *block = new omp_block_distribution(decomp,localsize,-1),
    *masked_block = new omp_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto
    whole_vector = std::shared_ptr<object>( new omp_object(block) ),
    masked_vector = std::shared_ptr<object>( new omp_object(masked_block) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK_NOTHROW( whole_vector->get_data(new processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    double *data;
    CHECK( block->lives_on(mytid) );
    if (mytid==0) {
      CHECK( masked_block->lives_on(mytid) );
      REQUIRE_NOTHROW( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    } else {
      CHECK( !masked_block->lives_on(mytid) );
      REQUIRE_THROWS( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  }
  
  double *indata,*outdata;
  // set the whole output to 1
  REQUIRE_NOTHROW( outdata = whole_vector->get_data(new processor_coordinate_zero(1)) );
  for (index_int i=0; i<whole_vector->global_volume(); i++) outdata[i] = 1.;
  // set input to 2, only on the mask
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    if (masked_vector->lives_on(mytid)) {
      REQUIRE_NOTHROW( indata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
      CHECK( indata!=nullptr );
      index_int f = masked_vector->first_index_r(mycoord).coord(0)-masked_vector->mask_shift(mytid),
	n = masked_vector->volume(mycoord);
      for (index_int i=f; i<f+n; i++) indata[i] = 2.;
    } else {
      REQUIRE_THROWS( indata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  }
  omp_kernel *copy;
  REQUIRE_NOTHROW( copy = new omp_copy_kernel(masked_vector,whole_vector) );

  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(masked_vector) ) );
  REQUIRE_NOTHROW( queue->add_kernel( copy ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  dependency *dep;
  REQUIRE_NOTHROW( dep = copy->last_dependency() );
  CHECK( !dep->get_halo_object()->has_mask() ); // mask is on input
  REQUIRE_NOTHROW( queue->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( outdata = whole_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    index_int f = whole_vector->first_index_r(mycoord).coord(0),
      n = whole_vector->volume(mycoord);
    if (mytid==0) { // output has copied value
      for (index_int i=f; i<f+n; i++) {
	INFO( ".. checking copied output on i=" << i );
	CHECK( outdata[i] == 2. );
      }
    } else { // output has original value
      for (index_int i=f; i<f+n; i++) {
	INFO( ".. checking original data on i=" << i );
	CHECK( outdata[i] == 1. );
      }
    }
  }
}

TEST_CASE( "omp-specific mask allocation issues","[distribution][mask][79]" ) {
  if (ntids<3) { printf("mask 79 needs 3 procs\n"); return; }
  index_int localsize = 10;
  omp_distribution *block = new omp_block_distribution(decomp,localsize,-1);
  processor_mask *mask = new processor_mask(decomp);
  {
    processor_coordinate *c = new processor_coordinate(1); c->set(0,0);
    REQUIRE_NOTHROW( mask->add(0) );
  }
  {
    processor_coordinate *c = new processor_coordinate(1); c->set(0,2);
    REQUIRE_NOTHROW( mask->add(2) );
  }
  REQUIRE_NOTHROW( block->add_mask(mask) );
  object *vector;
  REQUIRE_NOTHROW( vector = std::shared_ptr<object>( new omp_object(block) ) );
  CHECK( vector->global_allocation()==2*localsize );
  CHECK( !vector->lives_on(1) );
  double *dt;
  REQUIRE_THROWS( dt = vector->get_data(1) );
  index_int ls,fs;
  REQUIRE_THROWS( fs = vector->first_index_r(1) );
  REQUIRE_THROWS( ls = vector->local_size(1) );
  CHECK( vector->first_index_r(2)==vector->first_index_r(0)+2*localsize );
  REQUIRE_THROWS( vector->mask_shift(1) );
  CHECK( vector->mask_shift(2)==localsize );
}
#endif

TEST_CASE( "dividing","[distribution][ortho][15]" ) {
  int nlocal = 8, k;
  distribution *level_dist, *new_dist;
  SECTION( "k=1" ) {
    k = 1;
  }
  SECTION( "k=2" ) {
    k = 2;
  }
  INFO( "k=" << k );
  REQUIRE_NOTHROW( level_dist = new omp_block_distribution(decomp,k,nlocal,-1) );
  index_int nglobal = level_dist->global_volume();

  auto coarsen = ioperator(":2");

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  CHECK( level_dist->local_allocation()==k*nglobal );
  CHECK( new_dist->get_orthogonal_dimension()==k );
  CHECK( new_dist->local_allocation()==k*nglobal/2 );
}

TEST_CASE( "extending distributions","[distribution][extend][30]" ) {
  if (ntids<2) {
    printf("test 30 needs multiple processes\n"); return; }

  int nlocal=100,nglobal=nlocal*ntids;
  distribution *d1,*d2;
  REQUIRE_NOTHROW( d1 = new omp_block_distribution(decomp,nlocal,-1) );

  int shift;
  SECTION( "keep it contiguous" ) {
    shift = 1;
  }
  SECTION( "make it composite" ) {
    shift = 2;
  }
  INFO( "using shift: " << shift );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "p=" << mytid );
    index_int
      the_first = d1->first_index_r(mycoord).coord(0),
      the_last = d1->last_index_r(mycoord).coord(0);
    if (mytid!=0) {
      auto left = std::shared_ptr< multi_indexstruct >
	( new multi_indexstruct( std::shared_ptr<indexstruct>
				 ( new contiguous_indexstruct
				   ( the_first-shift ) ) ) );
      REQUIRE_NOTHROW( d1->extend(mycoord,left) );
      INFO( d1->get_processor_structure(mycoord)->as_string() );
      CHECK( d1->first_index_r(mycoord).coord(0)==(the_first-shift) );
    }
    if (mytid!=ntids-1) {
      auto right = std::shared_ptr< multi_indexstruct >
	( new multi_indexstruct( std::shared_ptr<indexstruct>
				 ( new contiguous_indexstruct
				   ( the_last+shift ) ) ) );
      REQUIRE_NOTHROW( d1->extend(mycoord,right) );
      INFO( d1->get_processor_structure(mycoord)->as_string() );
      CHECK( d1->last_index_r(mycoord).coord(0)==(the_last+shift) );
    }
  }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = mytid*nlocal, my_last = (mytid+1)*nlocal-1;
    if (mytid==0 || mytid==ntids-1)
      CHECK( d1->volume(mycoord)==(nlocal+1) );
    else
      CHECK( d1->volume(mycoord)==(nlocal+2) );
    if (mytid==0) 
      CHECK( d1->first_index_r(mycoord).coord(0)==my_first );
    else
      CHECK( d1->first_index_r(mycoord).coord(0)==my_first-shift );
    if (mytid==ntids-1) 
      CHECK( d1->last_index_r(mycoord).coord(0)==my_last );
    else
      CHECK( d1->last_index_r(mycoord).coord(0)==my_last+shift );
  }
}

TEST_CASE( "Cyclic distributions","[distribution][cyclic][40]" ) {
  distribution *d;
  //  REQUIRE_THROWS( d = new omp_cyclic_distribution(decomp,-1,-1) ); VLE why is this disabled?
  //  REQUIRE_THROWS( d = new omp_cyclic_distribution(decomp,1,ntids+1) );
  SECTION( "set from local" ) {
    REQUIRE_NOTHROW( d = new omp_cyclic_distribution(decomp,2,-1) );
  }
  SECTION( "set from global" ) {
    REQUIRE_NOTHROW( d = new omp_cyclic_distribution(decomp,-1,2*ntids) );
  }
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( d->volume(mycoord)==2 );
  }
}

index_int pfunc1(int p,index_int i) {
  return 3*p+i;
}

index_int pfunc2(int p,index_int i) {
  return 3*(p/2)+i;
}

TEST_CASE( "Function-specified distribution","[distribution][50]" ) {
  omp_distribution *d1,*d2;
  int nlocal = 3;

  CHECK_NOTHROW( d1 = new omp_distribution(decomp,&pfunc1,nlocal ) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid=" << mytid );
    CHECK( d1->volume(mycoord)==nlocal );
    for (int i=0; i<nlocal; i++) {
      index_int iglobal = 3*mytid+i;
      auto cig = domain_coordinate(std::vector<index_int>{iglobal});
      CHECK( d1->contains_element(mycoord,cig) );
      CHECK( d1->find_index(iglobal)==mytid );
    }
  }

  CHECK_NOTHROW( d2 = new omp_distribution(decomp,&pfunc2,nlocal ) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( d2->volume(mycoord)==nlocal );
    for (int i=0; i<nlocal; i++) {
      index_int iglobal = 3*(mytid/2)+i; // proc 0,1 have same data, likewise 2,3, 4,5
      auto cig = domain_coordinate(std::vector<index_int>{iglobal});
      CHECK( d2->contains_element(mycoord,cig) );
      CHECK( d2->find_index(iglobal,mytid)==mytid );
      CHECK( d2->find_index(iglobal)==2*(mytid/2) ); // the first proc with my data is 2*(p/2)
    }
  }
}

TEST_CASE( "multidimensional distributions","[multi][distribution][100]" ) {
  int ntids_i,ntids_j;
  if (ntids!=4) { printf("100 grid example needs exactly 4 procs\n"); return; }
  for (int n=sqrt(ntids); n>=1; n--)
    if (ntids%n==0) {
      ntids_j = n; ntids_i = ntids/n; break; }
  if (ntids_i==1) { printf("Could not split processor grid\n"); return; }
  CHECK( ntids_i>1 );
  CHECK( ntids_j>1 );
  CHECK( ntids==ntids_i*ntids_j );

  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(2) );
  omp_decomposition *mdecomp;
  SECTION( "default splitting of processor grid" ) {
    REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );
  }
  // SECTION( "explicit splitting of processor grid" ) {
  //   std::vector<int> grid; grid.push_back(2); grid.push_back(2);
  //   REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,grid) );
  // }

  processor_coordinate mycoord; 
  for (int mytid=0; mytid<ntids; mytid++) {
    int mytid_j = mytid%ntids_j, mytid_i = mytid/ntids_j;
    REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
    INFO( "p: " << mytid << ", pcoord: " << mycoord.coord(0) << "," << mycoord.coord(1) );
    CHECK( mytid==mycoord.linearize(mdecomp) );
    CHECK( mycoord.get_dimensionality()==2 );
    INFO( "mytid=" << mytid << ", s/b " << mytid_i << "," << mytid_j );
    CHECK( mycoord.coord(0)==mytid_i );
    CHECK( mycoord.coord(1)==mytid_j );
  }

  int nlocal = 10; index_int g;
  std::vector<index_int> domain_layout;
  g = ntids_i*(nlocal+1);
  domain_layout.push_back(g);

  g = ntids_j*(nlocal+2);
  domain_layout.push_back(g);

  omp_distribution *d;
  REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain_layout) );
  CHECK( d->get_dimensionality()==2 );
  std::shared_ptr<multi_indexstruct> local_domain;
  REQUIRE_NOTHROW( local_domain = d->get_processor_structure(mycoord) );
  CHECK( local_domain->get_dimensionality()==2 );
  CHECK( local_domain->get_component(0)->local_size()==nlocal+1 );
  CHECK( local_domain->get_component(1)->local_size()==nlocal+2 );

}

TEST_CASE( "multidimensional distributions error test","[multi][distribution][101]" ) {
  int ntids_i,ntids_j;
  if (ntids!=4) { printf("101 grid example needs exactly 4 procs\n"); return; }

  omp_decomposition *mdecomp;
  SECTION( "explicit splitting of processor grid" ) {
    processor_coordinate *endpoint = new processor_coordinate(2);
    endpoint->set(0,2); endpoint->set(1,2);
    REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,endpoint) );
  }
  // we should not test this because we can overdecompose, true?
  // SECTION( "incorrect splitting of processor grid" ) {
  //   processor_coordinate *endpoint = new processor_coordinate(2);
  //   endpoint->set(0,2); endpoint->set(1,1);
  //   REQUIRE_THROWS( mdecomp = new omp_decomposition(arch,endpoint) );
  // }
  SECTION( "pencil splitting of processor grid" ) {
    processor_coordinate *endpoint = new processor_coordinate(2);
    endpoint->set(0,1); endpoint->set(1,4);
    REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,endpoint) );
  }
}

TEST_CASE( "multidimensional data layout","[multi][distribution][data][102]" ) {
  if (ntids!=4) { printf("102 grid example needs exactly 4 procs\n"); return; }

  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(2) );
  omp_decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new omp_decomposition(arch,layout) );

  std::vector<index_int> domain_layout;
  domain_layout.push_back(2);
  domain_layout.push_back(2);

  omp_distribution *d;
  REQUIRE_NOTHROW( d = new omp_block_distribution(mdecomp,domain_layout) );
  CHECK( d->get_dimensionality()==2 );
  CHECK( d->global_volume()==4 );

  std::shared_ptr<object> o;
  REQUIRE_NOTHROW( o = std::shared_ptr<object>( new omp_object(d) ) );
  REQUIRE_NOTHROW( o->allocate() );
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      processor_coordinate p;
      REQUIRE_NOTHROW( p = processor_coordinate( std::vector<int>{i,j} ) );
      double *data;
      REQUIRE_NOTHROW( data = o->get_data(p) );
      index_int loc;
      REQUIRE_NOTHROW( loc = o->location_of_first_index(*o.get(),p) );
      fmt::print("Proc {} has index at {}\n",p.as_string(),loc);
      REQUIRE_NOTHROW( data[loc] = loc );
    }
  }
  auto data = o->get_raw_data();
  for (int i=0; i<4; i++)
    CHECK( data[i]==Approx(i) );
}

#if 0
TEST_CASE( "More masked distribution on output","[distribution][mask][700]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5; int alive;
  processor_mask *mask;

  const char *path;
  // SECTION( "create mask by adding 0" ) { path = "adding";
  //   alive = 0;
  //   REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  //   REQUIRE_NOTHROW( mask->add(alive) );
  // }
  // SECTION( "create mask by subtracting" ) { path = "subtracting";
  //   alive = 0;
  //   REQUIRE_NOTHROW( mask = new processor_mask(decomp, ntids) );
  //   for (int p=0; p<ntids; p++)
  //     if (p!=alive)
  // 	REQUIRE_NOTHROW( mask->remove(p) );
  // }
  SECTION( "create mask by adding 1" ) { path = "adding";
    alive = 1;
    REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
    processor_coordinate *c = new processor_coordinate(1); c->set(0,alive);
    REQUIRE_NOTHROW( mask->add(c) );
  }
  INFO( "mask on " << alive << " created by: " << path );

  omp_distribution *block, *masked_block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,localsize,-1) );
  REQUIRE_NOTHROW( masked_block = new omp_distribution( *block ) );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );
  CHECK( masked_block->lives_on(alive) );

  std::shared_ptr<object> whole_vector, masked_vector;
  REQUIRE_NOTHROW( whole_vector  = std::shared_ptr<object>( new omp_object(block) ) );
  REQUIRE_NOTHROW( masked_vector = std::shared_ptr<object>( new omp_object(masked_block) ) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( " mytid = " << mytid );
    double *data;
    CHECK_NOTHROW( data = whole_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    CHECK( block->lives_on(mytid) );
    if (mytid==alive) {
      CHECK( masked_block->lives_on(mytid) );
      REQUIRE_NOTHROW( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    } else {
      CHECK( !masked_block->lives_on(mytid) );
      REQUIRE_THROWS( data = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  }

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( " mytid = " << mytid );
    double *indata,*outdata;
    REQUIRE_NOTHROW( indata = whole_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    CHECK( whole_vector->volume(mycoord)==localsize );
    for (index_int i=0; i<localsize; i++) indata[i] = 1.;
    if (masked_vector->lives_on(mytid)) {
      REQUIRE_NOTHROW( outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
      CHECK( outdata!=nullptr );
      for (index_int i=0; i<localsize; i++) outdata[i] = 2.;
    } else {
      REQUIRE_THROWS( outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
    }
  }

  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(whole_vector) ) );
  omp_kernel *copy = new omp_kernel(whole_vector,masked_vector);
  copy->last_dependency()->set_type_local();
  copy->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( queue->add_kernel(copy) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( " mytid = " << mytid );
    double *data;
    if (masked_vector->lives_on(mytid)) { double *outdata;
      REQUIRE_NOTHROW( outdata = masked_vector->get_data(decomp->coordinate_from_linear(mytid)) );
      index_int first = masked_vector->first_index_r(mycoord).coord(0);
      for (index_int i=0; i<localsize; i++) {
	INFO( "global i=" << i+first );
	CHECK( outdata[i+first] == 1. );
      }
    } // output vector otherwise not defined
  }
}

TEST_CASE( "Two masks","[distribution][mask][71]" ) {

  if( ntids<4 ) { printf("test 71 needs 4 procs\n"); return; }

  INFO( "mytid=" << mytid );
  index_int localsize = 5;
  processor_mask *mask1,*mask2;

  mask1 = new processor_mask();
  mask2 = new processor_mask();
  for (int tid=0; tid<ntids; tid+=2)
    REQUIRE_NOTHROW( mask1->add(tid) );
  for (int tid=0; tid<ntids; tid+=4)
    REQUIRE_NOTHROW( mask2->add(tid) );

  omp_distribution
    *block = new omp_block_distribution(decomp,localsize,-1),
    *masked_block1 = new omp_distribution( *block ),
    *masked_block2 = new omp_distribution( *block );
  REQUIRE_NOTHROW( masked_block1->add_mask(mask1) );
  REQUIRE_NOTHROW( masked_block2->add_mask(mask2) );
  auto
    whole_vector = std::shared_ptr<object>( new omp_object(block) ),
    masked_vector1 = std::shared_ptr<object>( new omp_object(masked_block1) ),
    masked_vector2 = std::shared_ptr<object>( new omp_object(masked_block2) );
  omp_kernel
    *copy1 = new omp_kernel(whole_vector,masked_vector1),
    *copy2 = new omp_kernel(masked_vector1,masked_vector2);

  {
    double *data;
    REQUIRE_NOTHROW( data = whole_vector->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 1;
  }
  if (masked_vector1->lives_on(mytid)) {
    double *data;
    REQUIRE_NOTHROW( data = masked_vector1->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 2;
  }
  if (masked_vector2->lives_on(mytid)) {
    double *data;
    REQUIRE_NOTHROW( data = masked_vector2->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 4;
  }

  copy1->last_dependency()->set_type_local();
  copy1->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy1->analyze_dependencies() );
  REQUIRE_NOTHROW( copy1->execute() );

  copy2->last_dependency()->set_type_local();
  copy2->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy2->analyze_dependencies() );
  REQUIRE_NOTHROW( copy2->execute() );

  {
    double *data;
    if (mytid%4==0) {
      REQUIRE_NOTHROW( data = masked_vector2->get_data(new processor_coordinate_zero(1)) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data[i] == 1. );
    } else if (mytid%2==0) {
      REQUIRE_NOTHROW( data = masked_vector1->get_data(new processor_coordinate_zero(1)) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data[i] == 1. );
    }
  }
}
#endif

