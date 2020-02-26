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
 **** unit tests for mpi-based distributions
 ****     this file does not test kernels and such
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

// for the [61/2/3] tests
#include "balance_functions.h"

TEST_CASE( "decompositions","[mpi][decomposition][01]" ) {
  INFO( "mytid=" << mytid );
  int over;
  REQUIRE_NOTHROW( over = arch->get_over_factor() );
  std::vector<processor_coordinate> domains;
  REQUIRE_NOTHROW( domains = decomp->get_domains() );
  int count = 0;
  for ( auto dom : domains ) { int lindom = dom.coord(0);
    INFO( "domain=" << lindom );
    CHECK( lindom==over*mytid+count );
    count++;
  }
}

TEST_CASE( "coordinate conversion","[mpi][decomposition][02]" ) {
  decomposition *oned;
  REQUIRE_NOTHROW( oned = new mpi_decomposition
		   (arch,new processor_coordinate( std::vector<int>{4} ) ) );
  CHECK( oned->get_dimensionality()==1 );
  CHECK( oned->domains_volume()==4 );
  processor_coordinate onep;
  REQUIRE_NOTHROW( onep = oned->coordinate_from_linear(1) );
  CHECK( onep==processor_coordinate( std::vector<int>{1} ) );

  decomposition *twod;
  REQUIRE_NOTHROW( twod = new mpi_decomposition
		   (arch,new processor_coordinate( std::vector<int>{2,4} ) ) );
  CHECK( twod->get_dimensionality()==2 );
  CHECK( twod->domains_volume()==8 );
  processor_coordinate twop;
  REQUIRE_NOTHROW( twop = twod->coordinate_from_linear(6) );
  INFO( "6 translates to " << twop.as_string() );
  CHECK( twop==processor_coordinate( std::vector<int>{1,2} ) );

  decomposition *threed;
  REQUIRE_NOTHROW( threed = new mpi_decomposition
		   (arch,new processor_coordinate( std::vector<int>{6,2,4} ) ) );
  CHECK( threed->get_dimensionality()==3 );
  CHECK( threed->domains_volume()==6*2*4 ); 
  processor_coordinate threep; // {2,0,1} -> 2*(2*4) 0*2 + 1 = 17
  REQUIRE_NOTHROW( threep = threed->coordinate_from_linear(17) );
  INFO( "17 translates to " << threep.as_string() );
  CHECK( threep== processor_coordinate( std::vector<int>{2,0,1} ) );

}

TEST_CASE( "coordinate operations","[mpi][decomposition][03]" ) {
  index_int nlocal = 100, s = nlocal*ntids;
  distribution *d1;
  CHECK( decomp->get_dimensionality()==1 );
  CHECK( decomp->domains_volume()==ntids );
  fmt::print("decomposition: {}\n",decomp->as_string());
  const char *path;
  SECTION( "base distribution from structure" ) { path = "from structure";
    parallel_structure *pidx;
    fmt::print("!!!! Structure allocating:\n");
    REQUIRE_NOTHROW( pidx = new parallel_structure(decomp) );
    fmt::print("!!!! Structure creation:\n");
    REQUIRE_NOTHROW( pidx->create_from_global_size(s) );
    fmt::print("!!!! Distribution creation from structure:\n");
    REQUIRE_NOTHROW( d1 = new distribution(pidx) );
    fmt::print("!!!! all done\n");
  }
  // SECTION( "mpi distribution from structure" ) { path = "from structure";
  //   parallel_structure *pidx;
  //   fmt::print("!!!! Structure allocating:\n");
  //   REQUIRE_NOTHROW( pidx = new parallel_structure(decomp) );
  //   fmt::print("!!!! Structure creation:\n");
  //   REQUIRE_NOTHROW( pidx->create_from_global_size(s) );
  //   fmt::print("!!!! Distribution creation from structure:\n");
  //   REQUIRE_NOTHROW( d1 = new mpi_distribution(pidx) );
  //   fmt::print("!!!! all done\n");
  // }
  SECTION( "by create call" ) { path = "create call";
    REQUIRE_NOTHROW( d1 = new mpi_distribution(decomp) );
    REQUIRE_NOTHROW( d1->create_from_global_size(s) );
  }
  SECTION( "in one step" ) { path = "block call";
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,-1,s) );
  }
  INFO( "distribution created: " << path );
  auto dom = decomp->get_domains()[0];
  domain_coordinate f1(1),f2(1);
  REQUIRE_NOTHROW( f1 = d1->first_index_r(dom) );
  //  domain_coordinate f1 = d1->first_index_r(dom);
  CHECK( f1==dom.operate( mult_operator(nlocal) ) );
}

TEST_CASE( "test presence of numa structure","[mpi][numa][04]" ) {
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1 = 
    new mpi_block_distribution(decomp,s);
  indexstruct *numa;
  REQUIRE_NOTHROW( numa = d1->get_numa_structure(0) );
  CHECK( !numa->is_empty() );
  CHECK( numa->first_index()==0+nlocal*mytid );
  CHECK( numa->local_size()==nlocal );
}

TEST_CASE( "processor sets","[processor][05]" ) {
  processor_set set;
  // add a vector
  REQUIRE_NOTHROW
    ( set.add
      ( std::shared_ptr<processor_coordinate>
	( new processor_coordinate( std::vector<int>{1,2,3} ) ) ) );
  // check that it's there
  CHECK( set.size()==1 );
  CHECK( set.contains
	 ( std::shared_ptr<processor_coordinate>
	   ( new processor_coordinate( std::vector<int>{1,2,3} ) ) ) );
  // add another vector and check
  REQUIRE_NOTHROW
    ( set.add
      ( std::shared_ptr<processor_coordinate>
	( new processor_coordinate( std::vector<int>{3,2,1} ) ) ) );
  CHECK( set.size()==2 );
  CHECK( set.contains
	 ( std::shared_ptr<processor_coordinate>
	   ( new processor_coordinate( std::vector<int>{3,2,1} ) ) ) );
  // we can not add a vector of a different dimension
  CHECK_THROWS( set.add
	 ( std::shared_ptr<processor_coordinate>
	   ( new processor_coordinate( std::vector<int>{3,2} ) ) ) );
  // we can not test a vector of a different dimension
  CHECK_THROWS( set.contains
	 ( std::shared_ptr<processor_coordinate>
	   ( new processor_coordinate( std::vector<int>{3,2} ) ) ) );
  // everything still copacetic?
  CHECK( set.contains
	 ( std::shared_ptr<processor_coordinate>
	   ( new processor_coordinate( std::vector<int>{1,2,3} ) ) ) );
  CHECK( set.size()==2 );

  // see if we can iterate
  int count=0,sum=0;
  for ( auto p : set ) {
    switch (count) {
    case 0 :
      CHECK( p->equals( processor_coordinate( std::vector<int>{1,2,3} ) ) );
      sum += count;
      break;
    case 1 :
      CHECK( p->equals( processor_coordinate( std::vector<int>{3,2,1} ) ) );
      sum += count;
      break;
    }
    count++;
  }
  CHECK( count==2 );
  CHECK( sum==1 );
}

TEST_CASE( "distribution types","[type][06]" ) {
  parallel_indexstruct *pidx;
  REQUIRE_NOTHROW( pidx = new parallel_indexstruct(ntids) );
  SECTION( "global" ) {
    REQUIRE_NOTHROW( pidx->create_from_global_size(10*ntids) );
    REQUIRE( pidx->can_detect_type(distribution_type::CONTIGUOUS) );
  }
  SECTION( "indexstruct" ) {
    REQUIRE_NOTHROW( pidx->create_from_indexstruct
	     ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(1,10*ntids) ) ) );
    REQUIRE( pidx->can_detect_type(distribution_type::CONTIGUOUS) );
  }
  SECTION( "strided" ) {
    REQUIRE_NOTHROW( pidx->create_from_indexstruct
	     ( std::shared_ptr<indexstruct>( new strided_indexstruct(1,10*ntids,2) ) ) );
    REQUIRE( !pidx->can_detect_type(distribution_type::CONTIGUOUS) );
  }
  SECTION( "local" ) {
    REQUIRE_NOTHROW( pidx->create_from_uniform_local_size(10) );
    REQUIRE( pidx->can_detect_type(distribution_type::CONTIGUOUS) );
  }
}

TEST_CASE( "MPI distributions, sanity","[mpi][distribution][cookie][10]" ) {
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1;
  REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,-1,s) );
  CHECK( d1->get_cookie()==entity_cookie::DISTRIBUTION );
};

TEST_CASE( "MPI distributions, operations","[mpi][distribution][11]" ) {
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1;
  REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,-1,s) );
  int *senders = new int[ntids], receives;
  REQUIRE_NOTHROW( receives = d1->reduce_scatter(senders,mytid) );
}

TEST_CASE( "MPI distributions, local stuff","[mpi][distribution][13]" ) {
  
  int nlocal = 100, s = nlocal*ntids;
  distribution *d1 = 
    new mpi_block_distribution(decomp,-1,s);
  CHECK( d1->has_defined_type() );
  CHECK( d1->volume(mycoord)==nlocal );
  auto
    f = domain_coordinate( std::vector<index_int>{nlocal*mytid} ),
    l = domain_coordinate( std::vector<index_int>{nlocal*(mytid+1)-1} );
  CHECK( d1->first_index_r(mycoord)==f );
  CHECK( d1->contains_element(mycoord,d1->first_index_r(mycoord)) );
  CHECK( ( !d1->is_valid_index(f-1) || !d1->contains_element(mycoord,f-1)) );
	 
  // d2 is a copy of d1
  distribution *d2;
  REQUIRE_NOTHROW( d2 = new mpi_distribution(d1) );
  domain_coordinate d2c(0); index_int d2s;
  REQUIRE_NOTHROW( d2s = d2->volume(mycoord) );
  CHECK( d2s==nlocal );
  REQUIRE_NOTHROW( d2c = d2->first_index_r(mycoord) );
  CHECK( d2c==f );
  CHECK( d2->contains_element(mycoord,f) );
  CHECK( ( !d2->is_valid_index(f-1) || !d2->contains_element(mycoord,f-1)) );
  
  // make sure d1 was not corrupted by creating d2
  CHECK( d1->last_index_r(mycoord)==l );
  CHECK( d1->contains_element(mycoord,l) );
  CHECK_NOTHROW( d1->contains_element
		 (mycoord,domain_coordinate( std::vector<index_int>{s} ) ) );
  CHECK( ( !d1->is_valid_index(l+1) || !d1->contains_element(mycoord,l+1) ) );
  CHECK_NOTHROW
    ( d1->contains_element(mycoord,domain_coordinate( std::vector<index_int>{-1} )) );

  // replicated has everyone the same
  distribution *d3 = new mpi_replicated_distribution(decomp);
  CHECK( d3->volume(mycoord)==1 );
  CHECK( d3->first_index_r(mycoord)==domain_coordinate_zero(1) );
  CHECK( d3->last_index_r(mycoord)==domain_coordinate_zero(1) );

  // scalar is block size=1
  distribution *d4 = new mpi_scalar_distribution(decomp);
  CHECK( d4->volume(mycoord)==1 );
  CHECK( d4->first_index_r(mycoord).coord(0)==mytid );
  CHECK( d4->last_index_r(mycoord).coord(0)==mytid );
}

TEST_CASE( "Distribution creation","[mpi][distribution][14]" ) {
  distribution *d; //parallel_structure *p = new parallel_structure(decomp);
  index_int
    localsize = 10*(mytid+1),
    myfirst = 10*mytid*(mytid+1)/2,
    globalsize = 10*ntids*(ntids+1)/2; const char *path;
  REQUIRE_NOTHROW( d = new mpi_distribution(decomp) );
  SECTION( "vector of local" ) { path = "vector of locals";
    std::vector<index_int> sizes(ntids);
    for (int tid=0; tid<ntids; tid++)
      sizes[tid] = 10*(tid+1);
    REQUIRE_NOTHROW( d->create_from_local_sizes( sizes ) );
  }
  SECTION( "unique local" ) { path = "from unique local";
    auto my_local = std::shared_ptr<multi_indexstruct>
          ( new contiguous_multi_indexstruct
	    ( domain_coordinate( std::vector<index_int>{ 1 } ),
	      domain_coordinate( std::vector<index_int>{ localsize } ) ) );
    REQUIRE_NOTHROW( d->create_from_unique_local(my_local) );
  }
  REQUIRE_NOTHROW( d->memoize() );
  INFO( "distribution created: " << path );
  index_int s;
  REQUIRE_NOTHROW( s = d->global_volume() );
  CHECK( s==globalsize );
  REQUIRE_NOTHROW( s = d->volume(mycoord) );
  CHECK( s==localsize );
  REQUIRE_NOTHROW( s = d->first_index_r(mycoord)[0] );
  CHECK( s==myfirst );
}

TEST_CASE( "factories","[mpi][distribution][15]" ) {
  distribution *block,*scalar;
  REQUIRE_NOTHROW( block = new mpi_block_distribution(decomp,10,-1) );
  REQUIRE_NOTHROW( scalar = block->new_scalar_distribution() );
  CHECK( scalar->has_type_contiguous() );
  CHECK( scalar->volume(mycoord)==1 );
  CHECK( scalar->first_index_r(mycoord).at(0)==mytid );
  std::shared_ptr<object> scalar_object;
  REQUIRE_NOTHROW( scalar_object = block->new_object(scalar) );
  //  CHECK( scalar->has_data_status_unallocated() );
}

TEST_CASE( "Operated distributions with modulo","[mpi][distribution][modulo][24]" ) {

  INFO( "mytid=" << mytid );

  int nlocal = 10, gsize = nlocal*ntids;
  distribution *d1 = 
    new mpi_block_distribution(decomp,-1,gsize);
  // record information for the original distribution
  auto
    first = d1->first_index_r(mycoord),
    last = d1->last_index_r(mycoord);
  index_int
    localsize = d1->volume(mycoord);

  // the unshifted distribution
  CHECK( d1->volume(mycoord)==localsize );
  CHECK( arch->get_protocol()==protocol_type::MPI );
  CHECK( d1->get_protocol()==protocol_type::MPI );
  CHECK( d1->local_allocation()==nlocal );
  CHECK( d1->contains_element(mycoord,first) );
  CHECK( d1->contains_element(mycoord,last) );
  
  // now check information for the shifted distribution, modulo
  distribution *d1shift = 
    new mpi_block_distribution(decomp,-1,gsize);
  auto shift_op = new multi_ioperator( ioperator(">>1") );
  CHECK( shift_op->is_modulo_op() );
  REQUIRE_NOTHROW( d1shift->operate( shift_op ) );

  //int fshift=MOD(first+1,gsize),lshift=MOD(last+1,gsize);
  auto fshift=(first+1)%gsize, lshift=(last+1)%gsize;
  CHECK( d1shift->volume(mycoord)==localsize );
  CHECK( d1shift->contains_element(mycoord,fshift) );
}

TEST_CASE( "dividing parstruct","[structure][ortho][25]" ) {

  int nlocal = 8, k,gsize = nlocal*ntids;
  parallel_structure *level_dist, *new_dist;
  ioperator coarsen; const char *path;
  // SECTION( "div:" ) { path = "div:2";
  //   coarsen = new ioperator(":2");
  // }
  SECTION( "div/" ) { path = "div/2";
    coarsen = ioperator("/2");
  }
  INFO( "path: " << path );

  REQUIRE_NOTHROW( level_dist = new parallel_structure(decomp) );
  REQUIRE_NOTHROW( level_dist->create_from_global_size(gsize) );
  INFO( "original dist: " << level_dist->as_string() );
  CHECK( level_dist->volume(mycoord)==nlocal );

  { // see what we do with just a processor structure
    std::shared_ptr<multi_indexstruct> coarse,fine;
    REQUIRE_NOTHROW( fine = level_dist->get_processor_structure(mycoord) );
    INFO( "fine struct: " << fine->as_string() );
    REQUIRE_NOTHROW( coarse = fine->operate(coarsen) );
    INFO( "coarse struct: " << coarse->as_string() );
    CHECK( coarse->volume()==fine->volume()/2 );
  }
  // now for real with the distribution
  REQUIRE_NOTHROW( level_dist = level_dist->operate(coarsen) );
  INFO( "operated dist: " << level_dist->as_string() );
  fmt::print("operated dist: {}\n",level_dist->as_string() );

  CHECK( level_dist->volume(mycoord)==nlocal/2 );
}

TEST_CASE( "dividing distribution","[distribution][ortho][26]" ) {

  int nlocal = 8, k,gsize = nlocal*ntids;
  distribution *level_dist, *new_dist;
  ioperator coarsen; const char *path;
  // SECTION( "div:" ) { path = "div:2";
  //   coarsen = new ioperator(":2");
  // }
  SECTION( "div/" ) { path = "div/2";
    coarsen = ioperator("/2");
  }
  INFO( "path: " << path );
  for (int k=1; k<=3; k++) {
    INFO( "k=" << k );
    REQUIRE_NOTHROW( level_dist = new mpi_block_distribution(decomp,k,-1,gsize) );
    INFO( "original dist: " << level_dist->as_string() );
    CHECK( level_dist->get_orthogonal_dimension()==k );
    CHECK( level_dist->local_allocation()==k*nlocal );

    REQUIRE_NOTHROW( level_dist = level_dist->operate(coarsen) );
    INFO( "operated dist: " << level_dist->as_string() );

    CHECK( level_dist->local_allocation()==k*nlocal/2 );
    INFO( "divided dist: " << level_dist->as_string() );
    CHECK( level_dist->get_orthogonal_dimension()==k );
    //CHECK( level_dist->has_type_contiguous() );
  }
}

TEST_CASE( "extending distributions","[distribution][extend][27]" ) {
  if (ntids<2) {
    printf("test 27 needs multiple processes ????\n");
    //  return;
  }

  int dim = 1;
  int nlocal=100,nglobal=nlocal*ntids;
  distribution *d1,*d2;
  REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,nlocal,-1) );
  domain_coordinate
    my_first = d1->first_index_r(mycoord), my_last = d1->last_index_r(mycoord);

  int shift = 1;
  //  SECTION( "keep it contiguous" ) {
    shift = 1;
    //}
  // SECTION( "make it composite" ) {
  //   shift = 2;
  // }
  INFO( "mytid=" << mytid << "\nusing shift: " << shift );

  {
    domain_coordinate the_first(dim), the_last(dim);
    REQUIRE_NOTHROW( the_first = d1->first_index_r(mycoord) );
    REQUIRE_NOTHROW( the_last = d1->last_index_r(mycoord) );
    std::shared_ptr<multi_indexstruct> estruct,xstruct;

    processor_coordinate close(1);
    REQUIRE_NOTHROW( close = decomp->get_origin_processor() );
    if (mycoord==close)
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct ( the_first-shift ) ) );
    else if (mycoord==decomp->get_farpoint_processor())
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct( the_last+shift ) ) );
    else
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new empty_multi_indexstruct(dim) ) );

    REQUIRE_NOTHROW( d2 = d1->extend(mycoord,estruct) );
//     fmt::print("going to print\n");
//     fmt::print("Extended structure: {}",d2->as_string());
  }
  // the print statement in d1->extend succeeds, the following doesn't
  return;

  if (mytid==0 || mytid==ntids-1)
    CHECK( d2->volume(mycoord)==(nlocal+1) );
  else
    CHECK( d2->volume(mycoord)==(nlocal+2) );
  return ;
  if (mytid==0) 
    CHECK( d2->first_index_r(mycoord)==my_first );
  else
    CHECK( d2->first_index_r(mycoord)==my_first-shift );
  if (mytid==ntids-1) 
    CHECK( d2->last_index_r(mycoord)==my_last );
  else
    CHECK( d2->last_index_r(mycoord)==my_last+shift );
}

TEST_CASE( "orthogonal dimension","[distribution][ortho][30]" ) {
  int nlocal=100,nglobal=nlocal*ntids;
  distribution *d1; int k; const char *path;
  SECTION( "default k=1" ) { k=1; path = "k=1 by default";
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,nlocal,-1) );
  }
  SECTION( "explicit k=1" ) { k=1; path = "k=1 explicit";
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,k,nlocal,-1) );
  }
  SECTION( "k=2" ) { k=2; path = "k=2 explicit";
    REQUIRE_NOTHROW( d1 = new mpi_block_distribution(decomp,k,nlocal,-1) );
  }
  CHECK( d1->volume(mycoord)==nlocal );
  CHECK( d1->local_allocation_p(mycoord)==k*nlocal );
}

TEST_CASE( "Cyclic distributions","[distribution][cyclic][40]" ) {
  distribution *d;
  //  REQUIRE_THROWS( d = new mpi_cyclic_distribution(decomp,-1,-1) );
  //  REQUIRE_THROWS( d = new mpi_cyclic_distribution(decomp,1,ntids+1) );

  // each proc gets 2 elements: mytid,mytid+ntids
  REQUIRE_NOTHROW( d = new mpi_cyclic_distribution(decomp,-1,2*ntids) );
  CHECK( d->volume(mycoord)==2 );
  CHECK( d->first_index_r(mycoord).coord(0)==mytid );
  CHECK( d->last_index_r(mycoord).coord(0)==mytid+ntids );
}

TEST_CASE( "Block cyclic distributions","[distribution][cyclic][41]" ) {
  distribution *d;
  REQUIRE_NOTHROW( d = new mpi_blockcyclic_distribution(decomp,5,4,-1) );
  CHECK( d->volume(mycoord)==20 );
  CHECK( d->first_index_r(mycoord).coord(0)==mytid*5 );
  CHECK( d->get_processor_structure(mycoord)->get_component(0)->contains_element( 5*ntids + mytid*5 ) );
}

index_int pfunc1(int p,index_int i) {
  return 3*p+i;
}

index_int pfunc2(int p,index_int i) {
  return 3*(p/2)+i;
}

TEST_CASE( "Function-specified distribution","[distribution][50]" ) {
  INFO( "mytid=" << mytid );
  mpi_distribution *d1,*d2;
  int nlocal = 3;

  CHECK_NOTHROW( d1 = new mpi_distribution(decomp,&pfunc1,nlocal ) );
  CHECK( d1->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    index_int iglobal = 3*mytid+i;
    CHECK( d1->contains_element(mycoord,domain_coordinate(std::vector<index_int>{iglobal})) );
    CHECK( d1->find_index(iglobal)==mytid );
  }

  CHECK_NOTHROW( d2 = new mpi_distribution(decomp,&pfunc2,nlocal ) );
  CHECK( d2->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    index_int iglobal = 3*(mytid/2)+i; // proc 0,1 have same data, likewise 2,3, 4,5
    CHECK( d2->contains_element(mycoord,domain_coordinate(std::vector<index_int>{iglobal})) );
    CHECK( d2->find_index(iglobal,mytid)==mytid );
    CHECK( d2->find_index(iglobal)==2*(mytid/2) ); // the first proc with my data is 2*(p/2)
  }
}

TEST_CASE( "Distribution transformations","[distribution][abut][60]" ) {
  distribution *block =
    new mpi_block_distribution(decomp,10*(mytid+1),-1), *newblock;
  auto first = block->first_index_r(mycoord);
  auto times2 = ioperator("x2");
  index_int gsize = block->global_volume();
  domain_coordinate last_coord(1);
  REQUIRE_NOTHROW( last_coord = block->get_enclosing_structure()->last_index_r()+1 );
  const char *path;
  // SECTION( "operate pidx" ) { path = "operate pidx";
  //   parallel_indexstruct *pidx, *new_pidx;
  //   REQUIRE_NOTHROW( pidx = block->get_dimension_structure(0) );
  //   SECTION( "classic transformation" ) {
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(times2) );
  //   }
  //   SECTION( "sigma transformation by point" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator(times2);
  //     CHECK( pidx_times2->is_point_operator() );
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(pidx_times2) );
  //   }
  //   SECTION( "sigma transformation by struct" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator
  //   	( [times2] (indexstruct &i) -> std::shared_ptr<indexstruct>
  //   	  { return i.operate(times2); } );
  //     REQUIRE_NOTHROW( new_pidx = pidx->operate(pidx_times2) );
  //   }
  //   SECTION( "dynamic transformation" ) {
  //     sigma_operator *pidx_times2 = new sigma_operator
  //   	( [times2] (indexstruct &i) -> std::shared_ptr<indexstruct>
  //   	  { std::shared_ptr<indexstruct> opstruct = i.operate(times2);
  //   	    //fmt::print("operate on struct {} gives {}\n",i.as_string(),opstruct->as_string());
  //   	    return opstruct; } );
  //     distribution_sigma_operator *dist_times2 = new distribution_sigma_operator(pidx_times2);
  //     std::shared_ptr<indexstruct> newstruct;
  //     REQUIRE_NOTHROW( newstruct = dist_times2->operate(0,block,mycoord) );
  //     REQUIRE_NOTHROW( new_pidx = new parallel_indexstruct(decomp->domains_volume()) );
  //     REQUIRE_NOTHROW( new_pidx->set_processor_structure(mytid,newstruct) );
  //     CHECK( !new_pidx->is_known_globally() );
  //   }
  //   INFO( fmt::format("new structure: {}",new_pidx->as_string()) );
  //   CHECK( new_pidx->first_index_r(mytid)==pidx->first_index_r(mytid)*2 );
  //   CHECK( new_pidx->get_processor_structure(mytid)->stride()==1 );
  //   CHECK( new_pidx->local_size(mytid)==pidx->local_size(mytid)*2-1 );
  //   parallel_structure *parstruct;
  //   REQUIRE_NOTHROW( parstruct = new parallel_structure(decomp,new_pidx) );
  //   //REQUIRE_NOTHROW( newblock = new mpi_distribution(/*decomp,*/parstruct) );
  // }
  SECTION( "operate structure" ) { path = "operate structure";
    parallel_structure *parstruct,*new_struct;
    REQUIRE_NOTHROW( parstruct = dynamic_cast<parallel_structure*>(block) );
    CHECK( parstruct!=nullptr );
    SECTION( "operate parstruct" ) {
      REQUIRE_NOTHROW( new_struct = parstruct->operate(times2) );
    }
    SECTION( "sigma transformation" ) {
      auto *pidx_times2 =
	new multi_sigma_operator( sigma_operator(times2) );
      CHECK( pidx_times2->is_point_operator() );
      REQUIRE_NOTHROW( new_struct = parstruct->operate(pidx_times2) );
    }
    REQUIRE_NOTHROW( newblock = new mpi_distribution(new_struct) );
  }
  SECTION( "operate distribution" ) { path = "operate distribution";
    auto *pidx_times2 =
      new multi_sigma_operator( sigma_operator(times2) );
    REQUIRE_NOTHROW( newblock = block->operate(pidx_times2) );
  }
  SECTION( "stretch distribution" ) { path = "stretch distribution";
    fmt::print("Stretch to {}\n",last_coord.as_string());
    auto double_last = last_coord*2;
    auto *stretch2 =
      new distribution_stretch_operator(double_last);
    REQUIRE_NOTHROW( newblock = block->operate(stretch2) );
    distribution_sigma_operator *op;
    REQUIRE_NOTHROW( op = new distribution_abut_operator(mycoord) );
    REQUIRE_NOTHROW( newblock = newblock->operate(op) );
  }
  INFO( fmt::format("Derive new distribution by: {}",path) );
  INFO( fmt::format("old first: {}, new first: {}",
		    first.as_string(),newblock->first_index_r(mycoord).as_string()) );
  //CHECK( newblock->first_index_r(mycoord)==first*2 );
  CHECK( newblock->global_volume()==gsize*2 );
#if 0
  transform_localsize;
  parallel_structure *struc, *new_struc;
  transform_localsize;  
  distribution *newblock;
  REQUIRE_NOTHROW( newblock = block->transform_localsize
		   ( [] (index_int s) -> index_int { return 2*s; } ) );
#endif
}

std::shared_ptr<multi_indexstruct> transform_by_shift
    (distribution *unbalance,processor_coordinate &me,distribution *load) {
  if (!load->has_type_replicated())
    throw(std::string("Load description needs to be replicated"));
  if (load->volume(me)!=unbalance->domains_volume())
    throw(fmt::format("Load vector has {} items, for {} domains",
		      load->volume(me),unbalance->domains_volume()));
  if (!unbalance->is_known_globally())
    throw(fmt::format("Can only transform-shift globally known distributions"));

  // to work!

  decomposition *decomp = dynamic_cast<decomposition*>(unbalance);
  if (decomp==nullptr)
    throw(std::string("Could not case to decomposition"));
  std::shared_ptr<multi_indexstruct> old_pstruct,new_pstruct;
  try {
    old_pstruct = unbalance->get_processor_structure(me);
    new_pstruct = old_pstruct->operate
      ( new multi_ioperator( ioperator(">>2") ) );
    fmt::print("shift operation {} -> {}\n",
	       old_pstruct->as_string(),new_pstruct->as_string());
  } catch (std::string c) { fmt::print("{}: Error <<{}>>\n",me.as_string(),c);
    throw(std::string("Could not redistribute by shift")); }
  return new_pstruct;
}

TEST_CASE( "Distribution operation by simple shift","[distribution][operate][61]" ) {
  index_int nlocal = 10*mytid+1;
  distribution *block =
    new mpi_block_distribution(decomp,nlocal,-1), *newblock,
    *load = new mpi_replicated_distribution(decomp,ntids);
  block->set_name("blockdist61");
  auto first = block->first_index_r(mycoord);
  auto saved_mycoord = std::shared_ptr<processor_coordinate>
    ( new processor_coordinate(mycoord) );
  auto average =
    new distribution_sigma_operator
    ( [load] (distribution *d,processor_coordinate &p) -> std::shared_ptr<multi_indexstruct> {
      return transform_by_shift(d,p,load); } );
  REQUIRE_NOTHROW( newblock = block->operate(average) );
  index_int check_local;
  REQUIRE_NOTHROW( check_local = newblock->volume(mycoord) );
  CHECK( check_local==nlocal );
  auto p_old = block->get_processor_structure(mycoord),
    p_new = newblock->get_processor_structure(mycoord);
  INFO( fmt::format("old block: {}, shifted block: {}",
		    p_old->as_string(),p_new->as_string()) );
  CHECK( p_new->first_index_r()==p_old->first_index_r()+2 );
}

std::shared_ptr<multi_indexstruct> transform_by_multi
    (distribution *unbalance,processor_coordinate &me,distribution *load) {
  if (!load->has_type_replicated())
    throw(std::string("Load description needs to be replicated"));
  if (load->volume(me)!=unbalance->domains_volume())
    throw(fmt::format("Load vector has {} items, for {} domains",
		      load->volume(me),unbalance->domains_volume()));
  if (!unbalance->is_known_globally())
    throw(fmt::format("Can only transform-shift globally known distributions"));

  // to work!

  decomposition *decomp = dynamic_cast<decomposition*>(unbalance);
  if (decomp==nullptr)
    throw(std::string("Could not cast to decomposition"));
  std::shared_ptr<multi_indexstruct> old_pstruct,new_pstruct;
  try {
    old_pstruct = unbalance->get_processor_structure(me);
    new_pstruct = old_pstruct->operate
      ( new multi_ioperator( ioperator("x2") ) );
    fmt::print("shift operation {} -> {}\n",
	       old_pstruct->as_string(),new_pstruct->as_string());
  } catch (std::string c) { fmt::print("{}: Error <<{}>>\n",me.as_string(),c);
    throw(std::string("Could not redistribute by shift")); }
  return new_pstruct;
}

TEST_CASE( "Distribution operation by local blowup","[distribution][operate][abut][62]" ) {
  index_int nlocal = 10*mytid+1;
  distribution *block =
    new mpi_block_distribution(decomp,nlocal,-1), *newblock,
    *load = new mpi_replicated_distribution(decomp,ntids);
  block->set_name("blockdist61");
  auto first = block->first_index_r(mycoord);
  auto saved_mycoord = std::shared_ptr<processor_coordinate>
    ( new processor_coordinate(mycoord) );
  auto average =
    new distribution_sigma_operator
    ( [load] (distribution *d,processor_coordinate &p) -> std::shared_ptr<multi_indexstruct> {
      return transform_by_multi(d,p,load); } );
  REQUIRE_NOTHROW( newblock = block->operate(average) );
  REQUIRE_NOTHROW( newblock = newblock->operate( new distribution_abut_operator(mycoord) ) );
  auto p_old = block->get_processor_structure(mycoord),
    p_new = newblock->get_processor_structure(mycoord);
  INFO( fmt::format("old block: {}, shifted block: {}",
		    p_old->as_string(),p_new->as_string()) );
  CHECK( p_new->volume()==2*p_old->volume() );

  std::shared_ptr<object> new_object;
  REQUIRE_NOTHROW( new_object = std::shared_ptr<object>( new mpi_object(newblock) ) );
}

TEST_CASE( "Make a distribution abutting","[distribution][operate][abut][63]" ) {
  index_int nlocal = 10;
  parallel_structure *pstr;
  REQUIRE_NOTHROW( pstr = new parallel_structure(decomp) );
  for (int p=0; p<ntids; p++) {
    processor_coordinate pcoord( std::vector<int>{p} );
    domain_coordinate
      first(std::vector<index_int>{p*nlocal}),
      last(std::vector<index_int>{(p+2)*nlocal-1});
    auto pstruct = std::shared_ptr<multi_indexstruct>
      ( new contiguous_multi_indexstruct(first,last) );
    pstr->set_processor_structure(pcoord,pstruct);
  }
  distribution *messy;
  REQUIRE_NOTHROW( messy = new mpi_distribution(pstr) );
  std::shared_ptr<object> messy_object;
  REQUIRE_NOTHROW( messy_object = std::shared_ptr<object>( new mpi_object(messy) ) );
  distribution *sizes;
  REQUIRE_NOTHROW( sizes = new mpi_gathered_distribution(decomp) );
  CHECK( sizes->volume(mycoord)==ntids );
  std::shared_ptr<object> size_summary;
  REQUIRE_NOTHROW( size_summary = std::shared_ptr<object>( new mpi_object(sizes) ) );
  kernel *gather_sizes;
  REQUIRE_NOTHROW( gather_sizes = new mpi_gather_kernel
		   (messy_object,size_summary,[] ( kernel_function_args ) -> void {
		     //double *out_data = outvector->get_data(p);
		     //out_data[0] = invectors->at(0)->volume(p);
		   } ) );
  REQUIRE_NOTHROW( gather_sizes->analyze_dependencies() );
  REQUIRE_NOTHROW( gather_sizes->execute() );
  return;

  distribution *clean;
  REQUIRE_NOTHROW( clean = messy->operate( new distribution_abut_operator(mycoord) ) );
  for (int p=0; p<ntids; p++) {
    processor_coordinate pcoord( std::vector<int>{p} );
    auto pstruct = clean->get_processor_structure(pcoord);
    INFO( fmt::format("P={}, struct={}",p,pstruct->as_string()) );
    CHECK( pstruct->first_index_r().at(0)==2*p*nlocal );
    CHECK( pstruct->last_index_r().at(0)==2*(p+1)*nlocal );
  }
}

TEST_CASE( "Distribution stretch","[distribution][stretch][64]" ) {
  distribution *block, *stretched;
  float factor;

  SECTION( "regular blocked" ) {
    index_int nlocal = 10;
    REQUIRE_NOTHROW( block = new mpi_block_distribution(decomp,nlocal,-1) );
    SECTION( "2" ) { factor = 2; };
    SECTION( "5" ) { factor = 5; };
    SECTION( "1/3" ) { factor = 1./3; };
  }
  SECTION( "irregular blocked" ) {
    index_int nlocal = 10+8*mytid;
    REQUIRE_NOTHROW( block = new mpi_block_distribution(decomp,nlocal,-1) );
    SECTION( "2" ) { factor = 2; };
    SECTION( "5" ) { factor = 5; };
    SECTION( "1/3" ) { factor = 1./3; };
  }
  INFO( fmt::format("Stretching {} by factor {}",block->as_string(),factor) );
  index_int gsize;
  REQUIRE_NOTHROW( gsize = factor*block->global_volume() );
  auto big = domain_coordinate( std::vector<index_int>{gsize} );
  INFO( fmt::format("Stretching to {}",big.as_string()) );
  REQUIRE( big.get_dimensionality()==1 );

  distribution_sigma_operator *stretch;
  REQUIRE_NOTHROW( stretch = new distribution_stretch_operator(big) );
  REQUIRE_NOTHROW( stretched = block->operate(stretch) );
  INFO( fmt::format("Stretched distro: {}",stretched->as_string()) );
  CHECK( stretched->global_volume()==gsize );
  CHECK( stretched->is_known_globally() );

  auto
    old_pstruct = block->get_processor_structure(mycoord),
    new_pstruct = stretched->get_processor_structure(mycoord);
  index_int my_nlocal = old_pstruct->volume();
  index_int new_nlocal = factor*my_nlocal;
  INFO( fmt::format
	("old pstruct: {}, volume={},\nintended volume={},\nnew pstruct: {}, volume={}",
	 old_pstruct->as_string(),my_nlocal,
	 new_nlocal,
	 new_pstruct->as_string(),new_pstruct->volume()) );
  CHECK( ( new_pstruct->volume()>=new_nlocal-1 && new_pstruct->volume()<=new_nlocal+1 ) );

}

TEST_CASE( "Distribution operation by averaging","[distribution][operate][abut][65]" ) {
  index_int nlocal = 12*mytid+1;
  distribution *block =
    new mpi_block_distribution(decomp,nlocal,-1), *newblock,*finalblock,
    *load = new mpi_replicated_distribution(decomp,ntids);
  auto stats_object = std::shared_ptr<object>( new mpi_object(load) );
  index_int gsize = block->global_volume();
  domain_coordinate glast = block->global_size();
  block->set_name("blockdist61");
  double *stats_data = stats_object->get_data(mycoord);
  for (int p=0; p<ntids; p++) stats_data[p] = 1.;
  auto average =
    new distribution_sigma_operator
    ( [stats_data] (distribution *d) -> distribution* {
      return transform_by_average(d,stats_data); } );
  auto stretch_to =
    new distribution_abut_operator(mycoord);

  REQUIRE_NOTHROW( newblock = block->operate(average) );
  REQUIRE_NOTHROW( finalblock = newblock->operate(stretch_to) );

  INFO( fmt::format("old block: {}, avg block: {}, stretch block : {}",
		    block->get_processor_structure(mycoord)->as_string(),
		    newblock->get_processor_structure(mycoord)->as_string(),
		    finalblock->get_processor_structure(mycoord)->as_string()
		    ) );
  index_int checksize;
  REQUIRE_NOTHROW( checksize = finalblock->global_volume() );
  printf("[65] stretch operator result not checked\n");
  //  CHECK( checksize==gsize );

  std::shared_ptr<object> new_object;
  REQUIRE_NOTHROW( new_object = std::shared_ptr<object>( new mpi_object(finalblock) ) );
}

TEST_CASE( "Masked distribution creation","[distribution][mask][70]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5;
  processor_mask *mask;

  fmt::MemoryWriter path;
  SECTION( "create mask by adding" ) { path.write("adding odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
    for (int p=1; p<ntids; p+=2 ) {
      processor_coordinate c(1); c.set(0,p);
      REQUIRE_NOTHROW( mask->add(c) );
    }
  }
  SECTION( "create mask by subtracting" ) { path.write("subtracting odd");
    REQUIRE_NOTHROW( mask = new processor_mask(decomp,ntids) );
    for (int p=0; p<ntids; p+=2) {
      processor_coordinate c(1); c.set(0,p);
      REQUIRE_NOTHROW( mask->remove(p) );
    }
  }
  INFO( "masked created by: " << path.str() );

  mpi_distribution
    *block = new mpi_block_distribution(decomp,localsize,-1),
    *masked_block = new mpi_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto 
    whole_vector = std::shared_ptr<object>( new mpi_object(block) ),
    masked_vector = std::shared_ptr<object>( new mpi_object(masked_block) );
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK( whole_vector->get_allocated_space()==localsize );
  if (mytid%2==1)
    CHECK( masked_vector->get_allocated_space()==localsize );
  else
    CHECK( masked_vector->get_allocated_space()==0 );
  double *data;
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  {
    CHECK( block->lives_on(mycoord) );
    if (mytid%2==1) {
      CHECK( masked_block->lives_on(mycoord) );
      REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
    } else {
      CHECK( !masked_block->lives_on(mycoord) );
      REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
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
  int mytid_j = mytid%ntids_j, mytid_i = mytid/ntids_j;

  processor_coordinate layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(2) );
  mpi_decomposition *mdecomp;
  SECTION( "default splitting of processor grid" ) {
    REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,layout) );
  }
  // SECTION( "explicit splitting of processor grid" ) {
  //   std::vector<int> grid; grid.push_back(2); grid.push_back(2);
  //   REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,grid) );
  // }

  processor_coordinate mycoord; 
  REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
  INFO( "p: " << mytid << ", pcoord: " << mycoord.coord(0) << "," << mycoord.coord(1) );
  CHECK( mytid==mycoord.linearize(mdecomp) );
  CHECK( mycoord.get_dimensionality()==2 );
  INFO( "mytid=" << mytid << ", s/b " << mytid_i << "," << mytid_j );
  CHECK( mycoord.coord(0)==mytid_i );
  CHECK( mycoord.coord(1)==mytid_j );
  
  int nlocal = 10; index_int g;
  std::vector<index_int> domain_layout;
  g = ntids_i*(nlocal+1);
  domain_layout.push_back(g);

  g = ntids_j*(nlocal+2);
  domain_layout.push_back(g);

  mpi_distribution *d;
  REQUIRE_NOTHROW( d = new mpi_block_distribution(mdecomp,domain_layout) );
  CHECK( d->get_dimensionality()==2 );
  std::shared_ptr<multi_indexstruct> local_domain;
  REQUIRE_NOTHROW( local_domain = d->get_processor_structure(mycoord) );
  CHECK( local_domain->get_dimensionality()==2 );
  CHECK( local_domain->get_component(0)->volume()==nlocal+1 );
  CHECK( local_domain->get_component(1)->volume()==nlocal+2 );

}

TEST_CASE( "multidimensional distributions error test","[multi][distribution][101]" ) {
  int ntids_i,ntids_j;
  if (ntids!=4) { printf("101 grid example needs exactly 4 procs\n"); return; }

  mpi_decomposition *mdecomp;
  SECTION( "explicit splitting of processor grid" ) {
    processor_coordinate endpoint = processor_coordinate(2);
    endpoint.set(0,2); endpoint.set(1,2);
    REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,endpoint) );
  }
  // we should not test this because we can overdecompose, true?
  // SECTION( "incorrect splitting of processor grid" ) {
  //   processor_coordinate *endpoint = new processor_coordinate(2);
  //   endpoint->set(0,2); endpoint->set(1,1);
  //   REQUIRE_THROWS( mdecomp = new mpi_decomposition(arch,endpoint) );
  // }
  SECTION( "pencil splitting of processor grid" ) {
    processor_coordinate endpoint = processor_coordinate(2);
    endpoint.set(0,1); endpoint.set(1,4);
    REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,endpoint) );
  }
}

TEST_CASE( "pencil distribution","[multi][distribution][pencil][102]" ) {
}

