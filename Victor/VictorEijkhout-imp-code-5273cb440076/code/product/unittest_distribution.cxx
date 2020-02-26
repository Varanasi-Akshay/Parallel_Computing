/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the MPI+OMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for product-based distributions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "product_base.h"
#include "product_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"

TEST_CASE( "Inheritance structure","[00]" ) {
  // this test simply doesnt' compile if there is something wrong
  if (mytid==0) {
    CHECK( arch->is_first_proc(mytid) );
  } else {
    CHECK( !arch->is_first_proc(mytid) );
  }
}

TEST_CASE( "MPI distributions, local stuff","[mpi][distribution][13]" ) {
  
  int nlocal = 100, s = nlocal*mpi_nprocs;
  distribution *d1;
  REQUIRE_NOTHROW( d1 = new product_block_distribution(decomp,-1,s) );
  CHECK( d1->has_defined_type() );
  CHECK( d1->volume(mycoord)==nlocal );
  //int f=nlocal*mytid, l=nlocal*(mytid+1)-1;
  domain_coordinate f(std::vector<index_int>{nlocal*mytid}),
    l(std::vector<index_int>{nlocal*(mytid+1)-1});
  CHECK( d1->first_index_r(mycoord)==f );
  CHECK( d1->contains_element(mycoord,f) );
  auto fm1 = f-1;
  CHECK( ( !d1->is_valid_index(fm1) ||
	   !d1->contains_element(mycoord,fm1) ) );

  if (0) {
    distribution *d2;
    REQUIRE_NOTHROW( d2 = new product_distribution(d1) );
    domain_coordinate d2s(1);
    REQUIRE_NOTHROW( d2s = d2->local_size_r(mycoord) );
    CHECK( d2s[0]==nlocal );
    REQUIRE_NOTHROW( d2s = d2->first_index_r(mycoord) );
    CHECK( d2s==f );
    CHECK( d2->contains_element(mycoord,f) );
    auto fm1 = f-1;
    CHECK( ( !d2->is_valid_index(fm1) ||
	     !d2->contains_element(mycoord,fm1) ) );
  } else printf("copy constructor test disabled\n");

  CHECK( d1->last_index_r(mycoord)==l );
  CHECK( d1->contains_element(mycoord,l) );
  auto ds = domain_coordinate( std::vector<index_int>{s} );
  CHECK_NOTHROW( !d1->contains_element(mycoord,ds) );
  auto lp1 = l+1;
  CHECK( ( !d1->is_valid_index(lp1) ||
	   !d1->contains_element(mycoord,lp1) ) );
  auto dc1 = domain_coordinate( std::vector<index_int>{-1} );
  CHECK_NOTHROW( !d1->contains_element(mycoord,dc1) );

  distribution *d3 = new product_replicated_distribution(decomp);
  CHECK( d3->volume(mycoord)==1 );
  CHECK( d3->first_index_r(mycoord)[0]==0 );
  CHECK( d3->last_index_r(mycoord)[0]==0 );
}

TEST_CASE( "Operated distributions with modulo","[mpi][distribution][modulo][14]" ) {

  INFO( "mytid=" << mytid );

  int gsize = 10*mpi_nprocs;
  distribution *d1 = 
    new product_block_distribution(decomp,-1,gsize);
  // record information for the original distribution
  domain_coordinate first = d1->first_index_r(mycoord),last = d1->last_index_r(mycoord);
  index_int localsize = d1->volume(mycoord);

  // the unshifted distribution
  CHECK( d1->volume(mycoord)==localsize );
  CHECK( d1->contains_element(mycoord,first) );
  CHECK( d1->contains_element(mycoord,last) );

  // now check information for the shifted distribution, modulo
  distribution *d1shift = 
    new product_block_distribution(decomp,-1,gsize);
  auto shift_op =  ioperator(">>1") ;
  CHECK( shift_op.is_modulo_op() );
  d1shift = d1shift->operate( shift_op );

  int fshift=MOD(first[0]+1,gsize),lshift=MOD(last[0]+1,gsize);
  CHECK( d1shift->volume(mycoord)==localsize );
  auto dfsh = domain_coordinate( std::vector<index_int>{fshift} );
  CHECK( d1shift->contains_element(mycoord,dfsh) );
  // CHECK( d1shift->contains_element(mycoord,domain_coordinate(1,lshift)) ); VLE what's wrong here?
  // if (mpi_nprocs>1) {
  //   CHECK( !d1shift->contains_element(mycoord,domain_coordinate(1,first)) );
  // }
}

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

  REQUIRE_NOTHROW( level_dist = new product_block_distribution(decomp,k,nlocal,-1) );

  auto coarsen =  ioperator(":2") ;

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  CHECK( level_dist->local_allocation()==k*nlocal );
  CHECK( new_dist->get_orthogonal_dimension()==k );
  CHECK( new_dist->local_allocation()==k*nlocal/2 );
}

// TEST_CASE( "two-dimensional distribution","[distribution][grid][20]" ) {
//   distribution *d; int n;
//   REQUIRE( (mpi_nprocs%2)==0 );
//   product_environment *gridenv;
//   REQUIRE_NOTHROW( gridenv = new product_environment( *env ) );
//   CHECK_THROWS( n = gridenv->grid_dimension() );
//   REQUIRE_THROWS( gridenv->set_grid_2d( mpi_nprocs/2,3 ) );
//   CHECK( !env->is_processor_grid() );
//   REQUIRE_NOTHROW( gridenv->set_grid_2d( mpi_nprocs/2,2 ) );
//   CHECK( gridenv->is_processor_grid() );
//   CHECK( !env->is_processor_grid() );
//   CHECK_NOTHROW( n = gridenv->grid_dimension() );
//   CHECK( n==2 );
//   CHECK( gridenv->nprocs()==mpi_nprocs );
//   CHECK_NOTHROW( d = new product_block_distribution( gridenv, 3,3,-1,-1 ) );
//   std::vector<int> *coord;
//   CHECK_THROWS( coord = env->proc_id_on_grid(mytid) );
//   CHECK_NOTHROW( coord = gridenv->proc_id_on_grid(mytid) );
//   REQUIRE( coord->size()==2 );
//   // check 2 dimensions
//   CHECK( (*coord)[0]<mpi_nprocs/2 );
//   CHECK( (*coord)[1]<2 );
//   CHECK( d->volume(mycoord)==9 );
//   REQUIRE_NOTHROW( delete gridenv );
//   CHECK( !env->is_processor_grid() );
// }

TEST_CASE( "extending distributions","[distribution][extend][27]" ) {
  if (mpi_nprocs<2) {
    printf("test 27 needs multiple processes\n"); return; }

  INFO( "mytid=" << mytid );
  int dim = 1, nlocal = 100, nglobal = nlocal*mpi_nprocs;
  distribution *d1,*d2;
  REQUIRE_NOTHROW( d1 = new product_block_distribution(decomp,nlocal,-1) );
  domain_coordinate
    my_first = d1->first_index_r(mycoord), my_last = d1->last_index_r(mycoord);

  int shift;
  SECTION( "keep it contiguous" ) {
    shift = 1;
  }
  SECTION( "make it composite" ) {
    shift = 2;
  }
  INFO( "using shift: " << shift );

  for (int p=0; p<mpi_nprocs; p++) {
    processor_coordinate pcoord(1);
    REQUIRE_NOTHROW( pcoord = decomp->coordinate_from_linear(p) );
    domain_coordinate
      the_first = d1->first_index_r(pcoord), the_last = d1->last_index_r(pcoord);
    std::shared_ptr<multi_indexstruct> estruct,xstruct;

    processor_coordinate close(1);
    REQUIRE_NOTHROW( close = decomp->get_origin_processor() );
    if (pcoord==close)
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct ( the_first-shift ) ) );
    else if (pcoord==decomp->get_farpoint_processor())
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new contiguous_multi_indexstruct( the_last+shift ) ) );
    else
      REQUIRE_NOTHROW( estruct = std::shared_ptr<multi_indexstruct>
		       ( new empty_multi_indexstruct(dim) ) );

    REQUIRE_NOTHROW( d2 = d1->extend(mycoord,estruct) );

    // if (p!=0) {
    //   multi_indexstruct *left;
    //   REQUIRE_NOTHROW( left = new contiguous_multi_indexstruct( d1->first_index_r(pcoord)-shift ) );
    //   REQUIRE_NOTHROW( d2 = d1->extend(pcoord,left) );
    //   CHECK( d2->first_index_r(pcoord)==(the_first-shift) );
    // }
    // if (p!=mpi_nprocs-1) {
    //   multi_indexstruct *right;
    //   REQUIRE_NOTHROW( right = new contiguous_multi_indexstruct( d1->last_index_r(pcoord)+shift ) );
    //   REQUIRE_NOTHROW( d1->extend(pcoord,right) );
    //   CHECK( d1->last_index_r(pcoord)==(the_last+shift) );
    // }
  }

  if (mytid==0 || mytid==mpi_nprocs-1)
    CHECK( d1->volume(mycoord)==(nlocal+1) );
  else
    CHECK( d1->volume(mycoord)==(nlocal+2) );
  if (mytid==0) 
    CHECK( d1->first_index_r(mycoord)==my_first );
  else
    CHECK( d1->first_index_r(mycoord)==my_first-shift );
  if (mytid==mpi_nprocs-1) 
    CHECK( d1->last_index_r(mycoord)==my_last );
  else
    CHECK( d1->last_index_r(mycoord)==my_last+shift );

}

#if 0

TEST_CASE( "Cyclic distributions","[distribution][cyclic][40]" ) {
  distribution *d;
  REQUIRE_THROWS( d = new product_cyclic_distribution(decomp,-1,-1) );
  REQUIRE_THROWS( d = new product_cyclic_distribution(decomp,1,mpi_nprocs+1) );
  REQUIRE_NOTHROW( d = new product_cyclic_distribution(decomp,-1,2*mpi_nprocs) );
  CHECK( d->volume(mycoord)==2 );
}

index_int pfunc1(int p,index_int i) {
  return 3*p+i;
}

index_int pfunc2(int p,index_int i) {
  return 3*(p/2)+i;
}

TEST_CASE( "Function-specified distribution","[distribution][50]" ) {
  INFO( "mytid=" << mytid );
  product_distribution *d1,*d2;
  int nlocal = 3;

  CHECK_NOTHROW( d1 = new product_distribution(decomp,&pfunc1,nlocal ) );
  CHECK( d1->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    domain_coordinate iglobal( std::vector<index_int>{ 3*mytid+i } );
    CHECK( d1->contains_element(mycoord,iglobal) );
    CHECK( d1->find_index(iglobal)==mytid );
  }

  CHECK_NOTHROW( d2 = new product_distribution(decomp,&pfunc2,nlocal ) );
  CHECK( d2->volume(mycoord)==nlocal );
  for (int i=0; i<nlocal; i++) {
    // proc 0,1 have same data, likewise 2,3, 4,5
    domain_coordinate iglobal( std::vector<index_int>{ 3*(mytid/2)+i } );
    CHECK( d2->contains_element(mycoord,iglobal) );
    CHECK( d2->find_index(iglobal,mytid)==mytid );
    CHECK( d2->find_index(iglobal)==2*(mytid/2) ); // the first proc with my data is 2*(p/2)
  }
}

TEST_CASE( "Masked distributions","[distribution][mask][70]" ) {
  if( mpi_nprocs<2 ) { printf("masking requires two procs\n"); return; }

  index_int localsize = 5;
  processor_mask *mask;

  SECTION( "create mask by adding" ) { 
    REQUIRE_NOTHROW( mask = new processor_mask() );
    REQUIRE_NOTHROW( mask->add(0) );
  }
  SECTION( "create mask by subtracting" ) {
    REQUIRE_NOTHROW( mask = new processor_mask(mpi_nprocs) );
    for (int p=1; p<mpi_nprocs; p++)
      REQUIRE_NOTHROW( mask->remove(p) );
  }

  SECTION( "SPMD mode" ) {
    REQUIRE_NOTHROW( mask = new processor_mask() );
    if (mytid==0)
      REQUIRE_NOTHROW( mask->add(mytid) );
  }

  SECTION( "SPMD mode, subtractive" ) {
    REQUIRE_NOTHROW( mask = new processor_mask(mpi_nprocs) );
    if (mytid>0)
      REQUIRE_NOTHROW( mask->remove(mytid) );
  }

  product_distribution
    *block = new product_block_distribution(decomp,localsize,-1),
    *masked_block = new product_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto 
    whole_vector = std::shared_ptr<object>( new product_object(block) ),
    masked_vector = std::shared_ptr<object>( new product_object(masked_block) );
  double *data;
  CHECK_NOTHROW( data = whole_vector->get_data(mytid) );
  CHECK( block->lives_on(mytid) );
  if (mytid==0) {
    CHECK( masked_block->lives_on(mytid) );
    REQUIRE_NOTHROW( data = masked_vector->get_data(mytid) );
  } else {
    CHECK( !masked_block->lives_on(mytid) );
    REQUIRE_THROWS( data = masked_vector->get_data(mytid) );
  }

  double *indata,*outdata;
  REQUIRE_NOTHROW( indata = whole_vector->get_data(mytid) );
  for (index_int i=0; i<localsize; i++) indata[i] = 1.;
  if (masked_vector->lives_on(mytid)) {
    REQUIRE_NOTHROW( outdata = masked_vector->get_data(mytid) );
    CHECK( outdata!=nullptr );
    for (index_int i=0; i<localsize; i++) outdata[i] = 2.;
  } else {
    REQUIRE_THROWS( outdata = masked_vector->get_data(mytid) );
  }
  product_kernel *copy = new product_kernel(whole_vector,masked_vector);
  copy->last_dependency()->set_type_local();
  copy->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy->analyze_dependencies() );
  REQUIRE_NOTHROW( copy->execute() );

  if (mytid==0) {
    for (index_int i=0; i<localsize; i++) 
      CHECK( outdata[i] == 1. );
  }
}

TEST_CASE( "Two masks","[distribution][mask][71]" ) {

  if( mpi_nprocs<4 ) { printf("test 71 needs 4 procs\n"); return; }

  INFO( "mytid=" << mytid );
  index_int localsize = 5;
  processor_mask *mask1,*mask2;

  mask1 = new processor_mask();
  mask2 = new processor_mask();
  for (int tid=0; tid<mpi_nprocs; tid+=2)
    REQUIRE_NOTHROW( mask1->add(tid) );
  for (int tid=0; tid<mpi_nprocs; tid+=4)
    REQUIRE_NOTHROW( mask2->add(tid) );

  product_distribution
    *block = new product_block_distribution(decomp,localsize,-1),
    *masked_block1 = new product_distribution( *block ),
    *masked_block2 = new product_distribution( *block );
  REQUIRE_NOTHROW( masked_block1->add_mask(mask1) );
  REQUIRE_NOTHROW( masked_block2->add_mask(mask2) );
  auto
    whole_vector = std::shared_ptr<object>( new product_object(block) ),
    masked_vector1 = std::shared_ptr<object>( new product_object(masked_block1) ),
    masked_vector2 = std::shared_ptr<object>( new product_object(masked_block2) );
  product_kernel
    *copy1 = new product_kernel(whole_vector,masked_vector1),
    *copy2 = new product_kernel(masked_vector1,masked_vector2);

  {
    double *data;
    REQUIRE_NOTHROW( data = whole_vector->get_data(mytid) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 1;
  }
  if (masked_vector1->lives_on(mytid)) {
    double *data;
    REQUIRE_NOTHROW( data = masked_vector1->get_data(mytid) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 2;
  }
  if (masked_vector2->lives_on(mytid)) {
    double *data;
    REQUIRE_NOTHROW( data = masked_vector2->get_data(mytid) );
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
      REQUIRE_NOTHROW( data = masked_vector2->get_data(mytid) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data[i] == 1. );
    } else if (mytid%2==0) {
      REQUIRE_NOTHROW( data = masked_vector1->get_data(mytid) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data[i] == 1. );
    }
  }
}

#endif
