/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the MPI+OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for operating on parallel structures
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "product_base.h"
#include "product_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Elementary ioperator stuff","[operate][01]" ) {
  ioperator iop;
  CHECK_THROWS( iop = ioperator("no_such_thing") );
}

TEST_CASE( "Test ioperator right workings modulo","[operate][02][modulo]" ) {
  ioperator i1 = ioperator(">>1");
  CHECK( i1.is_shift_op() );
  CHECK( i1.is_modulo_op() );
  CHECK( i1.amount()==1 );
  CHECK( i1.operate(0)==1 );
  CHECK_NOTHROW( i1.operate(-1) );
  CHECK( i1.operate(5)==6 );
  CHECK( i1.is_shift_op() );
  CHECK( i1.amount()==1 );
  CHECK( i1.is_modulo_op() );
  CHECK( i1.operate(5,6)==6 );
  CHECK( i1.operate(6,6)==0 );
}

TEST_CASE( "Test ioperator right workings","[operate][03]" ) {
  ioperator i2 = ioperator(">=1");
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.operate(0)==1 );
  CHECK_NOTHROW( i2.operate(-1) );
  CHECK( i2.operate(15)==16 );
  CHECK( i2.operate(15,15)==15 );
}

TEST_CASE( "Test ioperator left workings modulo","[operate][04][modulo]" ) {
  ioperator i1 = ioperator("<<1");
  CHECK( i1.is_modulo_op() );
  CHECK( i1.operate(1)==0 );
  CHECK_NOTHROW( i1.operate(-1) );
  //  CHECK_THROWS( i1.operate(0) );
  CHECK( i1.operate(6)==5 );
  CHECK( i1.operate(0,6)==6 );
}

TEST_CASE( "Test ioperator left workings bump","[operate][05]" ) {
  ioperator i2 = ioperator("<=1");
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.operate(0)==-1 );
  CHECK_NOTHROW( i2.operate(-1) );
  CHECK( i2.operate(16)==15 );
  CHECK( i2.operate(15,16)==14 );
}

TEST_CASE( "Test ioperator shift workings modulo","[operate][06][modulo]" ) {
  ioperator i1 = ioperator(">>1");
  CHECK( i1.is_modulo_op() );
  CHECK( i1.inverse_operate(1)==0 );
  CHECK_THROWS( i1.inverse_operate(-1) );
  CHECK( i1.inverse_operate(6)==5 );
  CHECK_THROWS( i1.inverse_operate(0) );
  CHECK( i1.inverse_operate(0,6)==5 );

  ioperator i3 = ioperator("<<1");
  CHECK( i3.is_modulo_op() );
  CHECK( i3.operate(1)==0 );
  CHECK_NOTHROW( i3.operate(-1) );
  CHECK_NOTHROW( i3.operate(0) );
  CHECK( i3.operate(6)==5 );
  CHECK( i3.operate(0,6)==6 );

}

TEST_CASE( "Test ioperator shift workings","[operate][07]" ) {
  ioperator i2 = ioperator(">=1");
  CHECK( !i2.is_modulo_op() );
  CHECK( i2.inverse_operate(1)==0 );
  CHECK_THROWS( i2.inverse_operate(0) );
  CHECK_THROWS( i2.inverse_operate(-1) );
  CHECK( i2.inverse_operate(16)==15 );
  CHECK( i2.inverse_operate(15,16)==14 );

  ioperator i4 = ioperator("<=1");
  CHECK( !i4.is_modulo_op() );
  CHECK( i4.operate(0)==-1 );
  CHECK( i4.operate(0,5)==0 );
  CHECK_NOTHROW( i4.operate(-1) );
  CHECK( i4.operate(16)==15 );
  CHECK( i4.operate(15,16)==14 );
}

// TEST_CASE( "Test ioperator restrict/interpolate workings","[operate][08]" ) {

//   ioperator i1 = ioperator("*2");
//   CHECK( i1.is_restrict_op() );
//   CHECK( !i1.is_prolongate_op() );
//   CHECK( i1.operate(3)==6 );

//   ioperator i2 = ioperator("/2");
//   CHECK( !i2.is_restrict_op() );
//   CHECK( i2.is_prolongate_op() );
//   CHECK( i2.operate(8)==4 );
// }


TEST_CASE( "copy parallel indexstruct","[indexstruct][copy][51]" ) {

  int localsize = 12,gsize = mpi_nprocs*localsize;
  parallel_indexstruct *pstr,*qstr;
  REQUIRE_NOTHROW( pstr = new parallel_indexstruct( mpi_nprocs ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize ) );
  CHECK( pstr->local_size(mytid)==localsize );
  
  REQUIRE_NOTHROW( qstr = new parallel_indexstruct( *pstr ) );
  //  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize+mpi_nprocs ) );
  //  CHECK( pstr->local_size(mytid)==localsize+1 );
  CHECK( qstr->local_size(mytid)==localsize );
}

TEST_CASE( "shift parallel indexstruct","[indexstruct][operate][shift][52]" ) {
  
  int localsize = 12,gsize = mpi_nprocs*localsize;
  std::shared_ptr<parallel_indexstruct> pstr,qstr;
  REQUIRE_NOTHROW( pstr = std::shared_ptr<parallel_indexstruct> ( new parallel_indexstruct( mpi_nprocs ) ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize ) );
  CHECK( pstr->local_size(mytid)==localsize );
  
  REQUIRE_NOTHROW( qstr = pstr->operate( ioperator(">>1") ) );
  CHECK( pstr->local_size(mytid)==localsize );
  CHECK( qstr->local_size(mytid)==localsize );
  CHECK( pstr->first_index(mytid)==(mytid*localsize) );
  CHECK( qstr->first_index(mytid)==(mytid*localsize+1) );

}

TEST_CASE( "divide parallel indexstruct","[indexstruct][operate][divide][53]" ) {
  int localsize = 15,gsize = mpi_nprocs*localsize;
  std::shared_ptr<parallel_indexstruct> pstr,qstr;
  REQUIRE_NOTHROW( pstr = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( mpi_nprocs ) ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( 2*gsize ) );

  REQUIRE_NOTHROW( qstr = pstr->operate( ioperator("/2") ) );
  CHECK( pstr->local_size(mytid)==(2*localsize) );
  CHECK( qstr->local_size(mytid)==localsize );
  CHECK( pstr->first_index(mytid)==(2*mytid*localsize) );
  CHECK( qstr->first_index(mytid)==(mytid*localsize) );
}

TEST_CASE( "copy and operate distributions","[distribution][operate][copy][54]" ) {
  INFO( "mytid=" << mytid );
  int localsize = 10;
  distribution // VLE don't like this
    *d1 = new product_block_distribution(decomp,localsize,-1),
    *d2;

  // SECTION( "plain copy" ) {
  //   REQUIRE_NOTHROW( d2 = new product_distribution( d1 ) );
  //   index_int s1,s2;
  //   REQUIRE_NOTHROW( s1 = d1->volume(mycoord) );
  //   REQUIRE_NOTHROW( s2 = d2->volume(mycoord) );
  //   CHECK( s1==s2 );
  // }

  SECTION( "operated copy" ) {
    ioperator is = ioperator(">>1");
    parallel_structure *ps,*po;
    REQUIRE_NOTHROW( ps = dynamic_cast<parallel_structure*>(d1) );
    REQUIRE_NOTHROW( po = ps->operate(is) );

    // identical
    REQUIRE_NOTHROW( d2 = new product_distribution(ps ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==d1->first_index_r(mycoord) );

    // shifted from index structure
    REQUIRE_NOTHROW( d2 = new product_distribution(po ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)+1) );

    // shifted by operate
    REQUIRE_NOTHROW( d2 = d1->operate(is) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)+1) );
  }

  SECTION( "operate the other way" ) { // VLE when do we wrap around?
    REQUIRE_NOTHROW( d2 = d1->operate( ioperator("<<1") ) );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)-1) );
  }

  SECTION( "divide operate" ) {
    ioperator div2 = ioperator("/2");
    REQUIRE_NOTHROW( d2 = d1->operate(div2) );
    CHECK( d2->volume(mycoord)==(d1->volume(mycoord)/2) );
    CHECK( d2->first_index_r(mycoord)==(d1->first_index_r(mycoord)/2) );
  }

  SECTION( "multiply" ) {
    ioperator mul2 = ioperator("*2");
    REQUIRE_NOTHROW( d2 = d1->operate(mul2) );
    CHECK( d2->first_index_r(mycoord)==d1->first_index_r(mycoord)*2 );
    CHECK( d2->last_index_r(mycoord)==d1->last_index_r(mycoord)*2 );
    CHECK( d2->volume(mycoord)==d1->volume(mycoord) );
    CHECK( (d2->last_index_r(mycoord)-d2->first_index_r(mycoord))==
	   (d1->last_index_r(mycoord)-d1->first_index_r(mycoord))*2 );
  }
}

TEST_CASE( "merge parallel indexstruct","[indexstruct][merge][55]" ) {

  int localsize = 11,gsize = mpi_nprocs*localsize;
  std::shared_ptr<parallel_indexstruct> pstr,qstr,zstr;
  REQUIRE_NOTHROW( pstr = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( mpi_nprocs ) ) );
  REQUIRE_NOTHROW( pstr->create_from_global_size( gsize ) );
  CHECK( pstr->local_size(mytid)==localsize );
  
  SECTION( "shift right" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",1) ) );
    CHECK( pstr->local_size(mytid)==localsize );
    CHECK( qstr->local_size(mytid)==localsize );
    CHECK( pstr->first_index(mytid)==(mytid*localsize) );
    CHECK( qstr->first_index(mytid)==(mytid*localsize+1) );

    REQUIRE_NOTHROW( zstr = pstr->struct_union( qstr ) );
    CHECK( zstr->local_size(mytid)==localsize+1 );
    CHECK( zstr->first_index(mytid)==(mytid*localsize) );
  }

  SECTION( "shift left" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",-1) ) );
    CHECK( pstr->local_size(mytid)==localsize );
    CHECK( qstr->local_size(mytid)==localsize );
    CHECK( pstr->first_index(mytid)==(mytid*localsize) );
    CHECK( qstr->first_index(mytid)==(mytid*localsize-1) );

    REQUIRE_NOTHROW( zstr = pstr->struct_union( qstr ) );
    CHECK( zstr->local_size(mytid)==localsize+1 );
    CHECK( zstr->first_index(mytid)==(mytid*localsize-1) );
  }
}

TEST_CASE( "merge distributions","[indexstruct][merge][56]" ) {

  int localsize = 11,gsize = mpi_nprocs*localsize;
  distribution *pstr,*qstr,*zstr;
  REQUIRE_NOTHROW( pstr = new product_block_distribution(decomp,-1,gsize) );
  CHECK( pstr->volume(mycoord)==localsize );
  
  SECTION( "shift right" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",1) ) );
    CHECK( pstr->volume(mycoord)==localsize );
    CHECK( qstr->volume(mycoord)==localsize );
    CHECK( pstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize}) );
    CHECK( qstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize+1}) );

    REQUIRE_NOTHROW( zstr = pstr->distr_union( qstr ) );
    CHECK( zstr->volume(mycoord)==localsize+1 );
    CHECK( zstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize}) );
  }

  SECTION( "shift left" ) {
    REQUIRE_NOTHROW( qstr = pstr->operate_base( ioperator("shift",-1) ) );
    CHECK( pstr->volume(mycoord)==localsize );
    CHECK( qstr->volume(mycoord)==localsize );
    CHECK( pstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize}) );
    CHECK( qstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize-1}) );

    REQUIRE_NOTHROW( zstr = pstr->distr_union( qstr ) );
    CHECK( zstr->volume(mycoord)==localsize+1 );
    CHECK( zstr->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*localsize-1}) );
  }
}

TEST_CASE( "divide distributions","[distribution][operate][57]" ) {

  if (mpi_nprocs%2!=0) { printf("[57] Need an even number of processors\n"); return; }

  INFO( "mytid=" << mytid );

  product_distribution
    *twoper = new product_block_distribution(decomp,2*omp_nprocs,-1);
  distribution
    *oneper,*duplic;

  REQUIRE_NOTHROW( oneper = twoper->operate( ioperator("/2") ) );
  CHECK( oneper->volume(mycoord)==omp_nprocs );
  CHECK( oneper->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*omp_nprocs}) );

  REQUIRE_NOTHROW( duplic = oneper->operate( ioperator("/2") ) );
  CHECK( duplic->volume(mycoord)==omp_nprocs/2 );
  CHECK( duplic->first_index_r(mycoord)==domain_coordinate(std::vector<index_int>{mytid*omp_nprocs/2}) );
}

// TEST_CASE( "operate 2d distributions","[distribution][grid][operate][copy][2d][70][hide]" ) {
//   index_int nlocal=3;
//   product_distribution *d1;
//   distribution *d2;

//   REQUIRE( (mpi_nprocs%2)==0 ); // sorry, need even number
//   product_environment *gridenv;
//   REQUIRE_NOTHROW( gridenv = new product_environment( *env ) );
//   REQUIRE_NOTHROW( gridenv->set_grid_2d( mpi_nprocs/2,2 ) );
//   CHECK_NOTHROW( d1 = new product_block_distribution
// 	     ( gridenv,  nlocal,nlocal,-1,-1 ) );

//   SECTION( "operated copy" ) {
//     gridoperator *is = new gridoperator(">>1",">>1");
//     std::shared_ptr<parallel_indexstruct> ps,po;
//     index_int s;

//     // shifted by operate
//     REQUIRE_NOTHROW( d2 = d1->operate(is) );
//     REQUIRE_NOTHROW( s = d2->volume(mycoord) );
//     CHECK( s==nlocal*nlocal );

//     REQUIRE_NOTHROW( delete is );
//   }

//   delete gridenv;
// }
