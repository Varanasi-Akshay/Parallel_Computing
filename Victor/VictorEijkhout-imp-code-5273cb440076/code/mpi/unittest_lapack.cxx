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
 **** unit tests for lapack-style operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"

TEST_CASE( "processor grids","[processor][1]" ) {
  architecture *aa = new mpi_architecture( *arch );
  CHECK_THROWS( aa->get_processor_coordinates() );
  SECTION( "1d" ) {
    REQUIRE_NOTHROW( aa->set_1d() );
    processor_coordinates *c;
    REQUIRE_NOTHROW( c = aa->get_processor_coordinates() );
    CHECK( c->dim()==1 );
    CHECK( c->coord(0)==mytid );
  }
  SECTION( "2d" ) {
    REQUIRE_NOTHROW( aa->set_2d() );
    processor_coordinates *c;
    REQUIRE_NOTHROW( c = aa->get_processor_coordinates() );
    CHECK( c->dim()==2 );
    if (ntids==4) {
      CHECK( c->size(0)==2 );
      CHECK( c->size(1)==2 );
      if ( mytid==0 ) {
	CHECK( c->coord(0)==0 );
	CHECK( c->coord(1)==0 );
      } else if (mytid==1) {
	CHECK( c->coord(0)==0 );
	CHECK( c->coord(1)==1 );
      } else if (mytid==2) {
	CHECK( c->coord(0)==1 );
	CHECK( c->coord(1)==0 );
      } else if (mytid==3) {
	CHECK( c->coord(0)==1 );
	CHECK( c->coord(1)==1 );
      }
    } else printf("lapack test [1] for 4 procs exactly only ");
  }
}

TEST_CASE( "pivot is done only once","[mask][10]" ) {
  distribution *unique = new mpi_block_distribution(arch,1,-1);
  CHECK( unique->global_size()==ntids );
  CHECK( unique->local_size(mytid)==1 );
  object *vec0 = new mpi_object(unique);

  processor_mask *mask0;
  REQUIRE_NOTHROW( mask0 =  new processor_mask(arch) );
  REQUIRE_NOTHROW( mask0->add(0) );

  distribution *unique_on0;
  REQUIRE_NOTHROW( unique_on0 = new mpi_distribution(unique) );
  {
    object *vec1;
    INFO( "can we create from copied distribution?" );
    REQUIRE_NOTHROW( vec1 = new mpi_object(unique_on0) );
  }
  REQUIRE_NOTHROW( unique_on0->add_mask(mask0) );
  object *vec1;
  REQUIRE_NOTHROW( vec1 = new mpi_object(unique_on0) );
  for (int tid=0; tid<ntids; tid++)
    if (tid==0)
      CHECK( vec1->lives_on(tid) );
    else
      CHECK( !vec1->lives_on(tid) );


  algorithm *queue = new mpi_algorithm(arch);
  {
    double one=1.,*data; vec0->set_value(&one);
    REQUIRE_NOTHROW( data = vec0->get_data(mytid) );
    CHECK( data[0]==Approx(one) );
  }
  REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(vec0) ) );

  SECTION( "test0" ) {
    kernel *k = new mpi_kernel(vec0,vec1);
    REQUIRE_NOTHROW( queue->add_kernel(k) );
  }
  SECTION( "test1" ) {
    if (ntids<2) return;
    object *vec2 = new mpi_object(unique);
    processor_mask *mask1;
    REQUIRE_NOTHROW( mask1 =  new processor_mask(arch) );
    REQUIRE_NOTHROW( mask1->add(1) );
  }

}
