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
 **** unit tests for processor treatment
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"
#include "mpi_ops.h"

TEST_CASE( "basic processor stuff","[proc][01]" ) {

  processor_coordinate deflt;
  CHECK( deflt.get_dimensionality()==0 );

  processor_coordinate fv(5);
  CHECK( fv.get_dimensionality()==5 );

  processor_coordinate one( std::vector<int>{1,2,3} );
  CHECK( one.get_dimensionality()==3 );

  processor_coordinate two;
  REQUIRE_NOTHROW( two=one+1 );
  CHECK( two[0]==2 );
  CHECK( two[1]==3 );
  CHECK( two[2]==4 );
  REQUIRE_NOTHROW( two=two-1 );
  CHECK( two==one );
}

TEST_CASE( "multi-d decompositions","[multi][proc][03]" ) {
  int dim; const char *path;
  SECTION( "1-d" ) { path = "1";
    dim = 1;
  }
  SECTION( "2-d" ) { path = "2";
    dim = 2;
  }
  SECTION( "3-d" ) { path = "3";
    dim = 3;
  }
  INFO( fmt::format("testing in {}-d",dim) );
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,layout) );

  processor_coordinate zero,one;
  REQUIRE_NOTHROW( zero = mdecomp->get_origin_processor() );
  INFO( fmt::format("origin: {}",zero.as_string()) );
  for (int id=0; id<dim; id++)
    CHECK( zero[id]==0 );

  REQUIRE_NOTHROW( one = mdecomp->get_farpoint_processor() );
  INFO( fmt::format("farpoint: {}",one.as_string()) );
  int derive;
  switch (dim) {
  case 1:
    derive = one[0];
    break;
  case 2:
    derive = one[0]*layout->coord(1)+one[1];
    break;
  case 3:
    derive = one[0]*layout->coord(1)*layout->coord(2) + one[1]*layout->coord(2) + one[2];
    break;
  }
  CHECK( derive==ntids-1 );

  processor_coordinate mycoord;
  REQUIRE_NOTHROW( mycoord = mdecomp->coordinate_from_linear(mytid) );
  for (int id=0; id<dim; id++) {
    CHECK( mycoord.coord(id)>=0 );
    CHECK( mycoord.coord(id)<=layout->coord(id) );
  }
  switch (dim) {
  case 1:
    derive = mycoord.coord(0);
    break;
  case 2:
    derive = mycoord.coord(0)*layout->coord(1)+mycoord.coord(1);
    break;
  case 3:
    derive = mycoord.coord(0)*layout->coord(1)*layout->coord(2) + mycoord.coord(1)*layout->coord(2) + mycoord.coord(2);
    break;
  }
  CHECK( derive==mytid );

}

TEST_CASE( "iterate over processors, 1d","[proc][04]" ) {
  REQUIRE( decomp->get_dimensionality()==1 );
  int count=0;
  for ( auto p=decomp->begin(); p!=decomp->end(); ++p) {
    CHECK( (*p).at(0)==count );
    count++;
  }
  CHECK( count==ntids );
}

TEST_CASE( "iterate over processors, multi-d","[proc][05]" ) {
  int dim=2;
  processor_coordinate *layout;
  REQUIRE_NOTHROW( layout = arch->get_proc_layout(dim) );
  decomposition *mdecomp;
  REQUIRE_NOTHROW( mdecomp = new mpi_decomposition(arch,layout) );
  int np1,np0;
  REQUIRE_NOTHROW( np0 = layout->at(0) );
  REQUIRE_NOTHROW( np1 = layout->at(1) );
  CHECK( ntids==np0*np1 );
  INFO( fmt::format("Proc grid: {} x {}",np0,np1) );

  REQUIRE( mdecomp->get_dimensionality()==dim );
  int count=0;
  for ( auto p=mdecomp->begin(); p!=mdecomp->end(); ++p) {
    INFO( fmt::format("count={}, proc={}",count,(*p).as_string()) );
    CHECK( (*p).at(0)==count/np1 );
    CHECK( (*p).at(1)==count%np1 );
    count++;
  }
  CHECK( count==ntids );
}
