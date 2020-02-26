/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the OMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** unit tests for nbody calculations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"
#include "imp_functions.h"
#include "omp_ops.h"

TEST_CASE( "distribution derivation","[1]" ) {
  
  int nlocal = 8,nglobal = nlocal*ntids;
  distribution *level_dist, *new_dist;
  REQUIRE_NOTHROW( level_dist = new omp_block_distribution(decomp,nglobal) );
  auto coarsen = ioperator(":2");

  auto distributions = new std::vector<distribution*>;
  distributions->push_back(level_dist);
  index_int g; REQUIRE_NOTHROW( g = level_dist->global_volume() );
  for (int level=0; ; level++) {
    INFO( "level: " << level );
    //printf("On level %d: %s\n",level,level_dist->as_string().data());
    REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      INFO( "mytid=" << mytid );
      if (mytid<ntids-1) {
	processor_coordinate nextcoord;
	REQUIRE_NOTHROW( nextcoord = decomp->coordinate_from_linear(mytid+1) );
	index_int
	  ml = new_dist->last_index_r(mycoord)[0],
	  nf = new_dist->first_index_r(nextcoord)[0];
	CHECK( ((ml==nf-1)||(ml==nf)) );
      }
    }
    distributions->push_back(new_dist);
    g /= 2;
    REQUIRE_NOTHROW( g==new_dist->global_volume() );
    if (g==1) break;
    level_dist = new_dist;
  }
}

TEST_CASE( "center of mass function","[2]" ) {

  int nlocal = 8,nglobal = nlocal*ntids;
  distribution *level_dist, *new_dist;
  REQUIRE_NOTHROW( level_dist = new omp_block_distribution(decomp,nglobal) );
  double *data;
  auto coarsen = ioperator(":2");

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  auto
    bot = std::shared_ptr<object>( new omp_object(level_dist) ),
    top = std::shared_ptr<object>( new omp_object(new_dist) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( data = bot->get_data(mycoord) );
    index_int first,lsize;
    first = bot->first_index_r(mycoord)[0]; lsize = bot->volume(mycoord);
    for (index_int i=first; i<first+lsize; i++)
      data[i] = first+i;
  }

  kernel *calculate_cm;
  SECTION( "in separate pieces" ) {
    REQUIRE_NOTHROW( calculate_cm = new omp_kernel(bot,top) );
    REQUIRE_NOTHROW( calculate_cm->set_localexecutefn( &scansum ) );
    REQUIRE_NOTHROW( calculate_cm->last_dependency()
		     ->set_signature_function_function
		     ( [] (index_int i) -> std::shared_ptr<indexstruct> {
		       return doubleinterval(i); } ) );
  }
  SECTION( "as one kernel" ) {
    REQUIRE_NOTHROW( calculate_cm = new omp_centerofmass_kernel(bot,top) );
  }

  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(bot) ) );
  REQUIRE_NOTHROW( queue->add_kernel( calculate_cm ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int first = top->first_index_r(mycoord)[0], lsize = top->volume(mycoord);
    REQUIRE_NOTHROW( data = top->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=first; i<first+lsize; i++) {
      INFO( "i: " << i );
      index_int g = 2*(i+first);
      CHECK( data[i]==g+g+1 );
    }
  }
}

TEST_CASE( "force prolongation","[3]" ) {

  int nlocal = 8,nglobal = nlocal*ntids;
  distribution *level_dist, *new_dist;
  REQUIRE_NOTHROW( level_dist = new omp_block_distribution(decomp,nglobal) );
  double *data;
  auto coarsen = ioperator(":2" );

  omp_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new omp_sparse_matrix(level_dist) );
  index_int g = level_dist->global_volume();
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      f = level_dist->first_index_r(mycoord)[0], l = level_dist->last_index_r(mycoord)[0];
    for (index_int row=f; row<=l; row++) {
      index_int col; double v;
      col = row; v = 0;
      REQUIRE_NOTHROW( mat->add_element(row,col,v) );
      col = row-1; v = 1.;
      if (col>=0) 
	REQUIRE_NOTHROW( mat->add_element(row,col,v) );
      col = row+1; v = 1.;
      if (col<g)
	REQUIRE_NOTHROW( mat->add_element(row,col,v) );
    }
  }

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    CHECK( new_dist->volume(mycoord)==level_dist->volume(mycoord)/2 );
  }

  auto
    bot = std::shared_ptr<object>( new omp_object(level_dist) ), 
    top = std::shared_ptr<object>( new omp_object(new_dist) ),
    side = std::shared_ptr<object>( new omp_object(level_dist) ),
    expanded = std::shared_ptr<object>( new omp_object(level_dist) ),
    multiplied = std::shared_ptr<object>( new omp_object(level_dist) );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int first,lsize;
    // fill in the half size top level
    first = top->first_index_r(mycoord)[0]; lsize = top->volume(mycoord);
    REQUIRE_NOTHROW( data = top->get_data(mycoord) );
    for (index_int i=first; i<first+lsize; i++)
      data[i] = i;
    // fill in the other half tree
    first = side->first_index_r(mycoord)[0]; lsize = side->volume(mycoord);
    REQUIRE_NOTHROW( data = side->get_data(mycoord) );
    for (index_int i=first; i<first+lsize; i++)
      data[i] = 1.;
  }

  // did we do this right?
  {
    double *data = top->get_data(new processor_coordinate_zero(1));
    for (index_int i=0; i<top->global_volume(); i++)
      CHECK( data[i]==i );
  }

  algorithm *queue = new omp_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(top) ) );
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(side) ) );

  kernel *expand,*multiply, *calculate_cm;
  const char *path;
  SECTION( "in pieces" ) {
    path = "separate kernels";
    REQUIRE_NOTHROW( expand = new omp_kernel(top,expanded) );
    REQUIRE_NOTHROW( expand->set_localexecutefn( &scanexpand ) );
    REQUIRE_NOTHROW( expand->last_dependency()->set_signature_function_function
		     ( [] (index_int i) -> std::shared_ptr<indexstruct> {
		       return halfinterval(i); } ) );
    REQUIRE_NOTHROW( queue->add_kernel(expand) );
    
    REQUIRE_NOTHROW( multiply = new omp_spmvp_kernel(side,multiplied,mat) );
    REQUIRE_NOTHROW( queue->add_kernel(multiply) );
    
    REQUIRE_NOTHROW( calculate_cm = new omp_sum_kernel(expanded,multiplied,bot) );

    REQUIRE_NOTHROW( queue->add_kernel(calculate_cm) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );

    double *expanded_data;
    REQUIRE_NOTHROW( expanded_data = expanded->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<expanded->global_volume(); i++) {
      INFO( "expanded i=" << i );
      CHECK( expanded_data[i] == i/2 );
    }    
}
  SECTION( "as one kernel" ) {
    path = "all in one";
    REQUIRE_NOTHROW( calculate_cm = new omp_sidewaysdown_kernel(top,side,bot,mat) );

    REQUIRE_NOTHROW( queue->add_kernel(calculate_cm) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );

  }
  INFO( "path: " << path );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( data = bot->get_data(mycoord) );
    index_int first,lsize;
    first = top->first_index_r(mycoord)[0]; lsize = top->volume(mycoord);
    for (index_int i=first; i<first+lsize; i++) {
      INFO( "ig: " << i );
      if (i==0 || i==g-1 )
	CHECK( data[i]==i/2+1 ); // divide by two because of the expand
      else
	CHECK( data[i]==i/2+2 );
    }
  }
}

