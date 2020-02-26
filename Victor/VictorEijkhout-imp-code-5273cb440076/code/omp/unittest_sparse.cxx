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
 **** unit tests for the sparse matrix package
 **** (most tests do not actually rely on OMP)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_static_vars.h"

TEST_CASE( "element","[element][1]" ) {
  sparse_element *e;
  REQUIRE_NOTHROW( e = new sparse_element(3,5.41) );
  CHECK( e->get_index()==3 );
  CHECK( e->get_value()==Approx(5.41) );

  sparse_element *f;
  REQUIRE_NOTHROW( f = new sparse_element(2,7.3) );
  CHECK( *f<*e );
  CHECK( !(*e<*f) );
}

TEST_CASE( "row","[2]" ) {
  sparse_row *r;
  REQUIRE_NOTHROW( r = new sparse_row() );
  SECTION( "regular" ) {
    REQUIRE_NOTHROW( r->add_element(1,8.5) );
    REQUIRE_NOTHROW( r->add_element(2,7.5) );
  }
  SECTION( "reverse" ) {
    REQUIRE_NOTHROW( r->add_element(2,7.5) );
    REQUIRE_NOTHROW( r->add_element(1,8.5) );
  }
  CHECK( r->size()==2 );
  CHECK( r->at(0) < r->at(1) );
  CHECK( r->row_sum()==Approx(16.) );

  std::shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r->all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(1) );
  CHECK( i->contains_element(2) );
}

TEST_CASE( "irow","[3]" ) {
  sparse_rowi *r;
  REQUIRE_NOTHROW( r = new sparse_rowi(5) );
  SECTION( "regular" ) {
    REQUIRE_NOTHROW( r->add_element(11,8.5) );
    REQUIRE_NOTHROW( r->add_element(12,7.5) );
  }
  SECTION( "reverse" ) {
    REQUIRE_NOTHROW( r->add_element(12,7.5) );
    REQUIRE_NOTHROW( r->add_element(11,8.5) );
  }
  CHECK( r->size()==2 );
  CHECK( r->at(0) < r->at(1) );
  CHECK( r->row_sum()==Approx(16.) );

  std::shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = r->all_indices() );
  CHECK( i->local_size()==2 );
  CHECK( i->contains_element(11) );
  CHECK( i->contains_element(12) );
}

TEST_CASE( "sparse inprod","[4]" ) {
  int nrows = 10;
  sparse_row *r;
  distribution *d = new omp_block_distribution(decomp,nrows);
  auto vector = std::shared_ptr<object>( new omp_object(d) );
  vector->set_name("unit4vector");
  REQUIRE_NOTHROW( vector->allocate() );
  REQUIRE_NOTHROW( vector->has_type_blocked() );

  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "thread: " << mytid );
    REQUIRE_NOTHROW( r = new sparse_row() );
    for (index_int i=0; i<nrows; i+=2 )
      r->add_element( i,i+2 );

    double *data; // use get_raw_data?
    REQUIRE_NOTHROW( data = vector->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<nrows; i++ )
      data[i] = mytid+1;
    double inprod;

    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( inprod = r->inprod(vector,mycoord) );
    int n2 = nrows/2;
    CHECK( inprod==(mytid+1)*n2*(n2+1) );
  }
}

TEST_CASE( "matrix","[10]" ) {
  sparse_matrix *m;
  REQUIRE_NOTHROW( m = new sparse_matrix() );
  CHECK( m->local_size()==0 );

  REQUIRE_NOTHROW( m->add_element(1,2,3.) );
  REQUIRE_NOTHROW( m->add_element(3,5,8.) );
  REQUIRE_NOTHROW( m->add_element(3,3,7.) );
  REQUIRE_NOTHROW( m->add_element(7,3,9.) );
  
  CHECK( m->nnzeros()==4 );
  CHECK( m->local_size()==3 );
  CHECK( m->has_element(3,5) );
  CHECK( !m->has_element(3,4) );

  std::shared_ptr<indexstruct> idx;
  REQUIRE_NOTHROW( idx = m->row_indices() );
  CHECK( idx->local_size()==3 );
  CHECK( idx->first_index()==1 );
  CHECK( idx->last_index()==7 );

  int s;
  REQUIRE_NOTHROW( s = m->row_sum(1) );
  CHECK( s==Approx(3.) );
  REQUIRE_THROWS( s = m->row_sum(2) );
  REQUIRE_NOTHROW( s = m->row_sum(3) );
  CHECK( s==Approx(15.) );
  REQUIRE_THROWS( s = m->row_sum(4) );

  std::shared_ptr<indexstruct> i;
  REQUIRE_NOTHROW( i = m->all_columns() );
  CHECK( i->local_size()==3 );
  CHECK( i->contains_element(2) );
  CHECK( i->contains_element(3) );
  CHECK( i->contains_element(5) );
}

TEST_CASE( "sparse matprod","[11]" ) { int nlocal = 10;
  sparse_matrix *m;
  distribution *din = new omp_block_distribution(decomp,nlocal,-1),
    *dout = new omp_block_distribution(decomp,1,-1);

  const char *path;
  SECTION( "bare matrix" ) { path = "direct";
    REQUIRE_NOTHROW( m = new sparse_matrix() );
  }
  SECTION( "omp matrix class" ) { path = "derived";
    REQUIRE_NOTHROW( m = new omp_sparse_matrix(dout,nlocal*ntids) );
  }
  INFO( path << " matrix creation" );
  // matrix with just one row on each processor
  for (int mytid=0; mytid<ntids; mytid++)
    for (index_int i=0; i<nlocal; i+=2 )
      m->add_element( mytid,nlocal*mytid+i,i+2 );

  auto in = std::shared_ptr<object>( new omp_object(din) ),
    out = std::shared_ptr<object>( new omp_object(dout) );
  in->allocate(); double *data = in->get_data(new processor_coordinate_zero(1));
  for (int mytid=0; mytid<ntids; mytid++)
    for (index_int i=0; i<nlocal; i++ )
      data[mytid*nlocal+i] = mytid+1;

  for (int mytid=0; mytid<ntids; mytid++) {
    INFO( "multiply on " << mytid );
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    REQUIRE_NOTHROW( m->multiply(in,out,mycoord) );
    double *val = out->get_data(new processor_coordinate_zero(1));
    int n2 = nlocal/2;
    CHECK( val[mytid]==Approx( (mytid+1)*n2*(n2+1) ) );
  }
}

