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
 **** unit tests for the sparse matrix package
 **** (most tests do not actually rely on MPI)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"

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

TEST_CASE( "sparse inprod","[4]" ) { int nlocal = 10;
  sparse_row *r;
  REQUIRE_NOTHROW( r = new sparse_row() );
  for (index_int i=0; i<nlocal; i+=2 )
    r->add_element( nlocal*mytid+i,i+2 );
  distribution *d = new mpi_block_distribution(decomp,nlocal,-1);
  auto o = std::shared_ptr<object>( new mpi_object(d) );
  REQUIRE_NOTHROW( o->allocate() );
  double *data = o->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++ )
    data[i] = mytid+1;
  double inprod;
  REQUIRE_NOTHROW( inprod = r->inprod(o,mycoord) );
  int n2 = nlocal/2;
  CHECK( inprod==(mytid+1)*n2*(n2+1) );
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
  INFO( "columns: " << i->as_string() );
  CHECK( i->local_size()==3 );
  CHECK( i->contains_element(2) );
  CHECK( i->contains_element(3) );
  CHECK( i->contains_element(5) );
}

TEST_CASE( "index pattern collapsing","[pattern][11]" ) {
  // should this test be moved to unittest_sparse?
  index_int localsize=SMALLBLOCKSIZE+2,gsize=ntids*localsize;
  CHECK( decomp->get_dimensionality()==1 );
  mpi_distribution *d; REQUIRE_NOTHROW( d = new mpi_block_distribution(decomp,localsize,-1) );
  mpi_sparse_matrix *p; REQUIRE_NOTHROW( p = new mpi_sparse_matrix(d) );
  std::shared_ptr<indexstruct> columns;
  indexstruct *empty = new empty_indexstruct();
  int shift;

  empty->debug_on();
  SECTION( "shift 1 makes contiguous" ) {
    shift = 1;
    for (index_int i=d->first_index_r(mycoord)[0]; i<=d->last_index_r(mycoord)[0]; i++) {
      INFO( "row " << i );
      CHECK_NOTHROW( p->add_element(i,i) );
      if (i+shift<gsize) {
	CHECK_NOTHROW( p->add_element(i,i+shift) );
      } else {
	CHECK_THROWS( p->add_element(i,i+shift) );
      }
    }

    REQUIRE_NOTHROW( columns = p->all_columns_from(d->get_processor_structure(mycoord)) );
    INFO( "matrix column: " << columns->as_string() );
    CHECK( columns->is_contiguous() );
    CHECK( columns->first_index()==d->first_index_r(mycoord)[0] );
    if (mytid==ntids-1) {
      INFO( "last proc" );
      CHECK( columns->last_index()==d->last_index_r(mycoord)[0] );
      CHECK( columns->local_size()==localsize );
    } else {
      INFO( "arbitrary proc" );
      CHECK( columns->last_index()==(d->last_index_r(mycoord)[0]+shift) );
      CHECK( columns->local_size()==(localsize+1) );
    }
  }

  SECTION( "shift 2 makes gap" ) {
    shift = 2;
    for (index_int i=d->first_index_r(mycoord)[0]; i<=d->last_index_r(mycoord)[0]; i++) {
      INFO( "row " << i );
      CHECK_NOTHROW( p->add_element(i,i) );
      if (i+shift<gsize) {
	CHECK_NOTHROW( p->add_element(i,i+shift) );
      } else {
	CHECK_THROWS( p->add_element(i,i+shift) );
      }
    }

    REQUIRE_NOTHROW( columns = p->all_columns_from(d->get_processor_structure(mycoord)) );
    CHECK( columns->first_index()==d->first_index_r(mycoord)[0] );
    if (mytid==ntids-1) {
      INFO( "last proc" );
      CHECK( columns->last_index()==d->last_index_r(mycoord)[0] );
      CHECK( columns->local_size()==localsize );
    } else {
      INFO( "arbitrary proc" );
      CHECK( columns->last_index()==(d->last_index_r(mycoord)[0]+shift) );
      CHECK( columns->local_size()==(localsize+shift) );
    }
  }
  empty->debug_off();
}

TEST_CASE( "Sparse matrix tests","[sparse][12]" ) {
  INFO( "mytid: " << mytid );
  int nlocal = 10;
  sparse_matrix *pat; sparse_matrix *mat;

  SECTION( "pattern from indexstruct" ) {
    indexstruct *idx;
    REQUIRE_NOTHROW( idx = new contiguous_indexstruct(0,nlocal-1) );
    REQUIRE_NOTHROW( pat = new sparse_matrix( idx ) );
    REQUIRE_THROWS ( pat->add_element(nlocal,2) );
    REQUIRE_NOTHROW( pat->add_element(0,0) );
    REQUIRE_NOTHROW( pat->add_element(0,3) );
    REQUIRE_NOTHROW( pat->add_element(0,1) );
    REQUIRE_THROWS( pat->has_element(-1,0) );
    REQUIRE_THROWS( pat->has_element(nlocal,0) );
    REQUIRE_NOTHROW( pat->has_element(0,2*nlocal) );
    CHECK( pat->has_element(0,0) );
    CHECK( pat->has_element(0,1) );
    CHECK( !pat->has_element(0,2) );
    CHECK( pat->has_element(0,3) );
    CHECK( !pat->has_element(0,4) );
    REQUIRE_NOTHROW( pat->add_element(nlocal-1,nlocal-1) );
  }
  SECTION( "zero-based set of rows" ) {
    parallel_indexstruct *idx;
    REQUIRE_NOTHROW( idx = new parallel_indexstruct( ntids ) );
    REQUIRE_NOTHROW( idx->create_from_uniform_local_size(nlocal) );
    REQUIRE_NOTHROW( pat = new sparse_matrix( idx,mytid ) );
    index_int f,l;
    REQUIRE_NOTHROW( f = idx->first_index(mytid) );
    REQUIRE_NOTHROW( l = idx->last_index(mytid) );
    if (mytid!=1)
      REQUIRE_THROWS ( pat->add_element(nlocal,2) );
    else
      REQUIRE_NOTHROW ( pat->add_element(nlocal,2) );
    if (mytid==0)
      REQUIRE_NOTHROW( pat->add_element(0,0) );
    else
      REQUIRE_THROWS( pat->add_element(0,0) );
  }
  SECTION( "sparse matrix from indexstruct" ) {
    indexstruct *idx; double v;
    REQUIRE_NOTHROW( idx = new contiguous_indexstruct(0,nlocal-1) );
    REQUIRE_NOTHROW( mat = new sparse_matrix( idx ) );
    REQUIRE_THROWS ( mat->add_element(nlocal,2,2.) );
    REQUIRE_NOTHROW( mat->add_element(0,0,0.) );
    REQUIRE_NOTHROW( mat->add_element(0,3,3.) );
    REQUIRE_NOTHROW( mat->add_element(0,1,1.) );
    REQUIRE_THROWS( mat->has_element(-1,0) );
    REQUIRE_THROWS( mat->has_element(nlocal,0) );
    REQUIRE_NOTHROW( mat->has_element(0,2*nlocal) );

    CHECK( mat->has_element(0,0) );
    CHECK( mat->has_element(0,1) );
    // REQUIRE_NOTHROW( v = mat->get_element(0,1) );
    // CHECK( v==Approx(1.) );
    CHECK( !mat->has_element(0,2) );

    CHECK( mat->has_element(0,3) );
    // REQUIRE_NOTHROW( v = mat->get_element(0,3) );
    // CHECK( v==Approx(3.) );

    CHECK( !mat->has_element(0,4) );
    REQUIRE_NOTHROW( mat->add_element(nlocal-1,nlocal-1,nlocal-1.) );
  }
}

TEST_CASE( "sparse matprod","[20]" ) { int nlocal = 10;
  sparse_matrix *m;
  REQUIRE_NOTHROW( m = new sparse_matrix() );
  // matrix with just one row on each processor
  for (index_int i=0; i<nlocal; i+=2 )
    m->add_element( mytid,nlocal*mytid+i,i+2 );

  distribution *din = new mpi_block_distribution(decomp,nlocal,-1),
    *dout = new mpi_block_distribution(decomp,1,-1);
  auto in = std::shared_ptr<object>( new mpi_object(din) ),
    out = std::shared_ptr<object>( new mpi_object(dout) );
  in->allocate(); double *data = in->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++ )
    data[i] = mytid+1;

  REQUIRE_NOTHROW( m->multiply(in,out,mycoord) );
  double *val = out->get_data(mycoord);
  int n2 = nlocal/2;
  CHECK( val[0]==Approx( (mytid+1)*n2*(n2+1) ) );
}

