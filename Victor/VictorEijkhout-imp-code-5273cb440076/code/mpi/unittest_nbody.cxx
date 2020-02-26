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
 **** unit tests for nbody calculations
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

TEST_CASE( "distribution derivation","[1]" ) {
  INFO( "mytid=" << mytid );
  
  int nlocal = 8;
  distribution *level_dist =
    new mpi_block_distribution(decomp,nlocal,-1),
    *new_dist;
  auto coarsen = ioperator(":2");

  auto distributions = new std::vector<distribution*>;
  distributions->push_back(level_dist);
  index_int g = level_dist->outer_size();
  for (int level=0; ; level++) {
    INFO( "level: " << level );
    //printf("On level %d: %s\n",level,level_dist->as_string().data());
    REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
    if (mytid<ntids-1) {
      auto
	ml = new_dist->last_index_r(mycoord),
	nf = new_dist->first_index_r
	    ( processor_coordinate(mycoord.operate(ioperator(">>1")).data()) );
      CHECK( ((ml==nf-1)||(ml==nf)) );
    }
    distributions->push_back(new_dist);
    g /= 2;
    REQUIRE( g==new_dist->outer_size() );
    if (g==1) break;
    level_dist = new_dist;
  }
}

TEST_CASE( "center of mass function","[ortho][11]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 8,k,ipath;
  distribution *level_dist, *new_dist;

  SECTION( "k=1, pieces" ) {
    k = 1; ipath = 0;
  }
  SECTION( "k=1, kernel" ) {
    k = 1; ipath = 1;
  }
  SECTION( "k=2, pieces" ) {
    k = 2; ipath = 0;
  }
  SECTION( "k=2, kernel" ) {
    k = 2; ipath = 1;
  }

  INFO( "k=" << k );
  REQUIRE_NOTHROW( level_dist = new mpi_block_distribution(decomp,k,nlocal,-1) );

  double *data;
  auto coarsen = ioperator(":2");

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  CHECK( level_dist->local_allocation()==k*nlocal );
  CHECK( new_dist->local_allocation()==k*nlocal/2 );
  CHECK( new_dist->get_orthogonal_dimension()==k );
  std::shared_ptr<object> bot,top;
  REQUIRE_NOTHROW( bot = std::shared_ptr<object>( new mpi_object(level_dist) ) );
  REQUIRE_NOTHROW( bot->allocate() );
  REQUIRE_NOTHROW( top = std::shared_ptr<object>( new mpi_object(new_dist) ) );
  REQUIRE_NOTHROW( data = bot->get_data(mycoord) );
  index_int first,lsize;
  first = bot->first_index_r(mycoord).coord(0); lsize = bot->volume(mycoord);
  {
    int loc = 0;
    for (index_int i=0; i<lsize; i++) {
      data[loc++] = first+i;
      if (k>1) data[loc++] = first+i+1.;
    }
  }

  kernel *calculate_cm;
  const char *path;
  if (ipath==0) {
    path = "by pieces";
    REQUIRE_NOTHROW( calculate_cm = new mpi_kernel(bot,top) );
    REQUIRE_NOTHROW( calculate_cm->set_localexecutefn
		     ( [k] ( kernel_function_args ) -> void {
		       scansumk( kernel_function_call,k ); } ) );
    REQUIRE_NOTHROW( calculate_cm->set_signature_function_function
		     ( [] (index_int i) -> std::shared_ptr<indexstruct> {
		       return doubleinterval(i); } ) );
  } else if (ipath==1) {
    path = "as one kernel";
    REQUIRE_NOTHROW( calculate_cm = new mpi_centerofmass_kernel(bot,top,k) );
  }
  INFO( "path: " << path );

  REQUIRE_NOTHROW( calculate_cm->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_cm->execute() );

  first = top->first_index_r(mycoord).coord(0); lsize = top->volume(mycoord);
  REQUIRE_NOTHROW( data = top->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++) {
    INFO( "i: " << i );
    index_int g = 2*(i+first);
    CHECK( data[k*i]==g+g+1 );
    if (k>1)
      CHECK( data[k*i+1]>first );
  }
}

TEST_CASE( "center of mass function, entirely local","[12]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 8,ipath;
  distribution *level_dist, *new_dist;

  REQUIRE_NOTHROW( level_dist = new mpi_block_distribution(decomp,nlocal,-1) );

  auto coarsen = ioperator(":2");
  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  std::shared_ptr<object> bot,top;
  REQUIRE_NOTHROW( top = std::shared_ptr<object>( new mpi_object(new_dist) ) );
  REQUIRE_NOTHROW( bot = std::shared_ptr<object>( new mpi_object(level_dist) ) );

  kernel *calculate_cm;
  REQUIRE_NOTHROW( calculate_cm = new mpi_centerofmass_kernel(bot,top) );
  REQUIRE_NOTHROW( calculate_cm->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = calculate_cm->get_tasks() );
  CHECK( tsks.size()==1 );
  std::shared_ptr<task> tsk;
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  std::vector<message*> msgs; 
  message *msg;

  int recv = 0;
  SECTION( "recv" ) { recv = 1;
    REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  }
  SECTION( "send" ) {
    REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
  }
  CHECK( msgs.size()==1 );
  REQUIRE_NOTHROW( msg = msgs.at(0) );
  CHECK( msg->get_sender()==mycoord );
  CHECK( msg->get_receiver()==mycoord );

  std::shared_ptr<multi_indexstruct> bigstruct;
  REQUIRE_NOTHROW( bigstruct = bot->get_processor_structure(mycoord) );
  std::shared_ptr<multi_indexstruct> tststruct;
  REQUIRE_NOTHROW( tststruct = msg->get_global_struct() );
  INFO( "message struct " << bigstruct->as_string() << "; local struct " << tststruct->as_string() );
  CHECK( bigstruct->equals(tststruct) );
  if (recv==1) {
    REQUIRE_NOTHROW
      ( bigstruct = std::shared_ptr<multi_indexstruct>
	(new multi_indexstruct
	 ( std::shared_ptr<indexstruct>( new contiguous_indexstruct( 0,bigstruct->local_size(0)-1 ) ) )) );
    REQUIRE_NOTHROW( tststruct = msg->get_local_struct() );
    CHECK( bigstruct->equals(tststruct) );
  }
}

TEST_CASE( "center of mass function, redundant","[13]" ) {
  if (ntids<4) { printf("nbody 13 needs at least 4 procs\n"); return; }

  INFO( "mytid=" << mytid );

  distribution *level_dist, *new_dist;

  // one point per processor
  REQUIRE_NOTHROW( level_dist = new mpi_block_distribution(decomp,1,-1) );
  auto coarsen = ioperator(":2");
  REQUIRE_NOTHROW( level_dist = level_dist->operate(coarsen) );
  // so after two coarsenings we are four way redundant
  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );

  CHECK( new_dist->volume(mycoord)==1 );
  int f = new_dist->first_index_r(mycoord).coord(0);
  CHECK( f==mytid/4 );
  
  std::shared_ptr<object> bot,top;
  REQUIRE_NOTHROW( top = std::shared_ptr<object>( new mpi_object(new_dist) ) );
  REQUIRE_NOTHROW( bot = std::shared_ptr<object>( new mpi_object(level_dist) ) );

  kernel *calculate_cm;
  REQUIRE_NOTHROW( calculate_cm = new mpi_centerofmass_kernel(bot,top) );
  REQUIRE_NOTHROW( calculate_cm->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = calculate_cm->get_tasks() );
  CHECK( tsks.size()==1 );
  std::shared_ptr<task> tsk;
  REQUIRE_NOTHROW( tsk = tsks.at(0) );
  std::vector<message*> msgs; 
  message *msg;

  { INFO( "recv analysis" );
    REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
    for ( auto m : msgs ) {
      CHECK( m->get_receiver()==mycoord );
      auto tstruct = m->get_local_struct(),
	sstruct = m->get_global_struct();
      if ( tstruct->first_index_r()[0]==0 ) {
	CHECK( sstruct->first_index_r()[0]==2*f );
      } else if ( tstruct->first_index_r()[0]==1 ) {
	CHECK( sstruct->first_index_r()[0]==2*f+1 );
      } else REQUIRE( tstruct->first_index_r()[0]==-999 );
    }
    CHECK( msgs.size()==2 );
  }
  { INFO( "recv analysis" );
    REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
    for ( auto m : msgs ) {
      CHECK( m->get_sender()==mycoord );
      auto sstruct = m->get_global_struct();
      CHECK( ( sstruct->first_index_r()[0]/2 )==f );
    }
    printf("%d\n",msgs.size());
    // no. CHECK( msgs->size()==2 );
  }
}


TEST_CASE( "force prolongation","[21]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 8;
  distribution *level_dist =
    new mpi_block_distribution(decomp,nlocal,-1),
    *new_dist;
  double *data;
  auto coarsen = ioperator("/2");

  mpi_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new mpi_sparse_matrix(level_dist) );
  index_int
    f = level_dist->first_index_r(mycoord).coord(0), l = level_dist->last_index_r(mycoord).coord(0),
    g = level_dist->global_size().at(0);
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

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  INFO( "coarsened dist is of type " << new_dist->type_as_string() );
  CHECK( new_dist->has_type_locally_contiguous() );
  auto bot = std::shared_ptr<object>( new mpi_object(level_dist) ),
    top = std::shared_ptr<object>( new mpi_object(new_dist) ),
    side = std::shared_ptr<object>( new mpi_object(level_dist) ),
    expanded = std::shared_ptr<object>( new mpi_object(level_dist) ),
    multiplied = std::shared_ptr<object>( new mpi_object(level_dist) );

  REQUIRE_NOTHROW( top->allocate() );
  REQUIRE_NOTHROW( side->allocate() );
  index_int first,lsize;
  // fill in the half size top level
  first = top->first_index_r(mycoord).coord(0); lsize = top->volume(mycoord);
  REQUIRE_NOTHROW( data = top->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++)
    data[i] = first+i;
  // fill in the other half tree
  first = side->first_index_r(mycoord).coord(0); lsize = side->volume(mycoord);
  REQUIRE_NOTHROW( data = side->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++)
    data[i] = 1.;

  kernel *expand,*multiply, *calculate_cm;
  const char *path;
  SECTION( "in pieces" ) {
    path = "separate kernels";
    REQUIRE_NOTHROW( expand = new mpi_kernel(top,expanded) );
    REQUIRE_NOTHROW( expand->set_localexecutefn( &scanexpand ) );
    REQUIRE_NOTHROW( expand->last_dependency()->set_signature_function_function
		     ( [] (index_int i) -> std::shared_ptr<indexstruct> {
		       return halfinterval(i); } ) );
    REQUIRE_NOTHROW( expand->analyze_dependencies() );
    REQUIRE_NOTHROW( expand->execute() );
    
    REQUIRE_NOTHROW( multiply = new mpi_spmvp_kernel(side,multiplied,mat) );
    REQUIRE_NOTHROW( multiply->analyze_dependencies() );
    REQUIRE_NOTHROW( multiply->execute() );
    
    REQUIRE_NOTHROW( calculate_cm = new mpi_sum_kernel(expanded,multiplied,bot) );
  }
  SECTION( "as one kernel" ) {
    path = "all in one";
    REQUIRE_NOTHROW( calculate_cm = new mpi_sidewaysdown_kernel(top,side,bot,mat) );
  }
  INFO( "path: " << path );

  REQUIRE_NOTHROW( calculate_cm->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_cm->execute() );

  REQUIRE_NOTHROW( data = bot->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++) {
    index_int ig = first+i;
    INFO( "ig: " << ig );
    if (ig==0 || ig==g-1 )
      CHECK( data[i]==ig/2+1 ); // divide by two because of the expand
    else
      CHECK( data[i]==ig/2+2 );
  }
}

TEST_CASE( "force prolongation, short","[22]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 8, g = ntids*nlocal;
  distribution *level_dist =
    new mpi_block_distribution(decomp,g),
    *new_dist;
  double *data;
  auto coarsen = ioperator(":2");

  mpi_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new mpi_toeplitz3_matrix(level_dist,1.,0.,1.) );

  REQUIRE_NOTHROW( new_dist = level_dist->operate(coarsen) );
  auto bot = std::shared_ptr<object>( new mpi_object(level_dist) ),
    top = std::shared_ptr<object>( new mpi_object(new_dist) ),
    side = std::shared_ptr<object>( new mpi_object(level_dist) ),
    expanded = std::shared_ptr<object>( new mpi_object(level_dist) ),
    multiplied = std::shared_ptr<object>( new mpi_object(level_dist) );

  REQUIRE_NOTHROW( top->allocate() );
  REQUIRE_NOTHROW( side->allocate() );
  index_int first,lsize;
  // fill in the half size top level
  first = top->first_index_r(mycoord).coord(0); lsize = top->volume(mycoord);
  REQUIRE_NOTHROW( data = top->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++)
    data[i] = first+i;
  // fill in the other half tree
  first = side->first_index_r(mycoord).coord(0); lsize = side->volume(mycoord);
  REQUIRE_NOTHROW( data = side->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++)
    data[i] = 1.;

  kernel *expand,*multiply, *calculate_cm;
  const char *path;
  SECTION( "as one kernel" ) {
    path = "all in one";
    REQUIRE_NOTHROW( calculate_cm = new mpi_sidewaysdown_kernel(top,side,bot,mat) );
  }
  INFO( "path: " << path );

  REQUIRE_NOTHROW( calculate_cm->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_cm->execute() );

  REQUIRE_NOTHROW( data = bot->get_data(mycoord) );
  for (index_int i=0; i<lsize; i++) {
    index_int ig = first+i;
    INFO( "ig: " << ig );
    if (ig==0 || ig==g-1 )
      CHECK( data[i]==ig/2+1 ); // divide by two because of the expand
    else
      CHECK( data[i]==ig/2+2 );
  }
}

