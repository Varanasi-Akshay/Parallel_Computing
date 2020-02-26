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
 **** application level operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "product_base.h"
#include "product_ops.h"
#include "product_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Copy kernel","[kernel][copy][31]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000,gsize = nlocal*mpi_nprocs;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution *blocked = new product_block_distribution(decomp,gsize);
  auto
    x = std::shared_ptr<object>( new product_object(blocked) ),
    xnew = std::shared_ptr<object>( new product_object(blocked) ),
    z = std::shared_ptr<object>( new product_object(blocked) ),
    r = std::shared_ptr<object>( new product_object(blocked) ), 
    rnew = std::shared_ptr<object>( new product_object(blocked) ),
    p = std::shared_ptr<object>( new product_object(blocked) ), 
    pold = std::shared_ptr<object>( new product_object(blocked) ),
    q = std::shared_ptr<object>( new product_object(blocked) ), 
    qold = std::shared_ptr<object>( new product_object(blocked) );
  
  // scalars, all redundantly replicated
  distribution *scalar = new product_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new product_object(scalar) ), 
    rrp = std::shared_ptr<object>( new product_object(scalar) ),
    rnorm = std::shared_ptr<object>( new product_object(scalar) ),
    pap = std::shared_ptr<object>( new product_object(scalar) ),
    alpha = std::shared_ptr<object>( new product_object(scalar) ), 
    beta = std::shared_ptr<object>( new product_object(scalar) );
  
  kernel *rrcopy;
  int n; const char *path;
  double *rdata,*sdata;
  
  SECTION( "scalar" ) {
    n = 1; path = "scalar";
    rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata[i] = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = new product_copy_kernel( rr,rrp ) );
    CHECK( rrcopy->has_type_compute() );
  }
  SECTION( "vector" ) {
    n = nlocal; path = "vector";
    rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata[i] = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = new product_copy_kernel( r,z ) );
  }
  INFO( "variant: " << path );

  REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
  REQUIRE_NOTHROW( rrcopy->execute() );
  for (int i=0; i<n; i++)
    CHECK( sdata[i]==Approx(2.*(mytid*n+i)) );
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=12; const char *mode = "default";
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  ioperator *no_op = new ioperator("none");
  product_distribution *block = 
    new product_block_distribution(decomp,nlocal*mpi_nprocs);
  double *xdata = new double[nlocal];
  auto
    xvector = std::shared_ptr<object>( new product_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new product_object(block) );
  domain_coordinate
    my_first = block->first_index_r(mycoord), my_last = block->last_index_r(mycoord);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first[0]);
  product_kernel *scale;
  double *halo_data,*ydata;

  SECTION( "scale by constant" ) {

    SECTION( "kernel by pieces" ) { 
      scale = new product_kernel(xvector,yvector);
      scale->set_name("33scale");
      dependency *d;
      REQUIRE_NOTHROW( d = scale->last_dependency() );
      REQUIRE_NOTHROW( d->set_type_local() );

      SECTION( "constant in the function" ) {
	scale->set_localexecutefn( &vecscalebytwo );
      }

      SECTION( "constant passed as context" ) {
	double x = 2;
	scale->set_localexecutefn( &vecscaleby );
	scale->set_localexecutectx( (void*)&x );
      }

      SECTION( "constant passed inverted" ) {
	double x = 1./2;
	scale->set_localexecutefn( &vecscaledownby );
	scale->set_localexecutectx( (void*)&x );
      }
    }
    SECTION( "kernel as kernel from double-star" ) {
      mode = "scale kernel with scalar";
      double x = 2;
      scale = new product_scale_kernel(&x,xvector,yvector);
    }
    SECTION( "kernel as kernel from replicated object" ) {
      mode = "scale kernel with object";
      double xval = 2;
      distribution *scalar = new product_replicated_distribution(decomp);
      auto x = std::shared_ptr<object>( new product_object(scalar) );
      x->set_value(&xval);
      CHECK( scalar->has_type_replicated() );
      CHECK( x->has_type_replicated() );
      REQUIRE_NOTHROW( scale = new product_scale_kernel(x,xvector,yvector) );
      REQUIRE_NOTHROW( scale->analyze_dependencies() );
      std::vector<dependency*> *deps;
      REQUIRE_NOTHROW( deps = scale->get_dependencies() );
      CHECK( deps->size()==2 );
    }

    INFO( "mode is " << mode );
    REQUIRE_NOTHROW( scale->analyze_dependencies() );
    CHECK( scale->get_tasks().size()==1 );

    std::shared_ptr<task> scale_task;
    CHECK_NOTHROW( scale_task = (product_task*) ( *scale->get_tasks() )[0] );
    CHECK_NOTHROW( scale->execute() );
    CHECK_NOTHROW( halo_data = scale_task->get_beta_object(0)->get_data(mycoord) );
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    {
      int i;
      for (i=0; i<nlocal; i++) {
	CHECK( halo_data[i] == Approx( pointfunc33(i,my_first[0]) ) );
      }
    }
  }

  SECTION( "axpy kernel" ) { mode = "axpy kernel";
    double x=2;
    scale = new product_axpy_kernel(xvector,yvector,&x);
    scale->analyze_dependencies();
    scale->execute();
  }

  INFO( "mode is " << mode );
  {
    int i;
    CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    for (i=0; i<nlocal; i++) {
      CHECK( ydata[i] == Approx( 2*pointfunc33(i,my_first[0])) );
    }
  }
}

#if 0

TEST_CASE( "Beta from sparse matrix","[beta][sparse][41]" ) {

  int localsize=20,gsize=localsize*ntids;
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->local_size(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = .5;
  }
  product_kernel *kern = new product_kernel(in_obj,out_obj);
  kern->set_name("sparse-stuff");
  kern->set_localexecutefn( &local_sparse_matrix_vector_multiply );
  product_sparse_matrix *pattern;

  SECTION( "connect right" ) {
    REQUIRE_NOTHROW( pattern = new product_sparse_matrix(block) );
    for (index_int i=block->first_index_r(mycoord); i<=block->last_index_r(mycoord); i++) {
      pattern->add_element(i,i);
      if (i+1<gsize) {
	CHECK_NOTHROW( pattern->add_element(i,i+1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i+1) );
      }
    }
    REQUIRE_NOTHROW( kern->last_dependency()->set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = kern->get_tasks() );
    CHECK( tasks->size()==1 );
    std::shared_ptr<task> threetask;
    //REQUIRE_NOTHROW( threetask = (product_task*)(*tasks)[0] );
    REQUIRE_NOTHROW( threetask = tasks.at(0) );
    std::vector<message*> *rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==ntids-1) {
      CHECK( rmsgs->size()==1 );
    } else {
      CHECK( rmsgs->size()==2 );
    }
    for (auto m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
      message *msg = (message*)(*m);
      if (msg->get_sender().coord(0)==msg->get_receiver()) {
	CHECK( msg->size()==localsize );
      } else {
	CHECK( msg->size()==1 );
      }
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
    // double *outdata = out_obj->get_data(mycoord); index_int n = out_obj->local_size(mycoord);
    // if (mytid==ntids-1) {
    //   for (index_int i=0; i<n-1; i++)
    // 	CHECK( outdata[i] == Approx(1.) );
    //   CHECK( outdata[n-1] == Approx(.5) );
    // } else {
    //   for (index_int i=0; i<n; i++)
    // 	CHECK( outdata[i] == Approx(1.) );
    // }
  }

  SECTION( "connect left" ) {
    REQUIRE_NOTHROW( pattern = new product_sparse_matrix(block) );
    for (index_int i=block->first_index_r(mycoord); i<=block->last_index_r(mycoord); i++) {
      pattern->add_element(i,i);
      if (i-1>=0) {
	CHECK_NOTHROW( pattern->add_element(i,i-1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i-1) );
      }
    }
    REQUIRE_NOTHROW( kern->last_dependency()->set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = kern->get_tasks() );
    CHECK( tasks->size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = (product_task*)(*tasks)[0] );
    std::vector<message*> *rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==0) {
      CHECK( rmsgs->size()==1 );
    } else {
      CHECK( rmsgs->size()==2 );
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
  }

}

TEST_CASE( "Actual sparse matrix","[beta][sparse][spmvp][42]" ) {
  REQUIRE(ntids>1); // need at least two processors
  INFO( "mytid: " << mytid );
  
  int localsize=10,gsize=localsize*ntids;
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->local_size(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }
  product_kernel *spmvp = new product_kernel(in_obj,out_obj);
  spmvp->set_name("sparse-mvp");

  int ncols = 1;
  index_int mincol = block->first_index_r(mycoord), maxcol = block->last_index_r(mycoord);
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->global_ndomains()));

  // create a matrix with zero row sums and 1.5 excess on the diagonal: ncols elements of -1 off diagonal
  // also build up an indexstruct my_columns
  product_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new product_sparse_matrix(block) );
  index_int my_first = block->first_index_r(mycoord),my_last = block->last_index_r(mycoord);
  indexed_indexstruct *my_columns = new indexed_indexstruct();
  for (index_int row=my_first; row<=my_last; row++) {
    // diagonal element
    REQUIRE_NOTHROW( mat->add_element(row,row,(double)ncols+1.5) );
    my_columns->add_element( row );
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
      CHECK( xs>=0 );
      CHECK( xs<localsize );
      if (mytid<ntids-1) col = my_last+1+xs;
      else               col = xs;
      REQUIRE( col>=0 );
      REQUIRE( col<gsize );
      if ( !((col<my_first) || ( col>my_last )) )
	printf("range error in row %d: %d [%d-%d]\n",row,col,my_first,my_last);
      REQUIRE( ((col<my_first) || ( col>my_last )) );
      if (col<mincol) mincol = col;
      if (col>maxcol) maxcol = col;
      REQUIRE_NOTHROW( mat->add_element(row,col,-1.) );
      REQUIRE_NOTHROW( my_columns->add_element(col) );
    }
    CHECK( mat->row_sum(row)==Approx(1.5) );
  }

  // test that we can find the matrix columns correctly
  indexstruct *mstruct;
  REQUIRE_NOTHROW( mstruct = mat->all_local_columns() );
  REQUIRE_NOTHROW( mstruct = mstruct->convert_to_indexed() );
  // printf("matrix structure on %d: %s\ncollected columns on %d %s",
  // 	 mytid,mstruct->as_string().data(),mytid,my_columns->as_string().data());
  CHECK( mstruct->equals(my_columns) );

  spmvp->last_dependency()->set_index_pattern( mat );
  spmvp->set_localexecutefn( &local_sparse_matrix_vector_multiply );
  spmvp->set_localexecutectx( mat );
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
  CHECK( tasks->size()==1 );
  std::shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = (product_task*)(*tasks)[0] );
  std::vector<message*> *rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for (std::vector<message*>::iterator m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
    message *msg = (message*)(*m);
    if (msg->get_sender()!=msg->get_receiver())
      if (mytid==ntids-1)
	CHECK( msg->get_sender().coord(0)==0 );
      else 
	CHECK( msg->get_sender().coord(0)==mytid+1 );
  }

  {
    distribution *beta_dist;
    REQUIRE_NOTHROW( beta_dist = spmvp->last_dependency()->get_beta_distribution() );
    indexstruct *column_indices;
    REQUIRE_NOTHROW( column_indices = beta_dist->get_processor_structure(mycoord) );
    CHECK( column_indices->is_sorted() ); 
    REQUIRE_NOTHROW( mat->remap( block,beta_dist,mytid ) );
  }

  {
    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
    CHECK( tasks->size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = (product_task*)(*tasks)[0] );
  }
  REQUIRE_NOTHROW( spmvp->execute() );

  {
    double *data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->local_size(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data[i] == Approx(1.5) );
    }
  }
}

TEST_CASE( "Sparse matrix kernel","[beta][sparse][spmvp][43]" ) {
  if (ntids==1) { printf("test 43 is multi-processor\n"); return; };

  int localsize=200,gsize=localsize*ntids;
  product_distribution *block = new product_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new product_object(block) ),
    out_obj = std::shared_ptr<object>( new product_object(block) );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->local_size(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }

  int ncols = 4;
  index_int mincol = block->first_index_r(mycoord), maxcol = block->last_index_r(mycoord);
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->global_ndomains()));

  // create a matrix with zero row sums, shift diagonal up
  double dd = 1.5;
  product_sparse_matrix *mat = new product_sparse_matrix(block);
  index_int my_first = block->first_index_r(mycoord),my_last = block->last_index_r(mycoord);
  for (index_int row=my_first; row<=my_last; row++) {
    indexstruct *nzcol = new indexed_indexstruct();
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
      CHECK( xs>=0 );
      CHECK( xs<localsize );
      if (mytid<ntids-1) col = my_last+1+xs;
      else               col = xs;
      while (nzcol->contains_element(col)) col++;
      nzcol->add_element(col);
      if (col<mincol) mincol = col;
      if (col>maxcol) maxcol = col;
      mat->add_element(row,col,-1.); // off elt
    }
    mat->add_element(row,row,ncols+dd); // diag elt
  }

  product_kernel *spmvp = new product_spmvp_kernel(in_obj,out_obj,mat);
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
  CHECK( tasks->size()==1 );
  std::shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = (product_task*)(*tasks)[0] );
  std::vector<message*> *rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for (std::vector<message*>::iterator m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
    message *msg = (message*)(*m);
    if (msg->get_sender()!=msg->get_receiver())
      if (mytid==ntids-1)
	CHECK( msg->get_sender().coord(0)==0 );
      else 
	CHECK( msg->get_sender().coord(0)==mytid+1 );
  }

  {
    distribution *beta_dist;
    REQUIRE_NOTHROW( beta_dist = spmvp->last_dependency()->get_beta_distribution() );
    indexstruct *column_indices;
    REQUIRE_NOTHROW( column_indices = beta_dist->get_processor_structure(mycoord) );
    //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
    CHECK( column_indices->is_sorted() ); 
    REQUIRE_NOTHROW( mat->remap( block,beta_dist,mytid ) );
  }

  REQUIRE_NOTHROW( spmvp->execute() );

  {
    double *data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->local_size(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data[i] == Approx(dd) );
    }
  }
}

TEST_CASE( "matrix kernel analysis","[kernel][spmvp][44]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution *blocked =
    new product_block_distribution(decomp,g);
  auto
    x = std::shared_ptr<object>( new product_object(blocked) ), 
    y = std::shared_ptr<object>( new product_object(blocked) );

  // set the matrix to one lower diagonal
  product_sparse_matrix
    *Aup = new product_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_volume(),
      my_first = blocked->first_index_r(mycoord),
      my_last = blocked->last_index_r(mycoord);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      // A narrow is tridiagonal
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  kernel *k;
  REQUIRE_NOTHROW( k = new product_spmvp_kernel( x,y,Aup ) );
  REQUIRE_NOTHROW( k->analyze_dependencies() );

  // analyze the message structure
  auto *tasks = k->get_tasks();
  CHECK( tasks->size()==1 );
  for ( auto t : tasks ) { //->begin(); t!=tasks->end(); ++t) {
    if (t->get_step()==x->get_object_number()) {
      CHECK( t->get_dependencies()->size()==0 );
    } else {
      auto send = t->get_send_messages();
      if (mytid==ntids-1)
	CHECK( send.size()==1 );
      else
	CHECK( send.size()==2 );
      auto recv = t->get_receive_messages();
      if (mytid==0)
	CHECK( recv.size()==1 );
      else
	CHECK( recv.size()==2 );
    }
  }

  SECTION( "limited to proc 0" ) {
    // set the input vector to delta on the first element
    {
      double *d = x->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) d[i] = 0.;
      if (mytid==0) d[0] = 1.;
    }

    // check that we get a nicely propagating wave
    REQUIRE_NOTHROW( k->execute() );
    index_int my_first = blocked->first_index_r(mycoord);
    double *d = y->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) {
      index_int g = my_first+i;
      INFO( "global index " << g );
      if (g<2) // s=0, g=0,1
	CHECK( d[i]!=Approx(0.) );
      else
	CHECK( d[i]==Approx(0.) );
    }
  }
  SECTION( "crossing over" ) {
    // set the input vector to delta on the right edge
    {
      double *d = x->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) d[i] = 0.;
      d[nlocal-1] = 1.;
    }

    // check that we get a nicely propagating wave
    REQUIRE_NOTHROW( k->execute() );
    index_int my_first = blocked->first_index_r(mycoord);
    double *d = y->get_data(mycoord);
    for (index_int i=1; i<nlocal-1; i++)
      CHECK( d[i]==Approx(0.) );
    if (mytid==0)
      CHECK( d[0]==Approx(0.) );
    else 
      CHECK( d[0]==Approx(1.) );
    CHECK( d[nlocal-1]==Approx(1.) );
  }
}

TEST_CASE( "matrix iteration shift left","[kernel][45]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal, nsteps = 2; //2*nlocal+3;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution *blocked =
    new product_block_distribution(decomp,g);
  auto x = std::shared_ptr<object>( new product_object(blocked) );

  // set the input vector to delta on the first element
  {
    double *d = x->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) d[i] = 0.;
    if (mytid==0) d[0] = 1.;
  }
  
  auto y = std::vector<std::shared_ptr<object>>(nsteps);
  for (int i=0; i<nsteps; i++) {
    y[i] = std::shared_ptr<object>( new product_object(blocked) );
  }

  // set the matrix to one lower diagonal
  product_sparse_matrix
    *Aup = new product_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_volume(),
      my_first = blocked->first_index_r(mycoord),
      my_last = blocked->last_index_r(mycoord);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  // make a queue
  auto queue = std::shared_ptr<algorithm>( new product_algorithm(arch) );
  REQUIRE_NOTHROW( queue->add_kernel( new product_origin_kernel(x) ) );
  object *inobj = x, *outobj;
  for (int i=0; i<nsteps; i++) {
    outobj = y[i];
    kernel *k;
    REQUIRE_NOTHROW( k = new product_spmvp_kernel( inobj,outobj,Aup ) );
    fmt::MemoryWriter w; w.write("mvp-{}",i); k->set_name( w.str() );
    REQUIRE_NOTHROW( queue->add_kernel(k) );
    inobj = outobj;
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );

  // analyze the message structure
  auto *tasks = queue->get_tasks();
  CHECK( tasks->size()==(nsteps+1) );
  for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
    if ((*t)->get_step()==x->get_object_number()) {
      CHECK( (*t)->get_dependencies()->size()==0 );
    } else {
      std::vector<dependency*> *deps = (*t)->get_dependencies();
      CHECK( deps->size()==1 );
      dependency *dep = deps->at(0);

      INFO( "step " << (*t)->get_step() );

      std::vector<message*> recv,send;
      REQUIRE_NOTHROW( recv = (*t)->get_receive_messages() );
      REQUIRE_NOTHROW( send = (*t)->get_send_messages() );

      fmt::MemoryWriter w;
      w.write("[{}] receiving from: ",mytid);
      for (int i=0; i<recv->size(); i++)
	REQUIRE_NOTHROW( w.write("{},",recv.at(i)->get_sender()) );
      w.write(". sending to: ");
      for (int i=0; i<send.size(); i++)
	REQUIRE_NOTHROW( w.write("{},",send.at(i)->get_receiver()) );

      INFO( "Send/recv: " << w.str() );
      if (mytid==0)
      	CHECK( recv.size()==1 );
      else
      	CHECK( recv.size()==2 );

      if (mytid==ntids-1)
      	CHECK( send.size()==1 );
      else
      	CHECK( send.size()==2 );
    }
  }

  // check that we get a nicely propagating wave
  REQUIRE_NOTHROW( queue->execute() );
  index_int my_first = blocked->first_index_r(mycoord);
  for (int s=0; s<nsteps; s++) {
    INFO( "step " << s );
    double *d = y[s]->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) {
      index_int g = my_first+i;
      INFO( "global index " << g );
      if (g<s+2) // s=0, g=0,1
  	CHECK( d[i]!=Approx(0.) );
      else
  	CHECK( d[i]==Approx(0.) );
    }
  }
}

TEST_CASE( "compound kernel queue","[kernel][queue][50]" ) {

  int nlocal = 50;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution
    *block = new product_block_distribution(decomp,nlocal*ntids),
    *scalar = new product_replicated_distribution(decomp);
  auto
    x = std::shared_ptr<object>( new product_object(block) ),
    y = std::shared_ptr<object>( new product_object(block) ),
    xy = std::shared_ptr<object>( new product_object(scalar) );
  kernel
    *makex = new product_origin_kernel(x),
    *makey = new product_origin_kernel(y),
    *prod = new product_innerproduct_kernel(x,y,xy);
  auto queue = std::shared_ptr<algorithm>( new product_algorithm(arch) );
  int inprod_step;

  SECTION( "analyze in steps" ) {

    SECTION( "kernels in logical order" ) {
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      inprod_step = 2;
    }
  
    SECTION( "kernels in wrong order" ) {
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      inprod_step = 0;
    }
  
    //CHECK( queue->get_step_count()==5 ); // 3 for innerproduct, 2 for origins

    std::vector<std::shared_ptr<task>> tasks;
    std::vector<kernel*> *kernels;
    REQUIRE_NOTHROW( kernels = queue->get_kernels() );
    CHECK( kernels->size()==3 );
    for (int ik=0; ik<kernels->size(); ik++) {
      kernel *k;
      REQUIRE_NOTHROW( k = kernels->at(ik) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );
      REQUIRE_NOTHROW( tasks = k->get_tasks() );
      if (ik==inprod_step) 
	CHECK( tasks->size()==2 );
      else 
	CHECK( tasks->size()==1 );
      REQUIRE_NOTHROW( queue->add_kernel_tasks_to_queue(k) );
    }
    
    REQUIRE_NOTHROW( tasks = queue->get_tasks() );
    for (std::vector<task*>::iterator t=tasks->begin(); t!=tasks->end(); ++t) {
      if (!(*t)->has_type_origin()) {
	object *in; int inn; int ostep;
	CHECK_NOTHROW( in = (*t)->last_dependency()->get_in_object() );
	CHECK_NOTHROW( inn = in->get_object_number() );
	CHECK( inn>=0 );
      }
    }
  }
  SECTION( "single analyze call" ) {
    REQUIRE_NOTHROW( queue->add_kernel(makex) );
    REQUIRE_NOTHROW( queue->add_kernel(makey) );
    REQUIRE_NOTHROW( queue->add_kernel(prod) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    inprod_step = 2;
  }
  std::vector<std::shared_ptr<task>> tsks;
  REQUIRE_NOTHROW( tsks = queue->get_tasks() );
  for ( t : tsks ) { //std::vector<task*>::iterator t=tsks->begin(); t!=tsks->end(); ++t) {
    int ik;
    REQUIRE_NOTHROW( ik = t->get_step() );
    if (ik==inprod_step) {
      REQUIRE_NOTHROW( t->get_n_in_objects()==2 );
    } else {
      REQUIRE_NOTHROW( t->get_n_in_objects()==0 );
    }
  }
}

TEST_CASE( "r_norm1, different ways","[kernel][collective][55]" ) {

  INFO( "mytid: " << mytid );

  int
    P = env->get_architecture()->nprocs(), g=P-1;
  const char *mode;

  double *data;
  distribution *local_scalar = new product_block_distribution(decomp,1,-1);
  auto local_value = std::shared_ptr<object>( new product_object(local_scalar) );
  REQUIRE_NOTHROW( data = local_value->get_data(mycoord) );
  data[0] = (double)mytid;

  product_distribution *replicated;
  REQUIRE_NOTHROW( replicated = new product_replicated_distribution(decomp) );
  auto global_sum;
  REQUIRE_NOTHROW( global_sum = std::shared_ptr<object>( new product_object(replicated) ) );

  SECTION( "spell out the way to the top" ) {
    mode = "spell it out";
    int
      groupsize = MIN(4,P), mygroupnum = mytid/groupsize, nfullgroups = ntids/groupsize,
      grouped_tids = nfullgroups*groupsize, // how many procs are in perfect groups?
      remainsize = P-grouped_tids, ngroups = nfullgroups+(remainsize>0);
    CHECK( remainsize<groupsize );
    product_distribution *locally_grouped;
    { // every proc gets to know all the indices of its group
      parallel_indexstruct *groups;
      REQUIRE_NOTHROW( groups = new parallel_indexstruct( env->get_architecture() ) );
      for (int p=0; p<P; p++) {
	index_int groupnumber = p/groupsize,
	  f = groupsize*groupnumber,l=MIN(f+groupsize-1,g);
	REQUIRE( l>=f );
	indexstruct *pstruct;
	REQUIRE_NOTHROW( pstruct = new contiguous_indexstruct(f,l) );
	REQUIRE_NOTHROW( groups->set_processor_structure(p,pstruct ) );
      }
      REQUIRE_NOTHROW( locally_grouped = new product_distribution(groups) );
    }

    product_distribution *partially_reduced;
    {
      parallel_indexstruct
	*partials = new parallel_indexstruct( env->get_architecture() );
      for (int p=0; p<P; p++) {
	index_int groupnumber = p/groupsize;
	partials->set_processor_structure(p, new contiguous_indexstruct(groupnumber) );
      }
      REQUIRE_NOTHROW( partially_reduced = new product_distribution(partials) );
    }
    auto partial_sums;
    REQUIRE_NOTHROW( partial_sums = std::shared_ptr<object>( new product_object(partially_reduced) ) );
  
    SECTION( "group and sum separate" ) {
      auto local_groups;
      product_kernel *partial_grouping,*local_summing_to_global;

      // one kernel for gathering the local values
      REQUIRE_NOTHROW( local_groups = std::shared_ptr<object>( new product_object(locally_grouped) ) );
      REQUIRE_NOTHROW( partial_grouping = new product_kernel(local_value,local_groups) );
      REQUIRE_NOTHROW( partial_grouping->set_localexecutefn( &veccopy ) );
      REQUIRE_NOTHROW( partial_grouping->set_explicit_beta_distribution(locally_grouped) );
      REQUIRE_NOTHROW( partial_grouping->analyze_dependencies() );
      REQUIRE_NOTHROW( partial_grouping->execute() );

      std::vector<message*> msgs;
      std::vector<std::shared_ptr<task>> tsks;
      double *data;
      REQUIRE_NOTHROW( tsks = partial_grouping->get_tasks() );
      int nt = tsks.size(); CHECK( nt==1 );
      std::shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks.at(0) );
      REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
      if (mytid<grouped_tids) {
	CHECK( msgs->size()==groupsize );
      } else {
	CHECK( msgs->size()==remainsize );
      }
      REQUIRE_NOTHROW( msgs = t->get_send_messages() );
      CHECK( local_groups->first_index_r(mycoord)==mygroupnum*groupsize );
      if (mytid<grouped_tids) {
	CHECK( msgs.size()==groupsize );
	CHECK( local_groups->local_size(mycoord)==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
	CHECK( local_groups->local_size(mycoord)==remainsize );
      }
      REQUIRE_NOTHROW( data = local_groups->get_data(mycoord) );
      for (index_int i=0; i<local_groups->local_size(mycoord); i++) {
	index_int ig = (local_groups->first_index_r(mycoord)+i);
	INFO( "ilocal=" << i << ", iglobal=" << ig );
	CHECK( data[i]==ig );
      }

      REQUIRE_NOTHROW( local_summing_to_global = new product_kernel(local_groups,partial_sums) );
      //REQUIRE_NOTHROW( local_summing_to_global->set_type_local() );
      REQUIRE_NOTHROW( local_summing_to_global->set_explicit_beta_distribution
		       (locally_grouped) );
      REQUIRE_NOTHROW( local_summing_to_global->set_localexecutefn( &summing ) );
      REQUIRE_NOTHROW( local_summing_to_global->analyze_dependencies() );
      REQUIRE_NOTHROW( local_summing_to_global->execute() );

      // duplicate code with previous section
      REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
      CHECK( partial_sums->local_size(mycoord)==1 );
      index_int f = locally_grouped->first_index_r(mycoord),
	l = locally_grouped->last_index_r(mycoord), s = (l+f)*(l-f+1)/2;
      CHECK( data[0]==s );
    }

    SECTION( "group and sum in one" ) {
      product_kernel *partial_summing;
      REQUIRE_NOTHROW( partial_summing = new product_kernel(local_value,partial_sums) );
      REQUIRE_NOTHROW( partial_summing->set_explicit_beta_distribution(locally_grouped) );
      REQUIRE_NOTHROW( partial_summing->set_localexecutefn( &summing ) );
      REQUIRE_NOTHROW( partial_summing->analyze_dependencies() );
      REQUIRE_NOTHROW( partial_summing->execute() );

      std::vector<message*> msgs;
      std::vector<std::shared_ptr<task>> tsks;
      double *data;
      REQUIRE_NOTHROW( tsks = partial_summing->get_tasks() );
      int nt = tsks.size(); CHECK( nt==1 );
      std::shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks.at(0) );
      REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
      if (mytid<grouped_tids) {
	CHECK( msgs->size()==groupsize );
      } else {
	CHECK( msgs->size()==remainsize );
      }
      REQUIRE_NOTHROW( msgs = t->get_send_messages() );
      CHECK( partial_sums->local_size(mycoord)==1 );
      if (mytid<grouped_tids) {
	CHECK( msgs.size()==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
      }

      // duplicate code with previous section
      REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
      CHECK( partial_sums->local_size(mycoord)==1 );
      index_int f = locally_grouped->first_index_r(mycoord),
	l = locally_grouped->last_index_r(mycoord), s = (l+f)*(l-f+1)/2;
      CHECK( data[0]==s );
    }

    product_kernel *top_summing;
    REQUIRE_NOTHROW( top_summing = new product_kernel(partial_sums,global_sum) );
    parallel_indexstruct *top_beta = new parallel_indexstruct( env->get_architecture() );
    for (int p=0; p<P; p++)
      top_beta->set_processor_structure(p, new contiguous_indexstruct(0,ngroups-1));
    REQUIRE_NOTHROW( top_summing->set_explicit_beta_distribution
		     ( new product_distribution(top_beta) ) );
    REQUIRE_NOTHROW( top_summing->set_localexecutefn( &summing ) );
    REQUIRE_NOTHROW( top_summing->analyze_dependencies() );
    REQUIRE_NOTHROW( top_summing->execute() );
  }

  SECTION( "using the reduction kernel" ) {
    product_kernel *sumkernel;
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel, send/recv";
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( sumkernel = new product_reduction_kernel(local_value,global_sum) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel, grouping";
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( sumkernel = new product_reduction_kernel(local_value,global_sum) );
    }
    REQUIRE_NOTHROW( sumkernel->analyze_dependencies() );
    REQUIRE_NOTHROW( sumkernel->execute() );
  }
    
  INFO( "mode: " << mode );
  data = global_sum->get_data(mycoord);
  CHECK( data[0]==(g*(g+1)/2) );
};

TEST_CASE( "explore the reduction kernel","[reduction][kernel][56]" ) {
  product_distribution
    *local_scalar = new product_block_distribution(decomp,1,-1),
    *global_scalar = new product_replicated_distribution(decomp);
  CHECK( local_scalar->local_size(mycoord)==1 );
  CHECK( local_scalar->first_index_r(mycoord)==mytid );
  CHECK( global_scalar->local_size(mycoord)==1 );
  CHECK( global_scalar->first_index_r(mycoord)==0 );
  auto
    local_value = std::shared_ptr<object>( new product_object(local_scalar) ),
    global_value = std::shared_ptr<object>( new product_object(global_scalar) );
  double *data = local_value->get_data(mycoord); data[0] = mytid;
  product_kernel *reduction;

  int psqrt = sqrt(ntids);
  if (psqrt*psqrt<ntids) {
    printf("Test [56] needs square number of processors\n"); return; }

  printf("collective strategy disabled\n"); return;

  SECTION( "send/recv" ) {
    REQUIRE_NOTHROW( arch->set_collective_strategy_ptp() );
    REQUIRE_NOTHROW( reduction = new product_reduction_kernel(local_value,global_value) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->nmessages==ntids+1 );
  }
  SECTION( "grouping" ) {
    REQUIRE_NOTHROW( arch->set_collective_strategy_group() );
    REQUIRE_NOTHROW( env->set_processor_grouping(psqrt) );
    REQUIRE_NOTHROW( reduction = new product_reduction_kernel(local_value,global_value) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->nmessages==2*psqrt );
  }
}

TEST_CASE( "cg kernels","[cg][kernel][60]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution *blocked = new product_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new product_object(blocked) ),
    xnew = std::shared_ptr<object>( new product_object(blocked) ),
    z = std::shared_ptr<object>( new product_object(blocked) ),
    r = std::shared_ptr<object>( new product_object(blocked) ),
    rnew = std::shared_ptr<object>( new product_object(blocked) ),
    p = std::shared_ptr<object>( new product_object(blocked) ),
    pold = std::shared_ptr<object>( new product_object(blocked) ),
    q = std::shared_ptr<object>( new product_object(blocked) ),
    qold = std::shared_ptr<object>( new product_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new product_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new product_object(scalar) ), 
    rrp = std::shared_ptr<object>( new product_object(scalar) ),
    rnorm = std::shared_ptr<object>( new product_object(scalar) ),
    pap = std::shared_ptr<object>( new product_object(scalar) ),
    alpha = std::shared_ptr<object>( new product_object(scalar) ), 
    beta = std::shared_ptr<object>( new product_object(scalar) );
  double one = 1.;
  
  // the sparse matrix
  product_sparse_matrix *A;
  { 
    index_int globalsize = blocked->global_volume();
    A = new product_sparse_matrix(blocked);
    for (int row=blocked->first_index_r(mycoord); row<=blocked->last_index_r(mycoord); row++) {
      int col;
      col = row;     A->add_element(row,col,2.);
      col = row+1; if (col<globalsize)
		     A->add_element(row,col,-1.);
      col = row-1; if (col>=0)
		     A->add_element(row,col,-1.);
    }
  }
  
  SECTION( "r_norm squared" ) {
    kernel *r_norm;
    double
      *rdata = r->get_data(mycoord), *rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.;
    }
    REQUIRE_NOTHROW( r_norm = new product_normsquared_kernel( r,rnorm ) );
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( g==nlocal*ntids );
    CHECK( rrdata[0]==Approx(4*g) );
  }
  
  printf("strategy disabled\n"); return;
  SECTION( "r_norm1" ) {
    kernel *r_norm;
    double
      *rdata = r->get_data(mycoord), *rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.;
    }
    SECTION( "send/recv strategy" ) {
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( r_norm = new product_norm_kernel( r,rnorm ) );
    }
    SECTION( "grouping strategy" ) {
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = new product_norm_kernel( r,rnorm ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( rrdata[0]==Approx(2*sqrt((double)g)) );
  }
  
  SECTION( "r_norm2" ) {
    kernel *r_norm;
    double
      *rdata = r->get_data(mycoord), *rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = mytid*nlocal+i+1;
    }
    SECTION( "send/recv strategy" ) {
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( r_norm = new product_norm_kernel( r,rnorm ) );
    }
    SECTION( "grouping strategy" ) {
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = new product_norm_kernel( r,rnorm ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_volume();
    CHECK( pow(rrdata[0],2)==Approx(g*(g+1)*(2*g+1)/6.) );
  }
  
  SECTION( "rho_inprod" ) {
    const char *mode;
    product_innerproduct_kernel *rho_inprod;
    double
      *rdata = r->get_data(mycoord), *zdata = z->get_data(mycoord), *rrdata = rr->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.; zdata[i] = mytid*nlocal+i;
    }
    
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel uses send/recv";
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( rho_inprod = new product_innerproduct_kernel( r,z,rr ) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel uses grouping";
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( rho_inprod = new product_innerproduct_kernel( r,z,rr ) );
    }
    REQUIRE_NOTHROW( rho_inprod->analyze_dependencies() );
    REQUIRE_NOTHROW( rho_inprod->execute() );
    {
      kernel *prekernel;
      REQUIRE_NOTHROW( prekernel = rho_inprod->get_prekernel() );
      CHECK( prekernel->get_n_in_objects()==2 );
    }
    index_int g = r->global_volume();
    CHECK( rrdata[0]==Approx(g*(g-1)) );
  }
  
  SECTION( "copy" ) {
    kernel *rrcopy;
    int n;
    double *rdata,*sdata;
    
    SECTION( "scalar" ) {
      n = 1;
      rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata[i] = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = new product_copy_kernel( rr,rrp ) );
    }
    SECTION( "vector" ) {
      n = nlocal;
      rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata[i] = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = new product_copy_kernel( r,z ) );
    }
    
    REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
    REQUIRE_NOTHROW( rrcopy->execute() );
    for (int i=0; i<n; i++)
      CHECK( sdata[i]==Approx(2.*(mytid*n+i)) );
  }

  SECTION( "add" ) {
    kernel *sum,*makex,*makez;
    REQUIRE_NOTHROW( makex = new product_origin_kernel(x) );
    REQUIRE_NOTHROW( makez = new product_origin_kernel(z) );
    REQUIRE_NOTHROW( sum = new product_sum_kernel(x,z,xnew) );
    auto queue = std::shared_ptr<algorithm>( new product_algorithm(arch) );
    SECTION("the logical way") {
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makez) );
      REQUIRE_NOTHROW( queue->add_kernel(sum) );
    }
    SECTION("to be contrary") {
      REQUIRE_NOTHROW( queue->add_kernel(sum) );
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makez) );
    }
    {
      double *xdata = x->get_data(mycoord), *zdata = z->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) { xdata[i] = 1.; zdata[i] = 2.; }
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    {
      double *newdata = xnew->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
	CHECK( newdata[i]==Approx(3.) );
    }
  }
  
  SECTION( "scalar kernel error catching" ) {
    kernel *beta_calc;
    object *rr;
    REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new product_object(blocked) ) );
    REQUIRE_THROWS( beta_calc = new product_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_THROWS( beta_calc = new product_scalar_kernel( rrp,"/",rr,beta ) );
    REQUIRE_THROWS( beta_calc = new product_scalar_kernel( beta,"/",rrp,rr ) );
  }

  SECTION( "beta_calc" ) {
    kernel *beta_calc;
    CHECK_NOTHROW( rr->get_data(mycoord)[0] = 5. );
    CHECK_NOTHROW( rrp->get_data(mycoord)[0] = 4. );
    REQUIRE_NOTHROW( beta_calc = new product_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_NOTHROW( beta_calc->analyze_dependencies() );
    REQUIRE_NOTHROW( beta_calc->execute() );
    double *bd;
    REQUIRE_NOTHROW( bd = beta->get_data(mycoord) );
    CHECK( *bd==Approx(1.25) );
  }

  SECTION( "update" ) {
    double threeval = 3.;
    auto three = std::shared_ptr<object>( new product_object(scalar) );
    REQUIRE_NOTHROW( three->set_value(&threeval) );
    {
      double *threedata;
      REQUIRE_NOTHROW( threedata = three->get_data(mycoord) );
      CHECK( threedata[0]==Approx(threeval) );
    }
    kernel *pupdate;
    double
      *bdata = beta->get_data(mycoord),
      *zdata = z->get_data(mycoord),
      *odata = pold->get_data(mycoord),
      *pdata = p->get_data(mycoord);
    bdata[0] = 2.;
    for (int i=0; i<nlocal; i++) { // 3*2 , 2*7 : ++ = 20, -+ = 8, +- = -8, -- = -20
      zdata[i] = 2.; odata[i] = 7.;
    }
    SECTION( "pp test s1" ) {
      three = std::shared_ptr<object>( new product_object(blocked) );
      REQUIRE_THROWS( pupdate = new product_axbyz_kernel
		      ( '+',three,z, '+',beta,pold, p ) );
    }
    SECTION( "pp test s2" ) {
      beta = std::shared_ptr<object>( new product_object(blocked) );
      REQUIRE_THROWS( pupdate = new product_axbyz_kernel
		      ( '+',three,z, '+',beta,pold, p ) );
    }
    SECTION( "pp" ) {
      REQUIRE_NOTHROW( pupdate = new product_axbyz_kernel
		       ( '+',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      std::vector<dependency*> *deps;
      REQUIRE_NOTHROW( deps = pupdate->get_dependencies() );
      CHECK( deps->size()==4 );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(20.) );
    }
    SECTION( "mp" ) {
      REQUIRE_NOTHROW( pupdate = new product_axbyz_kernel
		       ( '-',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(8.) );
    }
    SECTION( "pm" ) {
      REQUIRE_NOTHROW( pupdate = new product_axbyz_kernel
		       ( '+',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-8.) );
    }
    SECTION( "mm" ) {
      REQUIRE_NOTHROW( pupdate = new product_axbyz_kernel
		       ( '-',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-20.) );
    }
  }

  SECTION( "matvec" ) {
    kernel *matvec;
    REQUIRE_NOTHROW( A = new product_sparse_matrix( blocked ) );

    int test;
    index_int
      my_first = blocked->first_index_r(mycoord),
      my_last = blocked->last_index_r(mycoord);
    SECTION( "diagonal matrix" ) {
      test = 1;
      for (index_int row=my_first; row<=my_last; row++) {
	REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
      }
    }
    SECTION( "threepoint matrix" ) {
      test = 2;
      index_int globalsize = blocked->global_volume();
      for (int row=my_first; row<=my_last; row++) {
	int col;
	col = row;     A->add_element(row,col,2.);
	col = row+1; if (col<globalsize)
		       A->add_element(row,col,-1.);
	col = row-1; if (col>=0)
		       A->add_element(row,col,-1.);
      }
    }

    double *pdata,*qdata;
    REQUIRE_NOTHROW( pdata = p->get_data(mycoord) );
    for (index_int row=0; row<blocked->local_size(mycoord); row++)
      pdata[row] = 3.;
    REQUIRE_NOTHROW( matvec = new product_spmvp_kernel( p,q,A ) );
    REQUIRE_NOTHROW( matvec->analyze_dependencies() );
    REQUIRE_NOTHROW( matvec->execute() );
    REQUIRE_NOTHROW( qdata = q->get_data(mycoord) );
    for (index_int row=my_first; row<my_last; row++) {
      index_int lrow = row-my_first;
      switch (test) {
      case 1: 
	CHECK( qdata[lrow]==Approx(6.) );
	break;
      case 2:
	if (row==0 || row==blocked->global_volume()-1)
	  CHECK( qdata[lrow]==Approx(3.) );
	else
	  CHECK( qdata[lrow]==Approx(0.) );
	break;
      }
    }
  }

  SECTION( "precon" ) {
    kernel *precon;
    REQUIRE_NOTHROW( precon = new product_preconditioning_kernel( r,z ) );
  }
}

TEST_CASE( "neuron kernel","[kernel][sparse][DAG][70]" ) {

  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  auto last_coordinate = domain_coordinate( std::vector<index_int>{localsize-1} );
  distribution *blocked =
    new product_block_distribution(decomp,g);
  auto
    a = std::shared_ptr<object>( new product_object(blocked) ),
    b = std::shared_ptr<object>( new product_object(blocked) ),
    c1 = std::shared_ptr<object>( new product_object(blocked) ), 
    c2 = std::shared_ptr<object>( new product_object(blocked) ),
    d = std::shared_ptr<object>( new product_object(blocked) );

  product_sparse_matrix
    *Anarrow = new product_sparse_matrix( blocked ),
    *Awide   = new product_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_volume(),
      my_first = blocked->first_index_r(mycoord),
      my_last = blocked->last_index_r(mycoord);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      // A narrow is tridiagonal
      col = row;     Anarrow->add_element(row,col,1.);
      col = row+1; if (col<globalsize) Anarrow->add_element(row,col,1.);
      col = row-1; if (col>=0)         Anarrow->add_element(row,col,1.);
      // A wide is distance 3 tridiagonal
      col = row;     Awide->add_element(row,col,1.);
      col = row+3; if (col<globalsize) Awide->add_element(row,col,1.);
      col = row-3; if (col>=0)         Awide->add_element(row,col,1.);
    }
  }
  kernel
    *make_input = new product_origin_kernel(a),
    *fast_mult1 = new product_spmvp_kernel(a,b,Anarrow),
    *fast_mult2 = new product_spmvp_kernel(b,c1,Anarrow),
    *slow_mult  = new product_spmvp_kernel(a,c2,Awide),
    *assemble   = new product_sum_kernel(c1,c2,d);

  auto queue = std::shared_ptr<algorithm>( new product_algorithm(arch) );
  CHECK_NOTHROW( queue->add_kernel(make_input) );
  CHECK_NOTHROW( queue->add_kernel(fast_mult1) );
  CHECK_NOTHROW( queue->add_kernel(fast_mult2) );
  CHECK_NOTHROW( queue->add_kernel(slow_mult) );
  CHECK_NOTHROW( queue->add_kernel(assemble) );

  SECTION( "kernel analysis" ) {
    CHECK_NOTHROW( make_input->analyze_dependencies() );
    CHECK_NOTHROW( fast_mult1->analyze_dependencies() );
    CHECK_NOTHROW( fast_mult2->analyze_dependencies() );
    CHECK_NOTHROW( slow_mult->analyze_dependencies() );
    CHECK_NOTHROW( assemble->analyze_dependencies() );
    return;
  }
  SECTION( "queue analysis" ) {
    CHECK_NOTHROW( queue->analyze_dependencies() );
  }
  
  SECTION( "strictly local" ) {
    int set = nlocal/2;
    {
      // check that we have enough space
      CHECK( a->local_size(mycoord)==nlocal );
      CHECK( (set-2)>0 );
      CHECK( (set+2)<(nlocal-1) );
      // set input vector to delta halfway each subdomain
      double *adata = a->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
	adata[i] = 0.;
      adata[set] = 1.;
    }
    CHECK_NOTHROW( queue->execute() );
    { // result of one narrow multiply
      INFO( "Anarrow" );
      double *data = b->get_data(mycoord); 
      for (index_int i=0; i<nlocal; i++) {
	INFO( "b i=" << i );
	if (i<set-1 || i>set+1)
	  CHECK( data[i]==Approx(0.) );
	else
	  CHECK( data[i]!=Approx(0.) );
      }
    }
    { // two narrow multiplies in a row
      INFO( "Anarrow^2" );
      double *data = c1->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) {
	INFO( "c1 i=" << i );
	if (i<set-2 || i>set+2)
	  CHECK( data[i]==Approx(0.) );
	else
	  CHECK( data[i]!=Approx(0.) );
      }
    }
    { // result of one wide multiply
      INFO( "Awide" );
      double *data = c2->get_data(mycoord); 
      for (index_int i=0; i<nlocal; i++) {
	INFO( "b i=" << i );
	if (i==set-3 || i==set+3 || i==set)
	  CHECK( data[i]!=Approx(0.) );
	else
	  CHECK( data[i]==Approx(0.) );
      }
    }
    { // adding it together
      double *data = d->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++) {
	INFO( "d i=" << i );
	if (i<set-3 || i>set+3)
	  CHECK( data[i]==Approx(0.) );
	else
	  CHECK( data[i]!=Approx(0.) );
      }
    }
  }
  SECTION( "spilling" ) {
    {
      double *adata = a->get_data(mycoord);
      for (index_int i=0; i<nlocal; i++)
  	adata[i] = 0.;
      if (mytid%2==1)
  	adata[0] = 1.;
    }
    CHECK_NOTHROW( queue->execute() );
    {
      double *data = d->get_data(mycoord);
      if (mytid%2==0) { // crud at the top
  	for (index_int i=0; i<nlocal; i++) {
  	  INFO( "i=" << i );
  	  if (i<nlocal-3)
  	    CHECK( data[i]==Approx(0.) );
  	  else
  	    CHECK( data[i]!=Approx(0.) );
  	}
      } else { // crud at the bottom
  	for (index_int i=0; i<nlocal; i++) {
  	  INFO( "i=" << i );
  	  if (i>3)
  	    CHECK( data[i]==Approx(0.) );
  	  else
  	    CHECK( data[i]!=Approx(0.) );
  	}
      }
    }
  }
}
#endif
