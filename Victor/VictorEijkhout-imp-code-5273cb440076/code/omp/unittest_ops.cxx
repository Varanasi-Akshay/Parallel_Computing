/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the OpenMP backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** application level operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_ops.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "Copy kernel, vector","[kernel][copy][31]" ) {

  int nglobal = 10;
  distribution *blocked;
  REQUIRE_NOTHROW( blocked = new omp_block_distribution(decomp,nglobal) );
  std::shared_ptr<object> x, xnew;
  REQUIRE_NOTHROW( x = std::shared_ptr<object>( new omp_object(blocked) ) );
  REQUIRE_NOTHROW( xnew = std::shared_ptr<object>( new omp_object(blocked) ) );
  auto
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) );
  REQUIRE_NOTHROW( r->allocate(); );
  
  kernel *rrcopy;
  omp_algorithm *queue = new omp_algorithm(decomp);
  const char *mode;
  std::shared_ptr<object> *in;
  
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    double *data; REQUIRE_NOTHROW( data = r->get_data(mycoord) );
    index_int f,l;
    REQUIRE_NOTHROW( f = r->first_index_r(mycoord)[0] );
    REQUIRE_NOTHROW( l = r->last_index_r(mycoord)[0] );
    for (int i=f; i<=l; i++)
      data[i] = 1.;
  }
  REQUIRE_NOTHROW( rrcopy = new omp_copy_kernel( r,z ) );
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );

  REQUIRE_NOTHROW( queue->add_kernel( rrcopy ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
  REQUIRE( queue->get_all_tasks_executed()==1 );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    double *data; REQUIRE_NOTHROW( data = z->get_data(mycoord) );
    index_int f,l;
    REQUIRE_NOTHROW( f = z->first_index_r(mycoord)[0] );
    REQUIRE_NOTHROW( l = z->last_index_r(mycoord)[0] );
    for (int i=f; i<=l; i++) {
      INFO( "result i: " << i );
      CHECK( data[i]==Approx(1.) );
    }
  }
}

TEST_CASE( "Copy kernel, scalar","[kernel][copy][32]" ) {

  // scalars, all redundantly replicated
  distribution *scalar = new omp_replicated_distribution(decomp);
  std::shared_ptr<object>
    rr,rrp,rnorm,pap,alpha,beta;
  REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new omp_object(scalar) ) );
  REQUIRE_NOTHROW( rr->allocate(); );
  REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar) ) );
  
  omp_algorithm *queue = new omp_algorithm(decomp);

  //CHECK( rr->global_allocation()==ntids );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    //CHECK( rr->local_allocation()==ntids );
    double *data; REQUIRE_NOTHROW( data = rr->get_data(mycoord) );
    index_int f = rr->first_index_r(mycoord)[0], l = rr->last_index_r(mycoord)[0];
    //CHECK( f==p );
    for (int i=f; i<=l; i++)
      data[i] = 1.;
  }
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(rr) ) );
  kernel *rrcopy;
  REQUIRE_NOTHROW( rrcopy = new omp_copy_kernel( rr,rrp ) );
  CHECK( rrcopy->has_type_compute() );
  REQUIRE_NOTHROW( queue->add_kernel( rrcopy ) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
  REQUIRE( queue->get_all_tasks_executed()==1 );

  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "p=" << mytid );
    double *data; REQUIRE_NOTHROW( data = rrp->get_data(mycoord) );
    index_int f = rrp->first_index_r(mycoord)[0], l = rrp->last_index_r(mycoord)[0];
    for (int i=f; i<=l; i++) {
      INFO( "result i: " << i );
      CHECK( data[i]==Approx(1.) );
    }
  }
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  int nlocal=12,nglobal=nlocal*ntids;
  auto no_op = ioperator("none");
  omp_distribution *block = 
    new omp_block_distribution(decomp,nglobal);
  double xdata[nglobal];
  auto
    xvector = std::shared_ptr<object>( new omp_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new omp_object(block) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = block->first_index_r(mycoord)[0], my_last = block->last_index_r(mycoord)[0];
    CHECK( my_first==mytid*nlocal );
    CHECK( my_last==(mytid+1)*nlocal-1 );
  }
  for (int i=0; i<nglobal; i++)
    xdata[i] = pointfunc33(i,0);
  omp_kernel *scale;
  double *halo_data,*ydata;

  omp_algorithm *queue;
  REQUIRE_NOTHROW( queue = new omp_algorithm(decomp) );
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(xvector) ) );
  const char *op,*mode;

  SECTION( "scale kernel" ) { op = "scale";
    SECTION( "kernel as kernel from double-star" ) {
      mode = "scale kernel with scalar";
      double x = 2;
      scale = new omp_scale_kernel(&x,xvector,yvector);
    }
    SECTION( "kernel as kernel from replicated object" ) {
      mode = "scale kernel with object";
      double xval = 2;
      distribution *scalar; REQUIRE_NOTHROW( scalar = new omp_replicated_distribution(decomp) );
      std::shared_ptr<object> x;
      REQUIRE_NOTHROW( x = std::shared_ptr<object>( new omp_object(scalar) ) );
      REQUIRE_NOTHROW( x->set_value(xval) );
      REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(x) ) );
      CHECK( scalar->has_type_replicated() );
      CHECK( x->has_type_replicated() );
      REQUIRE_NOTHROW( scale = new omp_scale_kernel(x,xvector,yvector) );
    }

    // std::shared_ptr<task> scale_task;
    // CHECK_NOTHROW( scale_task = scale->get_tasks()[0] );
    // CHECK_NOTHROW( scale->execute() );
    // CHECK_NOTHROW( halo_data = scale_task->get_halo_object(0)->get_data() );
    // CHECK_NOTHROW( ydata = yvector->get_data() );
    // {
    //   int i;
    //   for (i=0; i<nlocal; i++) {
    // 	CHECK( halo_data[i] == Approx( pointfunc33(i,my_first) ) );
    //   }
    // }
  }

  SECTION( "axpy kernel" ) { op = "axpy"; mode = "default";
    double x=2;
    scale = new omp_axpy_kernel(xvector,yvector,&x);
  }

  INFO( "using operation: " << op );
  INFO( "mode is " << mode );

  REQUIRE_NOTHROW( queue->add_kernel(scale) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
  REQUIRE( queue->get_all_tasks_executed()==1 );

  {
    int i;
    CHECK_NOTHROW( ydata = yvector->get_data(new processor_coordinate_zero(1)) );
    for (i=0; i<nglobal; i++) {
      INFO( "nlocal=" << nlocal << ", error in i=" << i );
      CHECK( xdata[i] == Approx( pointfunc33(i,0)) );
      CHECK( ydata[i] == Approx( 2*pointfunc33(i,0)) );
    }
  }
}

TEST_CASE( "Beta from sparse matrix","[beta][sparse][41]" ) {

  int localsize=20,gsize=localsize*ntids;
  omp_distribution *block;
  REQUIRE_NOTHROW( block = new omp_block_distribution(decomp,gsize) );
  std::shared_ptr<object> in_obj, out_obj;
  REQUIRE_NOTHROW( in_obj = std::shared_ptr<object>( new omp_object(block) ) );
  REQUIRE_NOTHROW( out_obj = std::shared_ptr<object>( new omp_object(block) ) );
  {
    double *indata = in_obj->get_data(new processor_coordinate_zero(1));
    for (index_int i=0; i<gsize; i++) indata[i] = .5;
  }
  omp_kernel *kern;
  REQUIRE_NOTHROW( kern = new omp_kernel(in_obj,out_obj) );
  kern->set_name("sparse-stuff");
  //  kern->set_localexecutefn( &local_sparse_matrix_vector_multiply );
  omp_sparse_matrix *pattern;

  SECTION( "connect right" ) {
    REQUIRE_NOTHROW( pattern = new omp_sparse_matrix(block,gsize) );
    for (index_int i=0; i<gsize; i++) {
      INFO( "creating row " << i << " out of " << gsize );
      REQUIRE_NOTHROW( pattern->add_element(i,i) );
      if (i+1<gsize) {
    	CHECK_NOTHROW( pattern->add_element(i,i+1) );
      } else {
    	REQUIRE_THROWS( pattern->add_element(i,i+1) );
      }
    }
    REQUIRE_NOTHROW( kern->last_dependency()->set_index_pattern(pattern) );
    REQUIRE_NOTHROW( kern->analyze_dependencies() );

    distribution *beta;
    REQUIRE_NOTHROW( beta = kern->last_dependency()->get_beta_distribution() );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      index_int
	my_first = block->first_index_r(mycoord)[0],
	my_last = block->last_index_r(mycoord)[0];
      std::shared_ptr<multi_indexstruct> pstruct,compare;
      REQUIRE_NOTHROW( pstruct = beta->get_processor_structure(mycoord) );
      if (mytid==ntids-1)
	REQUIRE_NOTHROW( compare = std::shared_ptr<multi_indexstruct>
			 ( new multi_indexstruct
			   ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(my_first,my_last) ) ))) ;
      else
	REQUIRE_NOTHROW( compare = std::shared_ptr<multi_indexstruct>
			 ( new multi_indexstruct
			   ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(my_first,my_last+1) ) )) );
      INFO( "beta struct <<" << pstruct->as_string()
	    << ">> s/b <<" << compare->as_string() << ">>" );
      CHECK( pstruct->equals(compare) );
    }

    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = kern->get_tasks() );
    for ( auto t : tasks ) {
    //for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
      std::vector<message*> rmsgs;
      REQUIRE_NOTHROW( rmsgs = t->get_receive_messages() );
      if (t->get_domain().coord(0)==ntids-1)
	CHECK( rmsgs.size()==1 );
      else
	CHECK( rmsgs.size()==2 );
      for ( auto m : rmsgs ) { 
    	//printf("msg: <<%s>>\n",(*m)->as_string().data());
      }
    }
    for ( auto t : tasks ) {
    //for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
      std::vector<message*> rmsgs;
      REQUIRE_NOTHROW( rmsgs = t->get_receive_messages() );
      for ( auto msg : rmsgs ) { //auto m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
    	if (msg->get_sender().equals(msg->get_receiver())) {
    	  CHECK( msg->volume()==localsize );
    	} else {
    	  CHECK( msg->volume()==1 );
    	}
      }
    }
  }

  // SECTION( "connect left" ) {
  //   REQUIRE_NOTHROW( pattern = new omp_sparse_matrix(block) );
  //   for (index_int i=0; i<gsize; i++) {
  //     REQUIRE_NOTHROW( pattern->add_element(i,i) );
  //     if (i-1>=0) {
  // 	CHECK_NOTHROW( pattern->add_element(i,i-1) );
  //     } else {
  // 	REQUIRE_THROWS( pattern->add_element(i,i-1) );
  //     }
  //   }
  //   REQUIRE_NOTHROW( kern->last_dependency()->set_index_pattern(pattern) );
  //   REQUIRE_NOTHROW( kern->analyze_dependencies() );

  //   std::vector<std::shared_ptr<task>> *tasks;
  //   REQUIRE_NOTHROW( tasks = kern->get_tasks() );
  //   for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
  //     std::vector<message*> *rmsgs;
  //     REQUIRE_NOTHROW( rmsgs = (*t)->get_receive_messages() );
  //     if ((*t)->get_domain().coord(0)==0) {
  // 	CHECK( rmsgs->size()==1 );
  //     } else {
  // 	CHECK( rmsgs->size()==2 );
  //     }
  //   }
  // }

  kern->set_localexecutefn
    ( [pattern] ( kernel_function_args ) -> void {
      return local_sparse_matrix_vector_multiply( kernel_function_call,(void*)pattern ); } );

}

TEST_CASE( "Actual sparse matrix","[beta][sparse][42]" ) {
  if (ntids==1) { printf("42 needs more than 1 proc\n"); return; }

  int localsize=20,gsize=localsize*ntids;
  omp_distribution *block = new omp_block_distribution(decomp,gsize);
  std::shared_ptr<object> in_obj = std::shared_ptr<object>( new omp_object(block) ),
    out_obj = std::shared_ptr<object>( new omp_object(block) );
  {
    REQUIRE_NOTHROW( in_obj->allocate() );
    double *indata; REQUIRE_NOTHROW( indata = in_obj->get_data(new processor_coordinate_zero(1)) );
    index_int n = in_obj->global_volume();
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }
  algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(decomp) );
  CHECK_NOTHROW( queue->add_kernel( new omp_origin_kernel(in_obj) ) );

  int ncols = 4;
  // initialize random
  srand((int)(ntids*(double)RAND_MAX/block->domains_volume()));

  // create a matrix with zero row sums
  //index_int mincol = block->first_index_r(mycoord)[0], maxcol = block->last_index_r(mycoord)[0];
  omp_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new omp_sparse_matrix(block) );
  indexstruct *my_columns = new contiguous_indexstruct(0,gsize-1);
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int my_first = block->first_index_r(mycoord)[0],my_last = block->last_index_r(mycoord)[0];
    for (index_int row=my_first; row<=my_last; row++) {
      int countcols=0;
      for (index_int ic=0; ic<ncols; ic++) {
	index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
	INFO( "icol=" << ic << ": (" << row << "," << col << ")" );
	CHECK( xs>=0 );
	CHECK( xs<localsize );
	col = MOD(my_last+1+xs,gsize);
	REQUIRE( col>=0 );
	REQUIRE( col<gsize );
	if (mytid<ntids-1)
	  REQUIRE( col>my_last );
	else
	  REQUIRE( col<my_first );
	REQUIRE( col!=row );
	int has; REQUIRE_NOTHROW( has = mat->has_element(row,col) );
	if (!has) {
	  REQUIRE_NOTHROW( mat->add_element(row,col,-1.) );
	  countcols++; }
	REQUIRE_NOTHROW( my_columns->add_element(col) );
	{ // just checking that my_columns is not altered
	  CHECK( my_columns->is_contiguous() );
	  CHECK( my_columns->first_index()==0 );
	  CHECK( my_columns->last_index()==gsize-1 );
	}
	// { // on the other hand, matrix columns should be growing
	//   indexstruct *mstruct = mat->all_columns();
	//   for (int prow=0; prow<=row; prow++)
	//     REQUIRE( mstruct->contains_element(prow) );
	// }
      }
      REQUIRE_NOTHROW( mat->add_element(row,row,(double)countcols+1.5) );
    }
  }

  { printf("[42] disabled equality test\n");
    // indexstruct *mstruct = mat->all_columns();
    // printf("going to test equality\n");
    // INFO( "comparing all_columns <<" << mstruct->as_string() <<
    // 	  ">> to accumulation <<" << my_columns->as_string() << ">>" )
    //   CHECK( mstruct->equals(my_columns) );
  }
  
  omp_kernel *spmvp = new omp_spmvp_kernel(in_obj,out_obj,mat);
  spmvp->set_name("omp-sparse-mvp");

  REQUIRE_NOTHROW( queue->add_kernel(spmvp) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
  CHECK( tasks.size()==ntids );
  for ( auto t : tasks ) {
  //for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = t->get_receive_messages() );
    for ( auto m : rmsgs ) { //auto m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
      int mytid;
      REQUIRE_NOTHROW( mytid = m->get_receiver().coord(0) );
      if (!m->get_sender().equals( m->get_receiver() ))
  	if (mytid==ntids-1)
  	  CHECK( m->get_sender().coord(0)==0 );
  	else 
  	  CHECK( m->get_sender().coord(0)==mytid+1 );
    }
  }

  REQUIRE_NOTHROW( queue->execute() );
  REQUIRE( queue->get_all_tasks_executed()==1 );

  INFO( "matrix: " << mat->contents_as_string() <<
	"\ninput: " << in_obj->values_as_string(processor_coordinate_zero(1)) <<
	"\noutput: " << out_obj->values_as_string(processor_coordinate_zero(1)) );
  {
    double *data;
    REQUIRE_NOTHROW( data = out_obj->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<gsize; i++) {
      INFO( "row " << i );
      CHECK( data[i] >= 1.5-1.e-5 ); //Approx(1.5) );
    }
  }
}

// TEST_CASE( "Sparse matrix kernel","[beta][sparse][43]" ) {
//   //  REQUIRE(ntids>1); // need at least two processors

//   int localsize=20,gsize=localsize*ntids;
//   omp_distribution *block = new omp_distribution(decomp,"disjoint-block",localsize,-1);
//   std::shared_ptr<object> in_obj = std::shared_ptr<object>( new omp_object(block) ), *out_obj = std::shared_ptr<object>( new omp_object(block) );
//   {
//     double *indata = in_obj->get_data(new processor_coordinate_zero(1)); index_int n = in_obj->local_size(mycoord);
//     for (index_int i=0; i<n; i++) indata[i] = 1.;
//   }

//   int ncols = 4;
//   index_int mincol = block->first_index_r(mycoord)[0], maxcol = block->last_index_r(mycoord)[0];
//   // initialize random
//   srand((int)(mytid*(double)RAND_MAX/block->domains_volume()));

//   // create a matrix with zero row sums
//   omp_sparse_matrix *mat = new omp_sparse_matrix(block);
//   index_int my_first = block->first_index_r(mycoord)[0],my_last = block->last_index_r(mycoord)[0];
//   for (index_int row=my_first; row<=my_last; row++) {
//     for (index_int ic=0; ic<ncols; ic++) {
//       index_int col, xs = (index_int) ( 1.*(localsize-1)*rand() / (double)RAND_MAX );
//       CHECK( xs>=0 );
//       CHECK( xs<localsize );
//       if (mytid<ntids-1) col = my_last+1+xs;
//       else               col = xs;
//       if (col<mincol) mincol = col;
//       if (col>maxcol) maxcol = col;
//       mat->add_element(row,col,-1.); // off elt
//     }
//     mat->add_element(row,row,(double)ncols+1.5); // diag elt
//   }

//   omp_kernel *spmvp = new omp_spmvp_kernel(in_obj,out_obj,mat);
//   REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

//   std::vector<std::shared_ptr<task>> tasks;
//   REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
//   CHECK( tasks->size()==1 );
//   std::shared_ptr<task> spmvptask;
//   REQUIRE_NOTHROW( spmvptask = (omp_task*)(*tasks)[0] );
//   std::vector<message*> *rmsgs;
//   REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
//   for (std::vector<message*>::iterator m=rmsgs->begin(); m!=rmsgs->end(); ++m) {
//     message *msg = (message*)(*m);
//     if (!msg->get_sender().equals(msg->get_receiver().coord(0)))
//       if (mytid==ntids-1)
// 	CHECK( msg->get_sender().coord(0)==0 );
//       else 
// 	CHECK( msg->get_sender().coord(0)==mytid+1 );
//   }

//   {
//     signature_function *beta;
//     REQUIRE_NOTHROW( beta = spmvp->get_beta_definition() );
//     parallel_indexstruct *structure;
//     REQUIRE_NOTHROW( structure = beta->derive_beta_structure(in_obj,out_obj) );
//     distribution *beta_dist;
//     REQUIRE_NOTHROW( beta_dist = new omp_distribution( env,structure ) );
//     indexstruct *column_indices;
//     REQUIRE_NOTHROW( column_indices = beta_dist->processor_structure(mytid) );
//     //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
//     CHECK( column_indices->is_sorted() ); 
//     REQUIRE_NOTHROW( mat->remap( beta_dist,mytid ) );
//   }

//   {
//     std::vector<std::shared_ptr<task>> tasks;
//     REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
//     CHECK( tasks->size()==1 );
//     std::shared_ptr<task> threetask;
//     REQUIRE_NOTHROW( threetask = tasks[0] );
//   }
//   REQUIRE_NOTHROW( spmvp->execute() );

//   {
//     double *data = out_obj->get_data(new processor_coordinate_zero(1));
//     index_int lsize = out_obj->local_size(mycoord);
//     for (index_int i=0; i<lsize; i++) {
//       CHECK( data[i] == Approx(1.5) );
//     }
//   }
// }

// TEST_CASE( "matrix kernel analysis","[kernel][44]" ) {
//   INFO( "mytid=" << mytid );
//   int nlocal = 10, g = ntids*nlocal;
//   distribution *blocked =
//     new omp_distribution(decomp,"disjoint-block",g);
//   omp_object
//     *x = std::shared_ptr<object>( new omp_object(blocked) ), *y = std::shared_ptr<object>( new omp_object(blocked) );

//   // set the matrix to one lower diagonal
//   omp_sparse_matrix
//     *Aup = new omp_sparse_matrix( blocked );
//   {
//     index_int
//       globalsize = blocked->global_volume(),
//       my_first = blocked->first_index_r(mycoord)[0],
//       my_last = blocked->last_index_r(mycoord)[0];
//     for (index_int row=my_first; row<=my_last; row++) {
//       int col;
//       // A narrow is tridiagonal
//       col = row;   Aup->add_element(row,col,1.);
//       col = row-1; if (col>=0) Aup->add_element(row,col,1.);
//     }
//   }

//   kernel *k;
//   REQUIRE_NOTHROW( k = new omp_spmvp_kernel( x,y,Aup ) );
//   REQUIRE_NOTHROW( k->analyze_dependencies() );

//   // analyze the message structure
//   auto tasks = k->get_tasks();
//   CHECK( tasks->size()==1 );
//   for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
//     if ((*t)->get_step()==x->get_object_number()) {
//       CHECK( (*t)->get_dependencies()->size()==0 );
//     } else {
//       auto send = (*t)->get_send_messages();
//       if (mytid==ntids-1)
// 	CHECK( send->size()==1 );
//       else
// 	CHECK( send->size()==2 );
//       auto recv = (*t)->get_receive_messages();
//       if (mytid==0)
// 	CHECK( recv->size()==1 );
//       else
// 	CHECK( recv->size()==2 );
//     }
//   }

//   SECTION( "limited to proc 0" ) {
//     // set the input vector to delta on the first element
//     {
//       double *d = x->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++) d[i] = 0.;
//       if (mytid==0) d[0] = 1.;
//     }

//     // check that we get a nicely propagating wave
//     REQUIRE_NOTHROW( k->execute() );
//     index_int my_first = blocked->first_index_r(mycoord)[0];
//     double *d = y->get_data(new processor_coordinate_zero(1));
//     for (index_int i=0; i<nlocal; i++) {
//       index_int g = my_first+i;
//       INFO( "global index " << g );
//       if (g<2) // s=0, g=0,1
// 	CHECK( d[i]!=Approx(0.) );
//       else
// 	CHECK( d[i]==Approx(0.) );
//     }
//   }
//   SECTION( "crossing over" ) {
//     // set the input vector to delta on the right edge
//     {
//       double *d = x->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++) d[i] = 0.;
//       d[nlocal-1] = 1.;
//     }

//     // check that we get a nicely propagating wave
//     REQUIRE_NOTHROW( k->execute() );
//     index_int my_first = blocked->first_index_r(mycoord)[0];
//     double *d = y->get_data(new processor_coordinate_zero(1));
//     for (index_int i=1; i<nlocal-1; i++)
//       CHECK( d[i]==Approx(0.) );
//     if (mytid==0)
//       CHECK( d[0]==Approx(0.) );
//     else 
//       CHECK( d[0]==Approx(1.) );
//     CHECK( d[nlocal-1]==Approx(1.) );
//   }
// }

// TEST_CASE( "matrix iteration shift left","[kernel][45]" ) {
//   INFO( "mytid=" << mytid );
//   int nlocal = 10, g = ntids*nlocal, nsteps = 2*nlocal+3;
//   distribution *blocked =
//     new omp_distribution(decomp,"disjoint-block",g);
//   omp_object
//     *x = std::shared_ptr<object>( new omp_object(blocked) ), **y;

//   // set the input vector to delta on the first element
//   {
//     double *d = x->get_data(new processor_coordinate_zero(1));
//     for (index_int i=0; i<nlocal; i++) d[i] = 0.;
//     if (mytid==0) d[0] = 1.;
//   }
  
//   y = std::vector<std::shared_ptr<object>[nsteps];
//   for (int i=0; i<nsteps; i++) {
//     y[i] = std::shared_ptr<object>( new omp_object(blocked) );
//   }

//   // set the matrix to one lower diagonal
//   omp_sparse_matrix
//     *Aup = new omp_sparse_matrix( blocked );
//   {
//     index_int
//       globalsize = blocked->global_volume(),
//       my_first = blocked->first_index_r(mycoord)[0],
//       my_last = blocked->last_index_r(mycoord)[0];
//     for (index_int row=my_first; row<=my_last; row++) {
//       int col;
//       // A narrow is tridiagonal
//       col = row;   Aup->add_element(row,col,1.);
//       col = row-1; if (col>=0) Aup->add_element(row,col,1.);
//     }
//   }

//   // make a queue
//   omp_algorithm *queue = new omp_algorithm(decomp);
//   REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(x) ) );
//   std::shared_ptr<object> inobj = x, outobj;
//   for (int i=0; i<nsteps; i++) {
//     outobj = y[i];
//     kernel *k;
//     REQUIRE_NOTHROW( k = new omp_spmvp_kernel( inobj,outobj,Aup ) );
//     REQUIRE_NOTHROW( queue->add_kernel(k) );
//     inobj = outobj;
//   }
//   REQUIRE_NOTHROW( queue->analyze_dependencies() );

//   // analyze the message structure
//   auto tasks = queue->get_tasks();
//   CHECK( tasks->size()==(nsteps+1) );
//   for (auto t=tasks->begin(); t!=tasks->end(); ++t) {
//     if ((*t)->get_step()==x->get_object_number()) {
//       CHECK( (*t)->get_dependencies()->size()==0 );
//     } else {
//       auto deps = (*t)->get_dependencies();
//       CHECK( deps->size()==1 );
//       dependency *dep = deps->at(0);

//       INFO( "step " << (*t)->get_step() );

//       auto recv = (*t)->get_receive_messages();
//       if (mytid==0)
// 	CHECK( recv->size()==1 );
//       else
// 	CHECK( recv->size()==2 );

//       auto send = (*t)->get_send_messages();
//       if (mytid==ntids-1)
// 	CHECK( send->size()==1 );
//       else
// 	CHECK( send->size()==2 );
//     }
//   }

//   // check that we get a nicely propagating wave
//   REQUIRE_NOTHROW( queue->execute() );
//   index_int my_first = blocked->first_index_r(mycoord)[0];
//   for (int s=0; s<nsteps; s++) {
//     INFO( "step " << s );
//     double *d = y[s]->get_data(new processor_coordinate_zero(1));
//     for (index_int i=0; i<nlocal; i++) {
//       index_int g = my_first+i;
//       INFO( "global index " << g );
//       if (g<s+2) // s=0, g=0,1
// 	CHECK( d[i]!=Approx(0.) );
//       else
// 	CHECK( d[i]==Approx(0.) );
//     }
//   }
// }

TEST_CASE( "special matrices","[kernel][spmvp][46]" ) {

  int nlocal = 10, g = ntids*nlocal, nsteps = g;
  distribution *blocked =
    new omp_block_distribution(decomp,g);
  sparse_matrix *A;

  SECTION( "lower diagonal" ) {
    const char *path;
    SECTION( "by element" ) { 
      // SECTION( "basic" ) { path = "by basic element";
      // 	REQUIRE_NOTHROW( A = new sparse_matrix( blocked->get_enclosing_structure() ) );
      // }
      SECTION( "derived" ) { path = "by derived element";
	REQUIRE_NOTHROW( A = new omp_sparse_matrix( blocked ) );
      }
      for (index_int i=0; i<g; i++)
	REQUIRE_NOTHROW( A->add_element(i,i,0.) );
      for (index_int i=1; i<g; i++)
	REQUIRE_NOTHROW( A->add_element(i,i-1,1.) );
    }
    SECTION( "by class" ) { path = "by class";
      REQUIRE_NOTHROW( A = new omp_lowerbidiagonal_matrix( blocked, 1,0 ) );
    }
    INFO( "matrix creation " << path );
    INFO( "lower bidiagonal: " << A->contents_as_string() );
    CHECK( A->nnzeros()==2*g-1 );

    nsteps = 1; // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    auto objs = std::vector<std::shared_ptr<object>>(g);
    for (int iobj=0; iobj<=nsteps; iobj++) {
      objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
      CHECK( objs[iobj]->domains_volume()==ntids );
    }
    REQUIRE_NOTHROW( objs[0]->allocate() );
    double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<g; i++) data[i] = 0.;
    data[0] = 1.;
    kernel *k;
    algorithm *queue = new omp_algorithm(decomp);
    REQUIRE_NOTHROW( k = new omp_origin_kernel(objs[0]) );
    k->set_name("1-in-first");
    REQUIRE_NOTHROW( queue->add_kernel( k ) );
    for (int istep=1; istep<=nsteps; istep++) {
      REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
      k->set_name( fmt::format("lo-bi-product-{}",istep) );
      REQUIRE_NOTHROW( queue->add_kernel(k) );
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    REQUIRE( queue->get_all_tasks_executed()==1 );
    auto p0 = new processor_coordinate_zero(1);
    INFO( "start: " << objs.at(0)->values_as_string(p0) );
    for (int istep=0; istep<=nsteps; istep++) {
      INFO( "object-" << istep << ": " << objs.at(istep)->values_as_string(p0) );
      REQUIRE_NOTHROW( data = objs[istep]->get_data(p0) );
      for (index_int i=0; i<g; i++) {
	INFO( "index " << i );
	if (i==istep) CHECK( data[i]==Approx(1.) );
	else          CHECK( data[i]==Approx(0.) );
      }
    }
  }

  SECTION( "upper diagonal" ) {
    REQUIRE_NOTHROW( A = new omp_upperbidiagonal_matrix( blocked, 0,1 ) );
    CHECK( A->nnzeros()==2*g-1 );

    auto objs = std::vector<std::shared_ptr<object>>(g);
    for (int iobj=0; iobj<g; iobj++)
      objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
    REQUIRE_NOTHROW( objs[0]->allocate() );
    double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<g; i++) data[i] = 0.;
    data[g-1] = 1.;
    algorithm *queue = new omp_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(objs[0]) ) );
    for (int istep=1; istep<g; istep++) {
      kernel *k;
      REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
      REQUIRE_NOTHROW( queue->add_kernel(k) );
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    REQUIRE( queue->get_all_tasks_executed()==1 );
    for (int istep=0; istep<g; istep++) {
      //	INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
      REQUIRE_NOTHROW( data = objs[istep]->get_data(new processor_coordinate_zero(1)) );
      for (index_int i=0; i<g; i++) {
	INFO( "index " << i );
	if (i==g-1-istep) CHECK( data[i]==Approx(1.) );
	else              CHECK( data[i]==Approx(0.) );
      }
    }
  }

  SECTION( "toeplitz" ) {
    REQUIRE_NOTHROW( A = new omp_toeplitz3_matrix( blocked, 0,2,0 ) );
    CHECK( A->nnzeros()==3*g-2 );

    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
  	objs[iobj] = std::shared_ptr<object>( new omp_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(new processor_coordinate_zero(1)) );
      for (index_int i=0; i<g; i++) data[i] = 1.;
      algorithm *queue = new omp_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(objs[0]) ) );
      int nsteps = 5;
      for (int istep=1; istep<nsteps; istep++) {
  	kernel *k;
  	REQUIRE_NOTHROW( k = new omp_spmvp_kernel(objs[istep-1],objs[istep],A) );
  	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      REQUIRE( queue->get_all_tasks_executed()==1 );
      REQUIRE_NOTHROW( data = objs[nsteps-1]->get_data(new processor_coordinate_zero(1)) );
      for (index_int i=0; i<g; i++) {
  	INFO( "index " << i );
  	CHECK( data[i]==Approx(pow(2,nsteps-1)) );
      }
    }
  }
}

// TEST_CASE( "compound kernel queue","[kernel][queue][50]" ) {

//   int nlocal = 50;
//   distribution
//     *block = new omp_distribution(decomp,"disjoint-block",nlocal*ntids),
//     *scalar = new omp_replicated_distribution(decomp);
//   auto
//     x = std::shared_ptr<object>( new omp_object(block) ),
//     y = std::shared_ptr<object>( new omp_object(block) ),
//     *xy = std::shared_ptr<object>( new omp_object(scalar) );
//   kernel
//     *makex = new omp_origin_kernel(x),
//     *makey = new omp_origin_kernel(y),
//     *prod = new omp_innerproduct_kernel(x,y,xy);
//   algorithm *queue = new omp_algorithm(decomp);
//   int inprod_step;

//   SECTION( "analyze in steps" ) {

//     SECTION( "kernels in logical order" ) {
//       REQUIRE_NOTHROW( queue->add_kernel(makex) );
//       REQUIRE_NOTHROW( queue->add_kernel(makey) );
//       REQUIRE_NOTHROW( queue->add_kernel(prod) );
//       inprod_step = 2;
//     }
  
//     SECTION( "kernels in wrong order" ) {
//       REQUIRE_NOTHROW( queue->add_kernel(prod) );
//       REQUIRE_NOTHROW( queue->add_kernel(makex) );
//       REQUIRE_NOTHROW( queue->add_kernel(makey) );
//       inprod_step = 0;
//     }
  
//     //CHECK( queue->get_step_count()==5 ); // 3 for innerproduct, 2 for origins

//     std::vector<std::shared_ptr<task>> tasks;
//     std::vector<kernel*> *kernels;
//     REQUIRE_NOTHROW( kernels = queue->get_kernels() );
//     CHECK( kernels->size()==3 );
//     for (int ik=0; ik<kernels->size(); ik++) {
//       kernel *k;
//       REQUIRE_NOTHROW( k = kernels->at(ik) );
//       REQUIRE_NOTHROW( k->analyze_dependencies() );
//       REQUIRE_NOTHROW( tasks = k->get_tasks() );
//       if (ik==inprod_step) 
// 	CHECK( tasks->size()==2 );
//       else 
// 	CHECK( tasks->size()==1 );
//       REQUIRE_NOTHROW( queue->add_kernel_tasks_to_queue(k) );
//     }
    
//     REQUIRE_NOTHROW( tasks = queue->get_tasks() );
//     for (std::vector<std::shared_ptr<task>>::iterator t=tasks->begin(); t!=tasks->end(); ++t) {
//       if (!(*t)->has_type_origin()) {
// 	std::shared_ptr<object> in; int inn; int ostep;
// 	CHECK_NOTHROW( in = (*t)->last_dependency()->get_in_object() );
// 	CHECK_NOTHROW( inn = in->get_object_number() );
// 	CHECK( inn>=0 );
//       }
//     }
//   }
//   SECTION( "single analyze call" ) {
//     REQUIRE_NOTHROW( queue->add_kernel(makex) );
//     REQUIRE_NOTHROW( queue->add_kernel(makey) );
//     REQUIRE_NOTHROW( queue->add_kernel(prod) );
//     REQUIRE_NOTHROW( queue->analyze_dependencies() );
//     inprod_step = 2;
//   }
//   std::vector<std::shared_ptr<task>> tsks;
//   REQUIRE_NOTHROW( tsks = queue->get_tasks() );
//   for (std::vector<std::shared_ptr<task>>::iterator t=tsks->begin(); t!=tsks->end(); ++t) {
//     int ik;
//     REQUIRE_NOTHROW( ik = (*t)->get_step() );
//     if (ik==inprod_step) {
//       REQUIRE_NOTHROW( (*t)->get_n_in_objects()==2 );
//     } else {
//       REQUIRE_NOTHROW( (*t)->get_n_in_objects()==0 );
//     }
//   }
// }

TEST_CASE( "cg update kernels","[kernel][cg][update][65]" ) {

  int nlocal = 1000, nglobal = nlocal*ntids;
  distribution *blocked = new omp_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new omp_object(blocked) ), 
    xnew = std::shared_ptr<object>( new omp_object(blocked) ),
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) ), 
    rnew = std::shared_ptr<object>( new omp_object(blocked) ),
    p = std::shared_ptr<object>( new omp_object(blocked) ), 
    pold = std::shared_ptr<object>( new omp_object(blocked) ),
    q = std::shared_ptr<object>( new omp_object(blocked) ), 
    qold = std::shared_ptr<object>( new omp_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new omp_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new omp_object(scalar) ), 
    rrp = std::shared_ptr<object>( new omp_object(scalar) ),
    rnorm = std::shared_ptr<object>( new omp_object(scalar) ),
    pap = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) );
  double one = 1.;
  rr->set_name("rr65"); rrp->set_name("rrp65");
  rnorm->set_name("rnorm65"); pap->set_name("pap65");
  alpha->set_name("alpha65"); 
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );

  double threeval = 3., betaval = 2.;
  auto three = std::shared_ptr<object>( new omp_object(scalar) ); three->set_name("three65"); 
  auto beta = std::shared_ptr<object>( new omp_object(scalar) ); beta->set_name("beta65");
  REQUIRE_NOTHROW( three->allocate() ); REQUIRE_NOTHROW( beta->allocate() );
  REQUIRE_NOTHROW( three->set_value(threeval) );
  REQUIRE_NOTHROW( beta->set_value(betaval) );  
  {
    double *data;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      REQUIRE_NOTHROW( data = three->get_data(mycoord) );
      CHECK( data[0]==Approx(threeval) );
      REQUIRE_NOTHROW( data = beta->get_data(mycoord) );
      CHECK( data[0]==Approx(betaval) );
    }
  }
  
  kernel *pupdate;
  auto
    zdata = z->get_raw_data(),
    odata = pold->get_raw_data(),
    pdata = p->get_raw_data();
  for (int i=0; i<nglobal; i++) { // 3*2 , 2*7 : ++ = 20, -+ = 8, +- = -8, -- = -20
    zdata[i] = 2.; odata[i] = 7.;
  }
  algorithm *update = new omp_algorithm(decomp);
  update->add_kernel( new omp_origin_kernel(three) );
  update->add_kernel( new omp_origin_kernel(z) );
  update->add_kernel( new omp_origin_kernel(beta) );
  update->add_kernel( new omp_origin_kernel(pold) );
  SECTION( "pp test s1" ) {
    auto
      three = std::shared_ptr<object>( new omp_object(blocked) ); three->set_name("three-blocked-WRONG");
    REQUIRE_THROWS( pupdate = new omp_axbyz_kernel
		    ( '+',three,z, '+',beta,pold, p ) );
  }
  SECTION( "pp test s2" ) {
    auto 
      beta = std::shared_ptr<object>( new omp_object(blocked) ); beta->set_name("beta-blocked-WRONG");
    REQUIRE_THROWS( pupdate = new omp_axbyz_kernel
		    ( '+',three,z, '+',beta,pold, p ) );
  }
  SECTION( "pp" ) { // 3*2 + 2*7 = 20
    REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel 
		     ( '+',three,z, '+',beta,pold, p ) );
    update->add_kernel(pupdate );
    REQUIRE_NOTHROW( update->analyze_dependencies() );
    std::vector<dependency*> deps;
    REQUIRE_NOTHROW( deps = pupdate->get_dependencies() );
    CHECK( deps.size()==4 );
    std::shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = pupdate->get_tasks().at(0) );
    auto msgs = tsk->get_receive_messages();
    CHECK( msgs.size()==4 );
    for ( auto msg : msgs ) {
      INFO( "message: " << msg->as_string() );
      std::shared_ptr<multi_indexstruct> global,local;
      REQUIRE_NOTHROW( global = msg->get_global_struct() );
      REQUIRE_NOTHROW( local = msg->get_local_struct() );
      index_int siz; REQUIRE_NOTHROW( siz = global->volume() );
      CHECK( siz==local->volume() );
      if (siz==1) {
      } else {
	CHECK( siz==nlocal ); // check on this one
      }
    }
    REQUIRE_NOTHROW( update->execute() );
    for (int i=0; i<nglobal; i++) {
      INFO( "i=" << i );
      CHECK( pdata[i]==Approx(20.) );
    }
  }
  SECTION( "mp" ) {
    REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		     ( '-',three,z, '+',beta,pold, p ) );
    update->add_kernel(pupdate);
    REQUIRE_NOTHROW( update->analyze_dependencies() );
    REQUIRE_NOTHROW( update->execute() );
    for (int i=0; i<nglobal; i++)
      CHECK( pdata[i]==Approx(8.) );
  }
  SECTION( "pm" ) {
    REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		     ( '+',three,z, '-',beta,pold, p ) );
    update->add_kernel(pupdate);
    REQUIRE_NOTHROW( update->analyze_dependencies() );
    REQUIRE_NOTHROW( update->execute() );
    for (int i=0; i<nglobal; i++)
      CHECK( pdata[i]==Approx(-8.) );
  }
  SECTION( "mm" ) {
    REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		     ( '-',three,z, '-',beta,pold, p ) );
    update->add_kernel(pupdate);
    REQUIRE_NOTHROW( update->analyze_dependencies() );
    REQUIRE_NOTHROW( update->execute() );
    for (int i=0; i<nglobal; i++)
      CHECK( pdata[i]==Approx(-20.) );
  }

}

TEST_CASE( "cg vector kernels, parallel","[cg][kernel][axbyz][60]" ) {

  int nlocal = 1000;
  distribution *blocked = new omp_block_distribution(decomp,ntids*nlocal);
  index_int globalsize = blocked->global_volume();
  auto
    x = std::shared_ptr<object>( new omp_object(blocked) ), 
    xnew = std::shared_ptr<object>( new omp_object(blocked) ),
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) ), 
    rnew = std::shared_ptr<object>( new omp_object(blocked) ),
    p = std::shared_ptr<object>( new omp_object(blocked) ), 
    pold = std::shared_ptr<object>( new omp_object(blocked) ),
    q = std::shared_ptr<object>( new omp_object(blocked) ), 
    qold = std::shared_ptr<object>( new omp_object(blocked) );
  x->set_name("x"); xnew->set_name("xnew");
  r->set_name("r"); rnew->set_name("rnew");
  p->set_name("p"); pold->set_name("pold");
  q->set_name("q"); qold->set_name("qold");
  z->set_name("z");  

  // scalars, all redundantly replicated
  distribution *scalar;
  REQUIRE_NOTHROW( scalar = new omp_replicated_distribution(decomp) );
  std::shared_ptr<object>
    rr,rrp;
  double one = 1.;
  REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new omp_object(scalar) ) );
  REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar) ) );
  rr->set_name("rr"); rrp->set_name("rrp");

  auto
    rnorm = std::shared_ptr<object>( new omp_object(scalar) ),
    pap = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) ), 
    beta = std::shared_ptr<object>( new omp_object(scalar) );

  algorithm *queue = new omp_algorithm(decomp);

  SECTION( "copy vectors" ) {
    kernel *rrcopy;
    double *rdata,*sdata;
    
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );
    rdata = r->get_data(new processor_coordinate_zero(1)); 
    for (int i=0; i<globalsize; i++)
      rdata[i] = 2.*i;
    REQUIRE_NOTHROW( rrcopy = new omp_copy_kernel( r,z ) );

    REQUIRE_NOTHROW( queue->add_kernel(rrcopy) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );

    sdata = z->get_data(new processor_coordinate_zero(1));
    for (int i=0; i<globalsize; i++)
      CHECK( sdata[i]==Approx(2.*i) );
  }

  SECTION( "copy scalars" ) {
    kernel *rrcopy;
    int n;
    double *rdata,*sdata;
    
    n = 1; 
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(rr) ) );
    REQUIRE_NOTHROW( rr->allocate() );
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord = decomp->coordinate_from_linear(mytid);
      rdata = rr->get_data(mycoord); 
      index_int
  	my_first = rr->first_index_r(mycoord)[0],
  	my_last = rr->last_index_r(mycoord)[0];
      for (int i=my_first; i<my_first+n; i++)
  	rdata[i] = 2.*(mytid*n+i);
    }
    REQUIRE_NOTHROW( rrcopy = new omp_copy_kernel( rr,rrp ) );

    REQUIRE_NOTHROW( queue->add_kernel(rrcopy) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );

    for (int mytid=0; mytid<ntids; mytid++) {
      INFO( "result for mytid=" << mytid );
      processor_coordinate mycoord = decomp->coordinate_from_linear(mytid);
      sdata = rrp->get_data(mycoord);
      index_int
  	my_first = rrp->first_index_r(mycoord)[0],
  	my_last = rrp->last_index_r(mycoord)[0];
      for (int i=my_first; i<my_first+n; i++)
  	CHECK( sdata[i]==Approx(2.*(mytid*n+i)) );
    }
  }

  SECTION( "add" ) {
    kernel *sum,*makex,*makez;
    //    index_int globalsize = x->global_volume();
    REQUIRE_NOTHROW( makex = new omp_origin_kernel(x) );
    REQUIRE_NOTHROW( makez = new omp_origin_kernel(z) );
    REQUIRE_NOTHROW( sum = new omp_sum_kernel(x,z,xnew) );
    algorithm *queue = new omp_algorithm(decomp);
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
      double *xdata = x->get_data(new processor_coordinate_zero(1)), *zdata = z->get_data(new processor_coordinate_zero(1));
      CHECK( x->global_volume()==globalsize );
      CHECK( z->global_volume()==globalsize );
      for (index_int i=0; i<globalsize; i++) { xdata[i] = 1.; zdata[i] = 2.; }
    }
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    {
      double *newdata = xnew->get_data(new processor_coordinate_zero(1));
      CHECK( xnew->global_volume()==globalsize );
      for (index_int i=0; i<globalsize; i++)
      	CHECK( newdata[i]==Approx(3.) );
    }
  }
  
  SECTION( "scalar kernel error catching" ) {
    kernel *beta_calc;
    std::shared_ptr<object> rr;
    REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new omp_object(blocked) ) );
    REQUIRE_THROWS( beta_calc = new omp_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_THROWS( beta_calc = new omp_scalar_kernel( rrp,"/",rr,beta ) );
    REQUIRE_THROWS( beta_calc = new omp_scalar_kernel( beta,"/",rrp,rr ) );
  }

  SECTION( "beta_calc" ) {
    kernel *beta_calc;
    for (int p=0; p<ntids; p++) {
      CHECK_NOTHROW( rr->get_data(new processor_coordinate_zero(1))[p] = 5. );
      CHECK_NOTHROW( rrp->get_data(new processor_coordinate_zero(1))[p] = 4. );
    }
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(rr) ) );
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(rrp) ) );
    REQUIRE_NOTHROW( beta_calc = new omp_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_NOTHROW( queue->add_kernel(beta_calc) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    double *bd;
    for (int p=0; p<ntids; p++) {
      REQUIRE_NOTHROW( bd = beta->get_data(decomp->coordinate_from_linear(p)) );
      CHECK( *bd==Approx(1.25) );
    }
  }

  SECTION( "update" ) {
    double threeval = 3.;
    auto three = std::shared_ptr<object>( new omp_object(scalar) );
    REQUIRE_NOTHROW( three->set_value(threeval) );
    {
      double *threedata;
      for (int p=0; p<ntids; p++) {
	REQUIRE_NOTHROW( threedata = three->get_data(decomp->coordinate_from_linear(p)) );
	CHECK( threedata[0]==Approx(threeval) );
      }
    }
    kernel *pupdate;
    double
      *bdata = beta->get_data(new processor_coordinate_zero(1)),
      *zdata = z->get_data(new processor_coordinate_zero(1)),
      *odata = pold->get_data(new processor_coordinate_zero(1)),
      *pdata = p->get_data(new processor_coordinate_zero(1));
    bdata[0] = 2.;
    for (int i=0; i<nlocal; i++) { // 3*2 , 2*7 : ++ = 20, -+ = 8, +- = -8, -- = -20
      zdata[i] = 2.; odata[i] = 7.;
    }
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(three) ) );
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(z) ) );
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(beta) ) );
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(pold) ) );
    SECTION( "pp" ) {
      REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		       ( '+',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( queue->add_kernel(pupdate) );
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(20.) );
    }
    SECTION( "mp" ) {
      REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		       ( '-',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( queue->add_kernel(pupdate) );
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(8.) );
    }
    SECTION( "pm" ) {
      REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		       ( '+',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( queue->add_kernel(pupdate) );
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-8.) );
    }
    SECTION( "mm" ) {
      REQUIRE_NOTHROW( pupdate = new omp_axbyz_kernel
		       ( '-',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( queue->add_kernel(pupdate) );
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-20.) );
    }
  }

}

TEST_CASE( "cg vector kernels, norm-like","[cg][kernel][norm][inprod][61]" ) {

  int nlocal = 1000;
  distribution *blocked = new omp_block_distribution(decomp,ntids*nlocal);
  index_int globalsize = blocked->global_volume();
  auto
    x = std::shared_ptr<object>( new omp_object(blocked) ), 
    xnew = std::shared_ptr<object>( new omp_object(blocked) ),
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) ), 
    rnew = std::shared_ptr<object>( new omp_object(blocked) ),
    p = std::shared_ptr<object>( new omp_object(blocked) ), 
    pold = std::shared_ptr<object>( new omp_object(blocked) ),
    q = std::shared_ptr<object>( new omp_object(blocked) ), 
    qold = std::shared_ptr<object>( new omp_object(blocked) );
  x->set_name("x"); xnew->set_name("xnew");
  r->set_name("r"); rnew->set_name("rnew");
  p->set_name("p"); pold->set_name("pold");
  q->set_name("q"); qold->set_name("qold");
  z->set_name("z");  

  // scalars, all redundantly replicated
  distribution *scalar;
  REQUIRE_NOTHROW( scalar = new omp_replicated_distribution(decomp) );
  std::shared_ptr<object>
    rr,rrp;
  double one = 1.;
  REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new omp_object(scalar) ) );
  REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar) ) );
  rr->set_name("rr"); rrp->set_name("rrp");

  auto
    rnorm = std::shared_ptr<object>( new omp_object(scalar) ),
    pap = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) ),
    beta = std::shared_ptr<object>( new omp_object(scalar) );

  algorithm *queue = new omp_algorithm(decomp);

  SECTION( "r_norm squared" ) {
    INFO( "r norm squared" );
    kernel *r_norm;
    REQUIRE_NOTHROW( r->allocate() );
    REQUIRE_NOTHROW( rnorm->allocate() );
    double *rdata,  *rrdata;
    REQUIRE_NOTHROW( rdata = r->get_raw_data() );
    REQUIRE_NOTHROW( rrdata = rnorm->get_raw_data() );
    for (int i=0; i<globalsize; i++) {
      rdata[i] = 2.;
    }
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );
    REQUIRE_NOTHROW( r_norm = new omp_normsquared_kernel( r,rnorm ) );
    REQUIRE_NOTHROW( queue->add_kernel(r_norm) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    // buncha tests of the halo integrity; wish we could test the local function
    auto tasks = queue->get_tasks();
    for ( auto t : tasks ) { //=tasks->begin(); t!=tasks->end(); ++t) {
      if (t->get_out_object()->get_object_number()==rnorm->get_object_number()) {
	std::vector<message*> msgs;
	REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
	CHECK( msgs.size()==ntids );
	for ( auto m : msgs ) { //auto m=msgs->begin(); m!=msgs->end(); ++m) {
	  int s = m->get_sender().coord(0);
	  auto 
	    fromstruct = m->get_global_struct(), tostruct = m->get_local_struct();
	  //printf("msg from %d global struct <<%s>>\n",s,fromstruct->as_string().data() );
	  CHECK( fromstruct->get_component(0)->equals( std::shared_ptr<indexstruct>( new contiguous_indexstruct(s)) ) );
	  CHECK( tostruct->get_component(0)->equals( std::shared_ptr<indexstruct>( new contiguous_indexstruct(s)) ) );
	}
	auto beta = t->last_dependency()->get_beta_object();
	CHECK( beta->has_type_replicated() );
	for (int mytid=0; mytid<ntids; mytid++) {
	  processor_coordinate mycoord;
	  REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
	  CHECK( beta->volume(mycoord)==ntids );
	}
      }
    }
    REQUIRE_NOTHROW( queue->execute() );
    REQUIRE( queue->get_all_tasks_executed()==1 );
    index_int g = r->global_volume();
    CHECK( rrdata[0]==Approx(4*g) );
  }
  
  SECTION( "r_norm1" ) {
    INFO( "r norm 1" );

    REQUIRE_NOTHROW( r->allocate() );
    REQUIRE_NOTHROW( rnorm->allocate() );
    double *rdata,  *rrdata;
    REQUIRE_NOTHROW( rdata = r->get_raw_data() );
    REQUIRE_NOTHROW( rrdata = rnorm->get_raw_data() );
    for (int i=0; i<globalsize; i++) {
      rdata[i] = 2.;
    }
    kernel *r_norm;
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );
    REQUIRE_NOTHROW( r_norm = new omp_norm_kernel( r,rnorm ) );
    REQUIRE_NOTHROW( queue->add_kernel(r_norm) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    REQUIRE( queue->get_all_tasks_executed()==1 );
    CHECK( rrdata[0]==Approx(2*sqrt(globalsize)) );
  }
  
  SECTION( "r_norm2" ) {
    INFO( "r norm 2" );
    kernel *r_norm;
    double
      *rdata = r->get_data(new processor_coordinate_zero(1)), *rrdata = rnorm->get_data(new processor_coordinate_zero(1));
    for (int i=0; i<globalsize; i++) {
      rdata[i] = i+1;
    }
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );
    REQUIRE_NOTHROW( r_norm = new omp_norm_kernel( r,rnorm ) );
    REQUIRE_NOTHROW( queue->add_kernel(r_norm) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    REQUIRE( queue->get_all_tasks_executed()==1 );
    index_int g = r->global_volume();
    CHECK( rrdata[0]==Approx(sqrt(g*(g+1)*(2*g+1)/6.)) );
  }
  
  SECTION( "rho_inprod" ) {
    INFO( "rho inprod" );
    omp_innerproduct_kernel *rho_inprod;
    double
      *rdata = r->get_data(new processor_coordinate_zero(1)), *zdata = z->get_data(new processor_coordinate_zero(1)), *rrdata = rr->get_data(new processor_coordinate_zero(1));
    for (int i=0; i<globalsize; i++) {
      rdata[i] = 2.; zdata[i] = i;
    }
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(r) ) );
    REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(z) ) );
    REQUIRE_NOTHROW( rho_inprod = new omp_innerproduct_kernel( r,z,rr ) );
    REQUIRE_NOTHROW( queue->add_kernel(rho_inprod) );
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
    {
      kernel *prekernel;
      REQUIRE_NOTHROW( prekernel = rho_inprod->get_prekernel() );
      CHECK( prekernel->get_n_in_objects()==2 );
    }
    index_int g = r->global_volume();
    CHECK( rrdata[0]==Approx(g*(g-1)) );
  }
  
}

TEST_CASE( "cg matrix kernels","[cg][sparse][kernel][65]" ) {

  int nlocal = 1000;
  distribution *blocked = new omp_block_distribution(decomp,ntids*nlocal);
  index_int globalsize = blocked->global_volume();
  auto
    x = std::shared_ptr<object>( new omp_object(blocked) ),
    xnew = std::shared_ptr<object>( new omp_object(blocked) ),
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) ),
    rnew = std::shared_ptr<object>( new omp_object(blocked) ),
    p = std::shared_ptr<object>( new omp_object(blocked) ),
    pold = std::shared_ptr<object>( new omp_object(blocked) ),
    q = std::shared_ptr<object>( new omp_object(blocked) ), 
    qold = std::shared_ptr<object>( new omp_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar;
  REQUIRE_NOTHROW( scalar = new omp_replicated_distribution(decomp) );
  std::shared_ptr<object>
    rr,rrp;
  double one = 1.;
  REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new omp_object(scalar) ) );
  REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar) ) );
  auto
    rnorm = std::shared_ptr<object>( new omp_object(scalar) ),
    pap = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) ),
    beta = std::shared_ptr<object>( new omp_object(scalar) );

  // the sparse matrix
  omp_sparse_matrix *A;
  REQUIRE_NOTHROW( A = new omp_sparse_matrix(blocked) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = blocked->first_index_r(mycoord)[0],
      my_last = blocked->last_index_r(mycoord)[0];
    for (int row=blocked->first_index_r(mycoord)[0]; row<=blocked->last_index_r(mycoord)[0]; row++) {
      int col;
      col = row;     REQUIRE_NOTHROW( A->add_element(row,col,2.) );
      col = row+1; if (col<globalsize)
		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
      col = row-1; if (col>=0)
		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
    }
  }
  
  // algorithm queue
  algorithm *queue = new omp_algorithm(decomp);

  // origin kernel
  double *pdata,*qdata;
  REQUIRE_NOTHROW( p->allocate() );
  REQUIRE_NOTHROW( pdata = p->get_data(new processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int f=p->first_index_r(mycoord)[0], l=p->last_index_r(mycoord)[0];
    for (index_int row=f; row<=l; row++)
      pdata[row] = 3.;
  }
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(p) ) );

  // make a matrix
  REQUIRE_NOTHROW( A = new omp_sparse_matrix( blocked ) );
  int test; const char *path;
  SECTION( "diagonal matrix" ) { path = "diagonal";
    test = 1;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      index_int
	my_first = blocked->first_index_r(mycoord)[0],
	my_last = blocked->last_index_r(mycoord)[0];
      for (index_int row=my_first; row<=my_last; row++) {
	REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
      }
    }
  }
  SECTION( "threepoint matrix" ) { path = "tridiagonal";
    test = 2;
    index_int globalsize = blocked->global_volume();
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      index_int
  	my_first = blocked->first_index_r(mycoord)[0],
  	my_last = blocked->last_index_r(mycoord)[0];
      for (int row=my_first; row<=my_last; row++) {
  	int col;
  	col = row;     A->add_element(row,col,2.);
  	col = row+1; if (col<globalsize)
  		       A->add_element(row,col,-1.);
  	col = row-1; if (col>=0)
  		       A->add_element(row,col,-1.);
      }
    }
  }
  INFO( path << " matrix" );

  kernel *matvec;
  REQUIRE_NOTHROW( matvec = new omp_spmvp_kernel( p,q,A ) );
  REQUIRE_NOTHROW( queue->add_kernel(matvec) );    

  // analyze

  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  {
    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = matvec->get_tasks() );
    CHECK( tasks.size()==ntids );
  }

  // execute
  REQUIRE_NOTHROW( queue->execute() );

  REQUIRE_NOTHROW( qdata = q->get_data(new processor_coordinate_zero(1)) );
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    index_int
      my_first = blocked->first_index_r(mycoord)[0],
      my_last = blocked->last_index_r(mycoord)[0];
    for (index_int row=my_first; row<=my_last; row++) {
      switch (test) {
      case 1: 
	CHECK( qdata[row]==Approx(6.) );
	break;
      case 2:
	if (row==0 || row==blocked->global_volume()-1)
	  CHECK( qdata[row]==Approx(3.) );
  	else
  	  CHECK( qdata[row]==Approx(0.) );
	break;
      }
    }
  }

  // SECTION( "precon" ) {
  //   kernel *precon;
  //   REQUIRE_NOTHROW( precon = new omp_preconditioning_kernel( r,z ) );
  // }
}

// TEST_CASE( "neuron kernel","[kernel][DAG][70]" ) {

//   INFO( "mytid=" << mytid );
//   int nlocal = 10, g = ntids*nlocal;
//   distribution *blocked =
//     new omp_distribution(decomp,"disjoint-block",g);
//   auto
//     a = std::shared_ptr<object>( new omp_object(blocked) ),
//     b = std::shared_ptr<object>( new omp_object(blocked) ),
//     c1 = std::shared_ptr<object>( new omp_object(blocked) ),
//     c2 = std::shared_ptr<object>( new omp_object(blocked) ),
//     d = std::shared_ptr<object>( new omp_object(blocked) );

//   omp_sparse_matrix
//     *Anarrow = new omp_sparse_matrix( blocked ),
//     *Awide   = new omp_sparse_matrix( blocked );
//   {
//     index_int
//       globalsize = blocked->global_volume(),
//       my_first = blocked->first_index_r(mycoord)[0],
//       my_last = blocked->last_index_r(mycoord)[0];
//     for (index_int row=my_first; row<=my_last; row++) {
//       int col;
//       // A narrow is tridiagonal
//       col = row;     Anarrow->add_element(row,col,1.);
//       col = row+1; if (col<globalsize) Anarrow->add_element(row,col,1.);
//       col = row-1; if (col>=0)         Anarrow->add_element(row,col,1.);
//       // A wide is distance 3 tridiagonal
//       col = row;     Awide->add_element(row,col,1.);
//       col = row+3; if (col<globalsize) Awide->add_element(row,col,1.);
//       col = row-3; if (col>=0)         Awide->add_element(row,col,1.);
//     }
//   }
//   kernel
//     *make_input = new omp_origin_kernel(a),
//     *fast_mult1 = new omp_spmvp_kernel(a,b,Anarrow),
//     *fast_mult2 = new omp_spmvp_kernel(b,c1,Anarrow),
//     *slow_mult  = new omp_spmvp_kernel(a,c2,Awide),
//     *assemble   = new omp_sum_kernel(c1,c2,d);

//   algorithm *queue = new omp_algorithm(decomp);
//   CHECK_NOTHROW( queue->add_kernel(make_input) );
//   CHECK_NOTHROW( queue->add_kernel(fast_mult1) );
//   CHECK_NOTHROW( queue->add_kernel(fast_mult2) );
//   CHECK_NOTHROW( queue->add_kernel(slow_mult) );
//   CHECK_NOTHROW( queue->add_kernel(assemble) );

//   CHECK_NOTHROW( queue->analyze_dependencies() );

//   SECTION( "strictly local" ) {
//     int set = nlocal/2;
//     {
//       // check that we have enough space
//       CHECK( a->local_size(mycoord)==nlocal );
//       CHECK( (set-2)>0 );
//       CHECK( (set+2)<(nlocal-1) );
//       // set input vector to delta halfway each subdomain
//       double *adata = a->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++)
// 	adata[i] = 0.;
//       adata[set] = 1.;
//     }
//     CHECK_NOTHROW( queue->execute() );
//     { // result of one narrow multiply
//       INFO( "Anarrow" );
//       double *data = b->get_data(new processor_coordinate_zero(1)); 
//       for (index_int i=0; i<nlocal; i++) {
// 	INFO( "b i=" << i );
// 	if (i<set-1 || i>set+1)
// 	  CHECK( data[i]==Approx(0.) );
// 	else
// 	  CHECK( data[i]!=Approx(0.) );
//       }
//     }
//     { // two narrow multiplies in a row
//       INFO( "Anarrow^2" );
//       double *data = c1->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++) {
// 	INFO( "c1 i=" << i );
// 	if (i<set-2 || i>set+2)
// 	  CHECK( data[i]==Approx(0.) );
// 	else
// 	  CHECK( data[i]!=Approx(0.) );
//       }
//     }
//     { // result of one wide multiply
//       INFO( "Awide" );
//       double *data = c2->get_data(new processor_coordinate_zero(1)); 
//       for (index_int i=0; i<nlocal; i++) {
// 	INFO( "b i=" << i );
// 	if (i==set-3 || i==set+3 || i==set)
// 	  CHECK( data[i]!=Approx(0.) );
// 	else
// 	  CHECK( data[i]==Approx(0.) );
//       }
//     }
//     { // adding it together
//       double *data = d->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++) {
// 	INFO( "d i=" << i );
// 	if (i<set-3 || i>set+3)
// 	  CHECK( data[i]==Approx(0.) );
// 	else
// 	  CHECK( data[i]!=Approx(0.) );
//       }
//     }
//   }
//   SECTION( "spilling" ) {
//     {
//       double *adata = a->get_data(new processor_coordinate_zero(1));
//       for (index_int i=0; i<nlocal; i++)
//   	adata[i] = 0.;
//       if (mytid%2==1)
//   	adata[0] = 1.;
//     }
//     CHECK_NOTHROW( queue->execute() );
//     {
//       double *data = d->get_data(new processor_coordinate_zero(1));
//       if (mytid%2==0) { // crud at the top
//   	for (index_int i=0; i<nlocal; i++) {
//   	  INFO( "i=" << i );
//   	  if (i<nlocal-3)
//   	    CHECK( data[i]==Approx(0.) );
//   	  else
//   	    CHECK( data[i]!=Approx(0.) );
//   	}
//       } else { // crud at the bottom
//   	for (index_int i=0; i<nlocal; i++) {
//   	  INFO( "i=" << i );
//   	  if (i>3)
//   	    CHECK( data[i]==Approx(0.) );
//   	  else
//   	    CHECK( data[i]!=Approx(0.) );
//   	}
//       }
//     }
//   }
// }
