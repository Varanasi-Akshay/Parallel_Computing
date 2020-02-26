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
 **** application level operations
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "setting values","[set][01]" ) {
  index_int nlocal=10;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto constant = std::shared_ptr<object>( new mpi_object(blocked) );
  double *data, v=1.;
  REQUIRE_NOTHROW( constant->allocate() );
  REQUIRE_NOTHROW( constant->set_value(v) );
  CHECK( constant->volume(mycoord)==nlocal );
  REQUIRE_NOTHROW( data = constant->get_data(mycoord) );
  for (int i=0; i<nlocal; i++)
    CHECK( data[i]==Approx(v) );
}

TEST_CASE( "Copy kernel","[kernel][copy][31]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ),
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ),
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ),
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ),
    qold = std::shared_ptr<object>( new mpi_object(blocked) );
  
  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto rr  = std::shared_ptr<object>( new mpi_object(scalar) );
  auto rrp = std::shared_ptr<object>( new mpi_object(scalar) );
  auto rnorm = std::shared_ptr<object>( new mpi_object(scalar) );
  auto pap = std::shared_ptr<object>( new mpi_object(scalar) );
  auto alpha = std::shared_ptr<object>( new mpi_object(scalar) );
  auto beta = std::shared_ptr<object>( new mpi_object(scalar) );
  
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  kernel *rrcopy;
  int n;
  double *rdata,*sdata;
  
  SECTION( "scalar" ) {
    n = 1;
    rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata[i] = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = new mpi_copy_kernel( rr,rrp ) );
    CHECK( rrcopy->has_type_compute() );
  }
  SECTION( "vector" ) {
    n = nlocal;
    rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
    for (int i=0; i<n; i++)
      rdata[i] = 2.*(mytid*n+i);
    REQUIRE_NOTHROW( rrcopy = new mpi_copy_kernel( r,z ) );
  }
  
  REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
  REQUIRE_NOTHROW( rrcopy->execute() );
  for (int i=0; i<n; i++)
    CHECK( sdata[i]==Approx(2.*(mytid*n+i)) );
}

TEST_CASE( "Scale kernel","[task][kernel][execute][33]" ) {

  INFO( "mytid=" << mytid );

  int nlocal=12; const char *mode = "default";
  auto no_op = ioperator("none");
  mpi_distribution *block = 
    new mpi_block_distribution(decomp,nlocal*ntids);
  auto xdata = new double[nlocal];
  auto
    xvector = std::shared_ptr<object>( new mpi_object(block,xdata) ),
    yvector = std::shared_ptr<object>( new mpi_object(block) );
  auto
    my_first = block->first_index_r(mycoord), my_last = block->last_index_r(mycoord);
  CHECK( my_first[0]==mytid*nlocal );
  CHECK( my_last[0]==(mytid+1)*nlocal-1 );
  for (int i=0; i<nlocal; i++)
    xdata[i] = pointfunc33(i,my_first[0]);
  mpi_kernel *scale;
  double *halo_data,*ydata;

  SECTION( "scale by constant" ) {

    SECTION( "kernel by pieces" ) {
      scale = new mpi_kernel(xvector,yvector);
      scale->set_name("33scale");
      dependency *d;
      REQUIRE_NOTHROW( d = scale->last_dependency() );
      REQUIRE_NOTHROW( d->set_type_local() );

      SECTION( "constant in the function" ) {
	scale->set_localexecutefn( &vecscalebytwo );
      }

      SECTION( "constant passed as context" ) {
	double x = 2;
	scale->set_localexecutefn
	  ( [x] ( kernel_function_args ) -> void {
	    return vecscalebyc( kernel_function_call,x ); } );
      }

      SECTION( "constant passed inverted" ) {
	double x = 1./2;
	scale->set_localexecutefn
	  ( [x] ( kernel_function_args ) -> void {
	    return vecscaledownbyc( kernel_function_call,x ); } );
      }
    }
    SECTION( "kernel as kernel from double-star" ) {
      mode = "scale kernel with scalar";
      double x = 2;
      scale = new mpi_scale_kernel(&x,xvector,yvector);
    }
    SECTION( "kernel as kernel from replicated object" ) {
      mode = "scale kernel with object";
      double xval = 2;
      distribution *scalar = new mpi_replicated_distribution(decomp);
      auto x = std::shared_ptr<object>( new mpi_object(scalar) );
      REQUIRE_NOTHROW( x->set_value(xval) );
      CHECK( scalar->has_type_replicated() );
      CHECK( x->has_type_replicated() );
      REQUIRE_NOTHROW( scale = new mpi_scale_kernel(x,xvector,yvector) );
      REQUIRE_NOTHROW( scale->analyze_dependencies() );
      std::vector<dependency*> deps;
      REQUIRE_NOTHROW( deps = scale->get_dependencies() );
      CHECK( deps.size()==2 );
    }

    INFO( "mode is " << mode );
    REQUIRE_NOTHROW( scale->analyze_dependencies() );
    CHECK( scale->get_tasks().size()==1 );

    std::shared_ptr<task> scale_task;
    CHECK_NOTHROW( scale_task = scale->get_tasks().at(0) );
    CHECK_NOTHROW( scale->execute() );
    // CHECK_NOTHROW( halo_data = scale_task->get_beta_object(0)->get_data(mycoord) );
    // CHECK_NOTHROW( ydata = yvector->get_data(mycoord) );
    // {
    //   int i;
    //   for (i=0; i<nlocal; i++) {
    // 	CHECK( halo_data[i] == Approx( pointfunc33(i,my_first) ) );
    //   }
    // }
  }

  SECTION( "axpy kernel" ) { 
    double x=2;
    scale = new mpi_axpy_kernel(xvector,yvector,&x);
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

TEST_CASE( "Stats kernel","[task][kernel][execute][34][hide]" ) {

  INFO( "mytid=" << mytid );
  index_int nlocal = 10;

  distribution
    *block_structure = new mpi_block_distribution(decomp,nlocal,-1);
  auto data = std::shared_ptr<object>( new mpi_object(block_structure) );
  kernel *setinput;

  REQUIRE_NOTHROW( setinput = new mpi_origin_kernel(data) );
  REQUIRE_NOTHROW( setinput->set_localexecutefn(&vecsetlinear) );
  REQUIRE_NOTHROW( setinput->analyze_dependencies() );
  REQUIRE_NOTHROW( setinput->execute() );

  distribution
    *stat_structure = new mpi_gathered_distribution(decomp);
  auto data_stats = std::shared_ptr<object>( new mpi_object(stat_structure) );

  SECTION( "spell out the stats kernel" ) {
    // sum everything locally to a single scalar
    distribution
      *scalar_structure = new mpi_block_distribution(decomp,1,-1);
    auto local_value = std::shared_ptr<object>( new mpi_object(scalar_structure) );
    kernel
      *local_stats = new mpi_kernel(data,local_value);

    REQUIRE_NOTHROW( local_stats->set_localexecutefn( &summing ) );
    REQUIRE_NOTHROW( local_stats->set_explicit_beta_distribution(data.get()) );
    REQUIRE_NOTHROW( local_stats->analyze_dependencies() );
    {
      std::shared_ptr<task> t;
      REQUIRE_NOTHROW( t = local_stats->get_tasks().at(0) );
      auto snds = t->get_send_messages(), rcvs = t->get_receive_messages();
      CHECK( snds.size()==1 );
      CHECK( rcvs.size()==1 );
    }
    REQUIRE_NOTHROW( local_stats->execute() );
    {
      double sum=0;
      for (index_int n=0; n<nlocal; n++)
	sum += mytid*nlocal+n;
      double *data;
      REQUIRE_NOTHROW( data = local_value->get_data(mycoord) );
      CHECK( data[0]==Approx(sum) );
    }

    kernel
      *global_stats = new mpi_kernel(local_value,data_stats);

    REQUIRE_NOTHROW( global_stats->set_localexecutefn( &veccopy ) );
    REQUIRE_NOTHROW( global_stats->set_explicit_beta_distribution(data_stats.get()) );
    REQUIRE_NOTHROW( global_stats->analyze_dependencies() );
    CHECK( data_stats->volume(mycoord)==ntids );
    {
      std::shared_ptr<task> t;
      REQUIRE_NOTHROW( t = global_stats->get_tasks().at(0) );
      auto snds = t->get_send_messages(), rcvs = t->get_receive_messages();
      CHECK( snds.size()==ntids );
      CHECK( rcvs.size()==ntids );
    }
    REQUIRE_NOTHROW( global_stats->execute() );
  }

  SECTION( "actual kernel" ) {
    kernel *compute_stats;
    REQUIRE_NOTHROW( compute_stats = new mpi_stats_kernel(data,data_stats,summing) );
    REQUIRE_NOTHROW( compute_stats->analyze_dependencies() );
    REQUIRE_NOTHROW( compute_stats->execute() );
  }

  {
    double *data;
    REQUIRE_NOTHROW( data = data_stats->get_data(mycoord) );
    for (int t=0; t<ntids; t++) {
      double sum=0;
      for (index_int n=0; n<nlocal; n++)
	sum += t*nlocal+n;
      CHECK( data[t]==Approx(sum) );
    }
  }
}

TEST_CASE( "Beta from sparse matrix","[beta][sparse][41]" ) {

  int localsize=20,gsize=localsize*ntids;
  mpi_distribution *block = new mpi_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new mpi_object(block) );
  auto out_obj = std::shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = .5;
  }
  mpi_kernel *kern = new mpi_kernel(in_obj,out_obj);
  kern->set_name("sparse-stuff");
  mpi_sparse_matrix *pattern;
  kern->set_localexecutefn
    ( [pattern] (kernel_function_args) -> void {
      return local_sparse_matrix_vector_multiply(kernel_function_call,(void*)pattern); } );

  index_int myfirst,mylast;
  REQUIRE_NOTHROW( myfirst = block->first_index_r(mycoord)[0] );
  REQUIRE_NOTHROW( mylast = block->last_index_r(mycoord)[0] );
  SECTION( "connect right" ) {
    REQUIRE_NOTHROW( pattern = new mpi_sparse_matrix(block /*,gsize*/ ) );
    for (index_int i=myfirst; i<=mylast; i++) {
      pattern->add_element(i,i);
      if (i+1<gsize) {
	CHECK_NOTHROW( pattern->add_element(i,i+1) );
      } else {
	REQUIRE_THROWS( pattern->add_element(i,i+1) );
      }
    }
    REQUIRE_NOTHROW( kern->last_dependency()->set_index_pattern(pattern) );
    SECTION( "just inspect halo" ) {
      REQUIRE_NOTHROW( kern->last_dependency()->ensure_beta_distribution(out_obj) );
      REQUIRE_NOTHROW( kern->last_dependency()->create_beta_vector(out_obj) );
      std::shared_ptr<object> halo;
      REQUIRE_NOTHROW( halo = kern->last_dependency()->get_beta_object() );
      std::shared_ptr<multi_indexstruct> mystruct;
      REQUIRE_NOTHROW( mystruct = halo->get_processor_structure(mycoord) );
      REQUIRE( mystruct!=nullptr );
      REQUIRE( !mystruct->is_empty() );
      if (mytid<ntids-1)
	CHECK( mystruct->get_component(0)
	       ->equals( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast+1)) ) );
      else
	CHECK( mystruct->get_component(0)
	       ->equals( std::shared_ptr<indexstruct>( new contiguous_indexstruct(myfirst,mylast)) ) );
    }
    SECTION( "all the way" ) {
      REQUIRE_NOTHROW( kern->analyze_dependencies() );

      std::vector<std::shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = kern->get_tasks() );
      CHECK( tasks.size()==1 );
      std::shared_ptr<task> threetask;
      REQUIRE_NOTHROW( threetask = tasks.at(0) );
      std::vector<message*> rmsgs;
      REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
      if (mytid==ntids-1) {
	CHECK( rmsgs.size()==1 );
      } else {
	CHECK( rmsgs.size()==2 );
      }
      for ( auto msg : rmsgs ) {
	if (msg->get_sender().equals(msg->get_receiver())) {
	  CHECK( msg->volume()==localsize );
	} else {
	  CHECK( msg->volume()==1 );
	}
      }
      //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
      // double *outdata = out_obj->get_data(mycoord); index_int n = out_obj->volume(mycoord);
      // if (mytid==ntids-1) {
      //   for (index_int i=0; i<n-1; i++)
      // 	CHECK( outdata[i] == Approx(1.) );
      //   CHECK( outdata[n-1] == Approx(.5) );
      // } else {
      //   for (index_int i=0; i<n; i++)
      // 	CHECK( outdata[i] == Approx(1.) );
      // }
    }
  }

  SECTION( "connect left" ) {
    REQUIRE_NOTHROW( pattern = new mpi_sparse_matrix(block) );
    auto pidx = block->get_dimension_structure(0);
    for (index_int i=pidx->first_index(mytid); i<=pidx->last_index(mytid); i++) {
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
    CHECK( tasks.size()==1 );
    std::shared_ptr<task> threetask;
    REQUIRE_NOTHROW( threetask = tasks.at(0) );
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = threetask->get_receive_messages() );
    if (mytid==0) {
      CHECK( rmsgs.size()==1 );
    } else {
      CHECK( rmsgs.size()==2 );
    }
    //REQUIRE_NOTHROW( kern->execute() ); // VLE no local function defined, so just copy
  }

}

TEST_CASE( "Actual sparse matrix","[beta][sparse][spmvp][42]" ) {
  REQUIRE(ntids>1); // need at least two processors
  INFO( "mytid: " << mytid );
  
  int localsize=10,gsize=localsize*ntids;
  mpi_distribution *block = new mpi_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new mpi_object(block) );
  auto out_obj = std::shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    double *indata = in_obj->get_data(mycoord);
    index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++)
      REQUIRE_NOTHROW( indata[i] = 1. );
  }
  mpi_kernel *spmvp = new mpi_kernel(in_obj,out_obj);
  spmvp->set_name("sparse-mvp");

  int ncols = 1;
  index_int mincol = block->first_index_r(mycoord)[0],
    maxcol = block->last_index_r(mycoord)[0];
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->domains_volume()));

  // create a matrix with zero row sums and 1.5 excess on the diagonal: ncols elements of -1 off diagonal
  // also build up an indexstruct my_columns
  mpi_sparse_matrix *mat;
  REQUIRE_NOTHROW( mat = new mpi_sparse_matrix(block) );
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0);
  auto my_columns = std::shared_ptr<indexstruct>( new indexed_indexstruct() );
  for (index_int row=my_first; row<=my_last; row++) {
    // diagonal element
    REQUIRE_NOTHROW( mat->add_element(row,row,(double)ncols+1.5) );
    REQUIRE_NOTHROW( my_columns->add_element( row ) );
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
  std::shared_ptr<indexstruct> mstruct;
  REQUIRE_NOTHROW( mstruct = mat->all_local_columns() );
  REQUIRE_NOTHROW( mstruct = mstruct->convert_to_indexed() );
  {
    INFO( fmt::format("matrix columns: {} s/b {}",
		      mstruct->as_string(),my_columns->as_string()) )
    CHECK( mstruct->equals(my_columns) );
  }

  spmvp->last_dependency()->set_index_pattern( mat );
  spmvp->set_localexecutefn
    ( [mat] ( kernel_function_args ) -> void {
      return local_sparse_matrix_vector_multiply(kernel_function_call,(void*)mat); } );
  SECTION( "analyze in bits and pieces" ) {
    REQUIRE_NOTHROW( spmvp->split_to_tasks() );
    auto t = spmvp->get_tasks().at(0);
    CHECK( !t->has_type_origin() );
    auto d = spmvp->get_dependencies().at(0);
    std::shared_ptr<object> out,in,halo;
    REQUIRE_NOTHROW( out = spmvp->get_out_object() );
    REQUIRE_NOTHROW( in = d->get_in_object() );
    REQUIRE_NOTHROW( d->ensure_beta_distribution(out) );
    REQUIRE_NOTHROW( d->create_beta_vector(out) );

    SECTION( "receive in pieces" ) {
      REQUIRE_NOTHROW( halo = d->get_beta_object() );
      auto beta_block = halo->get_processor_structure(mycoord);
      auto numa_block = halo->get_numa_structure();
      fmt::print("{}: beta={}, numa={}\n",
		 mycoord.as_string(),beta_block->as_string(),numa_block->as_string());
      self_treatment doself = self_treatment::INCLUDE;
      std::vector<message*> msgs;
      auto buildup = std::shared_ptr<multi_indexstruct>( new multi_indexstruct(1) );
      auto buildup2 = std::shared_ptr<multi_indexstruct>( new multi_indexstruct(1) );
      for (int p=0; p<ntids; p++) {
	auto pcoord = in->coordinate_from_linear(p);
	std::shared_ptr<multi_indexstruct> pstruct,mintersect;
	REQUIRE_NOTHROW( pstruct = in->get_processor_structure(pcoord) );
	REQUIRE_NOTHROW( mintersect = beta_block->intersect(pstruct) );
	REQUIRE_NOTHROW( buildup = buildup->struct_union(mintersect) ); 
      }
      INFO( fmt::format("buildup: {}",buildup->as_string()) );
      REQUIRE_NOTHROW( buildup = buildup->force_simplify(/* true */) );
      return;
      //fmt::print("built up {}, compare beta {}\n",
      //buildup->as_string(),beta_block->as_string());
      REQUIRE_NOTHROW( msgs = in->messages_for_segment
		       ( mycoord,doself,beta_block,numa_block ) );
    }
    return;
    SECTION( "receive in one" ) {
       REQUIRE_NOTHROW( t->derive_receive_messages() );
    }
    REQUIRE_NOTHROW( t->derive_send_messages() );
  }
  SECTION( "analyze and execute and test" ) {
    return;
    REQUIRE_NOTHROW( spmvp->analyze_dependencies() );
    printf("premature return from 42\n");
    return;

    std::vector<std::shared_ptr<task>> tasks;
    REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
    CHECK( tasks.size()==1 );
    std::shared_ptr<task> spmvptask;
    REQUIRE_NOTHROW( spmvptask = tasks.at(0) );
    std::vector<message*> rmsgs;
    REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
    for ( auto msg : rmsgs ) {
      if (!msg->get_sender().equals(msg->get_receiver()))
	if (mytid==ntids-1)
	  CHECK( msg->get_sender().coord(0)==0 );
	else 
	  CHECK( msg->get_sender().coord(0)==mytid+1 );
    }

    {
      distribution *beta_dist;
      REQUIRE_NOTHROW( beta_dist = spmvp->last_dependency()->get_beta_distribution() );
      std::shared_ptr<indexstruct> column_indices;
      REQUIRE_NOTHROW( column_indices = 
		       beta_dist->get_dimension_structure(0)->get_processor_structure(mytid) );
    }

    {
      std::vector<std::shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
      CHECK( tasks.size()==1 );
      std::shared_ptr<task> threetask;
      REQUIRE_NOTHROW( threetask = tasks.at(0) );
    }
    REQUIRE_NOTHROW( spmvp->execute() );

    {
      double *data = out_obj->get_data(mycoord);
      index_int lsize = out_obj->volume(mycoord);
      for (index_int i=0; i<lsize; i++) {
	INFO( "local i=" << i );
	CHECK( data[i] == Approx(1.5) );
      }
    }
  }
}

TEST_CASE( "Sparse matrix kernel","[beta][sparse][spmvp][hide][43]" ) {
  if (ntids==1) { printf("test 43 is multi-processor\n"); return; };

  INFO( fmt::format("{}",mycoord.as_string()) );
  int dim=1, localsize=100, gsize=localsize*ntids;
  INFO( "[43] sparse matrix example with " << localsize << " points local" );
  printf("ex 43 fails when localsize=200\n");
  mpi_distribution *block = new mpi_block_distribution(decomp,localsize,-1);
  auto in_obj = std::shared_ptr<object>( new mpi_object(block) );
  auto out_obj = std::shared_ptr<object>( new mpi_object(block) );
  REQUIRE_NOTHROW( in_obj->allocate() );
  {
    double *indata = in_obj->get_data(mycoord); index_int n = in_obj->volume(mycoord);
    for (index_int i=0; i<n; i++) indata[i] = 1.;
  }

  // create a matrix with zero row sums, shift diagonal up
  double dd = 1.5;
  mpi_sparse_matrix *mat = new mpi_sparse_matrix(block);
  index_int
    my_first = block->first_index_r(mycoord).coord(0),
    my_last = block->last_index_r(mycoord).coord(0),
    mincol,maxcol;
  int skip=0;
  // initialize random
  srand((int)(mytid*(double)RAND_MAX/block->domains_volume()));
  auto neighbr = std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,localsize-1) );
  if (mytid<ntids-1)
    neighbr = neighbr->operate( shift_operator(my_last+1) );
  INFO( fmt::format("neighbour connection will be limited to {}",neighbr->as_string()) );
  for (index_int row=my_first; row<=my_last; row++) {
    int ncols = 4;
    auto nzcol = std::shared_ptr<indexstruct>( new indexed_indexstruct() );
    for (index_int ic=0; ic<ncols; ic++) {
      index_int col,oops_col,
	col_shift = (index_int) ( 1.*(localsize/2-1)*rand() / (double)RAND_MAX );
      CHECK( col_shift>=0 );
      CHECK( col_shift<localsize );
      if (mytid<ntids-1) { col = my_last+1+col_shift; oops_col = my_last+localsize; }
      else               { col = col_shift;           oops_col = localsize-1; }
      if (ic==0) {
	mincol=col; maxcol = col; }
      while (nzcol->contains_element(col))
	col++; // make sure we hit every column just once
      REQUIRE( col<oops_col );
      if ( (mytid<ntids-1 && col>=my_last+localsize)
	   || (mytid==ntids-1 && col>=localsize) ) { // can't fit this in the next proc
	skip++;
      } else {
	nzcol->add_element(col);
	if (col<mincol) mincol = col;
	if (col>maxcol) maxcol = col;
	REQUIRE_NOTHROW( mat->add_element(row,col,-1.) ); // off elt
      }
    }
    INFO( fmt::format("Elements added: {}",nzcol->as_string()) );
    CHECK( neighbr->contains(nzcol) );
    REQUIRE_NOTHROW( mat->add_element(row,row,ncols+dd-skip) ); // diag elt
  }

  mpi_kernel *spmvp;
  REQUIRE_NOTHROW( spmvp = new mpi_spmvp_kernel(in_obj,out_obj,mat) );
  REQUIRE_NOTHROW( spmvp->analyze_dependencies() );

  std::vector<std::shared_ptr<task>> tasks;
  REQUIRE_NOTHROW( tasks = spmvp->get_tasks() );
  CHECK( tasks.size()==1 );
  std::shared_ptr<task> spmvptask;
  REQUIRE_NOTHROW( spmvptask = tasks.at(0) );
  std::vector<message*> rmsgs;
  REQUIRE_NOTHROW( rmsgs = spmvptask->get_receive_messages() );
  for ( auto msg : rmsgs ) {
    auto sender = msg->get_sender(), receiver = msg->get_receiver();
    INFO( "receive message " << sender.as_string() << "->" << receiver.as_string() );
    if (!sender.equals(receiver))
      if (mytid==ntids-1) {
	// auto proc0 = processor_coordinate_zero(dim);
	// CHECK( *sender==proc0 );
	CHECK( sender.coord(0)==0 );
      } else {
	// auto procn = mycoord+1;
	// CHECK( *sender==procn );
	CHECK( sender.coord(0)==mytid+1 );
      }
  }
  std::vector<message*> smsgs;
  REQUIRE_NOTHROW( smsgs = spmvptask->get_send_messages() );
  for ( auto msg : smsgs ) {
    auto sender = msg->get_sender(), receiver = msg->get_receiver();
    INFO( "receive message " << sender.as_string() << "->" << receiver.as_string() );
    if (!sender.equals(receiver))
      if (mytid==0)
	CHECK( receiver.coord(0)==ntids-1 );
      else 
	CHECK( receiver.coord(0)==mytid-1 );
  }

  {
    distribution *beta_dist;
    REQUIRE_NOTHROW( beta_dist = spmvp->last_dependency()->get_beta_distribution() );
    std::shared_ptr<indexstruct> column_indices;
    REQUIRE_NOTHROW( column_indices =
		 beta_dist->get_dimension_structure(0)->get_processor_structure(mytid) );
    //    printf("column_indices on %d: %s\n",mytid,column_indices->as_string());
    //    CHECK( column_indices->is_sorted() ); 
  }

  REQUIRE_NOTHROW( spmvp->execute() );

  {
    double *data = out_obj->get_data(mycoord);
    index_int lsize = out_obj->volume(mycoord);
    for (index_int i=0; i<lsize; i++) {
      CHECK( data[i] == Approx(dd) );
    }
  }
}

TEST_CASE( "matrix kernel analysis","[kernel][spmvp][sparse][44]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  distribution *blocked =
    new mpi_block_distribution(decomp,g);
  auto x = std::shared_ptr<object>( new mpi_object(blocked) ),
    y = std::shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );

  // set the matrix to one lower diagonal
  mpi_sparse_matrix
    *Aup = new mpi_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_size().at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      // A narrow is tridiagonal
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  kernel *k;
  REQUIRE_NOTHROW( k = new mpi_spmvp_kernel( x,y,Aup ) );
  REQUIRE_NOTHROW( k->analyze_dependencies() );

  // analyze the message structure
  auto tasks = k->get_tasks();
  CHECK( tasks.size()==1 );
  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
    if ((*t)->get_step()==x->get_object_number()) {
      CHECK( (*t)->get_dependencies().size()==0 );
    } else {
      auto send = (*t)->get_send_messages();
      if (mytid==ntids-1)
	CHECK( send.size()==1 );
      else
	CHECK( send.size()==2 );
      auto recv = (*t)->get_receive_messages();
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
    index_int my_first = blocked->first_index_r(mycoord).coord(0);
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
    index_int my_first = blocked->first_index_r(mycoord).coord(0);
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

TEST_CASE( "matrix iteration shift left","[kernel][sparse][45]" ) {
  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal, nsteps = 2; //2*nlocal+3;
  distribution *blocked =
    new mpi_block_distribution(decomp,g);
  auto x = std::shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );

  // set the input vector to delta on the first element
  {
    double *d = x->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++) d[i] = 0.;
    if (mytid==0) d[0] = 1.;
  }
  
  std::vector<std::shared_ptr<object>> y(nsteps);
  for (int i=0; i<nsteps; i++) {
    y.at(i) = std::shared_ptr<object>( new mpi_object(blocked) );
  }

  // set the matrix to one lower diagonal
  mpi_sparse_matrix
    *Aup = new mpi_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_size().at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    for (index_int row=my_first; row<=my_last; row++) {
      int col;
      col = row;   Aup->add_element(row,col,1.);
      col = row-1; if (col>=0) Aup->add_element(row,col,1.);
    }
  }

  // make a queue
  mpi_algorithm *queue = new mpi_algorithm(decomp);
  REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(x) ) );
  std::shared_ptr<object> inobj = x, outobj;
  for (int i=0; i<nsteps; i++) {
    outobj = y[i];
    kernel *k;
    REQUIRE_NOTHROW( k = new mpi_spmvp_kernel( inobj,outobj,Aup ) );
    fmt::MemoryWriter w; w.write("mvp-{}",i); k->set_name( w.str() );
    REQUIRE_NOTHROW( queue->add_kernel(k) );
    inobj = outobj;
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );

  // analyze the message structure
  auto tasks = queue->get_tasks();
  CHECK( tasks.size()==(nsteps+1) );
  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
    if ((*t)->get_step()==x->get_object_number()) {
      CHECK( (*t)->get_dependencies().size()==0 );
    } else {
      std::vector<dependency*> deps = (*t)->get_dependencies();
      CHECK( deps.size()==1 );
      dependency *dep = deps.at(0);

      INFO( "step " << (*t)->get_step() );

      std::vector<message*> recv,send;
      REQUIRE_NOTHROW( recv = (*t)->get_receive_messages() );
      REQUIRE_NOTHROW( send = (*t)->get_send_messages() );

      fmt::MemoryWriter w;
      w.write("[{}] receiving from: ",mytid);
      for (int i=0; i<recv.size(); i++)
	REQUIRE_NOTHROW( w.write("{},",recv.at(i)->get_sender().coord(0)) );
      w.write(". sending to: ");
      for (int i=0; i<send.size(); i++)
	REQUIRE_NOTHROW( w.write("{},",send.at(i)->get_receiver().coord(0)) );

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
  index_int my_first = blocked->first_index_r(mycoord).coord(0);
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

TEST_CASE( "special matrices","[kernel][spmvp][sparse][46]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, g = ntids*nlocal;
  distribution *blocked =
    new mpi_block_distribution(decomp,g);
  index_int my_first = blocked->first_index_r(mycoord).coord(0), my_last = blocked->last_index_r(mycoord).coord(0);
  mpi_sparse_matrix *A;

  SECTION( "lower diagonal" ) {
    REQUIRE_NOTHROW( A = new mpi_lowerbidiagonal_matrix( blocked, 1,0 ) );
    if (mytid==0)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto x = std::shared_ptr<object>( new mpi_object(blocked) ),
	y = std::shared_ptr<object>( new mpi_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new mpi_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies().size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  if (mytid==ntids-1)
	    CHECK( send.size()==1 );
	  else
	    CHECK( send.size()==2 );
	  auto recv = (*t)->get_receive_messages();
	  if (mytid==0)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector< std::shared_ptr<object> >(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data[i] = 0.;
      if (mytid==0) data[0] = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(objs[0]) ) );
      for (int istep=1; istep<g; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	//INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_data(mycoord) );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==istep) CHECK( data[i-my_first]==Approx(1.) );
	  else          CHECK( data[i-my_first]==Approx(0.) );
	}
      }
    }
  }

  SECTION( "upper diagonal" ) {
    REQUIRE_NOTHROW( A = new mpi_upperbidiagonal_matrix( blocked, 0,1 ) );
    if (mytid==ntids-1)
      CHECK( A->nnzeros()==2*nlocal-1 );
    else
      CHECK( A->nnzeros()==2*nlocal );

    SECTION( "kernel analysis" ) {
      auto x = std::shared_ptr<object>( new mpi_object(blocked) ),
	y = std::shared_ptr<object>( new mpi_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new mpi_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies().size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  if (mytid==0)
	    CHECK( send.size()==1 );
	  else
	    CHECK( send.size()==2 );
	  auto recv = (*t)->get_receive_messages();
	  if (mytid==ntids-1)
	    CHECK( recv.size()==1 );
	  else
	    CHECK( recv.size()==2 );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data[i] = 0.;
      if (mytid==ntids-1) data[nlocal-1] = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(objs[0]) ) );
      for (int istep=1; istep<g; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );
      for (int istep=0; istep<g; istep++) {
	//INFO( "object " << istep << ": " << objs[istep]->values_as_string(mytid) );
	REQUIRE_NOTHROW( data = objs[istep]->get_data(mycoord) );
	for (index_int i=my_first; i<=my_last; i++) {
	  INFO( "index " << i );
	  if (i==g-1-istep) CHECK( data[i-my_first]==Approx(1.) );
	  else            CHECK( data[i-my_first]==Approx(0.) );
	}
      }
    }
  }

  SECTION( "toeplitz" ) {
    REQUIRE_NOTHROW( A = new mpi_toeplitz3_matrix( blocked, 0,2,0 ) );
    CHECK( A->nnzeros()==3*nlocal-(mytid==0)-(mytid==ntids-1) );

    SECTION( "kernel analysis" ) {
      auto x = std::shared_ptr<object>( new mpi_object(blocked) ),
	y = std::shared_ptr<object>( new mpi_object(blocked) );
      kernel *k;
      REQUIRE_NOTHROW( k = new mpi_spmvp_kernel( x,y,A ) );
      REQUIRE_NOTHROW( k->analyze_dependencies() );

      // analyze the message structure
      auto tasks = k->get_tasks();
      CHECK( tasks.size()==1 );
      for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
	if ((*t)->get_step()==x->get_object_number()) {
	  CHECK( (*t)->get_dependencies().size()==0 );
	} else {
	  auto send = (*t)->get_send_messages();
	  CHECK( send.size()==3-(mytid==0)-(mytid==ntids-1) );
	  auto recv = (*t)->get_receive_messages();
	  CHECK( send.size()==3-(mytid==0)-(mytid==ntids-1) );
	}
      }
    }
    SECTION( "run!" ) {
      auto objs = std::vector<std::shared_ptr<object>>(g);
      for (int iobj=0; iobj<g; iobj++)
	objs[iobj] = std::shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_NOTHROW( objs[0]->allocate() );
      double *data; REQUIRE_NOTHROW( data = objs[0]->get_data(mycoord) );
      for (index_int i=0; i<nlocal; i++) data[i] = 1.;
      //      if (mytid==ntids-1) data[nlocal-1] = 1.;
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(objs[0]) ) );
      int nsteps = 5;
      for (int istep=1; istep<nsteps; istep++) {
	kernel *k;
	REQUIRE_NOTHROW( k = new mpi_spmvp_kernel(objs[istep-1],objs[istep],A) );
	REQUIRE_NOTHROW( queue->add_kernel(k) );
      }
      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );

      REQUIRE_NOTHROW( data = objs[nsteps-1]->get_data(mycoord) );
      for (index_int i=my_first; i<=my_last; i++) {
	INFO( "index " << i );
	CHECK( data[i-my_first]==Approx(pow(2,nsteps-1)) );
      }
    }
  }
}

TEST_CASE( "central difference matrices","[kernel][47]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 1, g = ntids*nlocal;
  distribution *blocked =
    new mpi_block_distribution(decomp,g);
  auto x = std::shared_ptr<object>( new mpi_object(blocked) ),
    y = std::shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( x->set_value(1.) );
  kernel *diff; REQUIRE_NOTHROW( diff = new mpi_centraldifference_kernel(x,y) );
  REQUIRE_NOTHROW( diff->analyze_dependencies() );
  auto tsks = diff->get_tasks();
  CHECK( tsks.size()==1 );
  auto tsk = tsks.at(0);
  auto msgs = tsk->get_receive_messages();
  if (mytid==0 || mytid==ntids-1)
    CHECK( msgs.size()==2 );
  else
    CHECK( msgs.size()==3 );
  REQUIRE_NOTHROW( diff->execute() );
  double *data;
  REQUIRE_NOTHROW( data = y->get_data(mycoord) );
  printf("ops 47 premature end because of data problem\n");
  return;
  // global left end point
  int e = 0;
  if (mytid==0) {
    CHECK( data[0]==Approx(1.) ); e++;
  } else if (mytid==ntids-1 && nlocal==1) {
    CHECK( data[0]==Approx(1.) );
  } else 
    CHECK( data[0]==Approx(0.) );
  // global right endpoint
  if (mytid==ntids-1) {
    CHECK( data[nlocal-1]==Approx(1.) ); e++;
  } else if (mytid==0 && nlocal==1) {
    CHECK( data[nlocal-1]==Approx(1.) );
  } else 
    CHECK( data[nlocal-1]==Approx(0.) );
  // interior
  for (int i=1; i<nlocal-1; i++) {
    CHECK( data[i]==Approx(0.) );
  }
  // consistency check on end points
  CHECK( e==(mytid==0)+(mytid==ntids-1) );
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto norm = std::shared_ptr<object>( new mpi_object(scalar) );
  kernel *take_norm = new mpi_norm_kernel(y,norm);
  take_norm->analyze_dependencies();
  take_norm->execute();
  REQUIRE_NOTHROW( data = norm->get_data(mycoord) );
  CHECK( data[0]==Approx(sqrt(2.)) );
}

TEST_CASE( "central difference as in CG","[kernel][48]" ) {

  index_int nlocal = 5;

  // architecture *aa; decomposition *decomp;
  // REQUIRE_NOTHROW( aa = env->make_architecture() );
  // int can_embed = 0;
  // REQUIRE_NOTHROW( aa->set_can_embed_in_beta(can_embed) );
  // decomposition *ddecomp = new decomposition(aa);

  decomposition *ddecomp = new decomposition(decomp);

  const char *path;
  SECTION( "ptp" ) { path = "ptp";
    ddecomp->set_collective_strategy( collective_strategy::ALL_PTP );
  }
  // SECTION( "group" ) { path = "group";
  //   ddecomp->set_collective_strategy( collective_strategy::GROUP );
  // }
  // SECTION( "recursive" ) { path = "recursive";
  //   ddecomp->set_collective_strategy( collective_strategy::RECURSIVE );
  // }
  // SECTION( "mpi" ) { path = "mpi";
  //   ddecomp->set_collective_strategy( collective_strategy::MPI );
  // }
  INFO( "collective strategy: " << path );

  // a bunch of vectors, block distributed
  distribution *blocked = new mpi_block_distribution(ddecomp,nlocal,-1);
  std::shared_ptr<object> xt,b0,r0,ax0;
  xt    = std::shared_ptr<object>( new mpi_object(blocked) );
  xt->set_name(fmt::format("xtrue"));
  xt->allocate(); //xt->set_value(1.);
  b0    = std::shared_ptr<object>( new mpi_object(blocked) );
  b0->set_name(fmt::format("b0"));


  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(ddecomp);
  auto rr0 = std::shared_ptr<object>( new mpi_object(scalar) );

  // let's define the steps of the loop body
  algorithm *cg = new mpi_algorithm(ddecomp);
  cg->set_name("Conjugate Gradients Method");

  { kernel *xorigin = new mpi_origin_kernel( xt );
    xorigin->set_localexecutefn(&vecsetlinear);
    cg->add_kernel(xorigin); xorigin->set_name("origin xtrue");
  }

  kernel *matvec = new mpi_centraldifference_kernel( xt,b0 );
  cg->add_kernel(matvec); matvec->set_name("b0=A xtrue");

  {
    kernel *r0inp = new mpi_norm_kernel( b0,rr0 );
    cg->add_kernel( r0inp );
  }
  REQUIRE_NOTHROW( cg->analyze_dependencies() );

  // get rolling.....
  REQUIRE_NOTHROW( cg->execute() );

  // something weird going on with the matrix-vector product
  std::vector<std::shared_ptr<task>> tsks; REQUIRE_NOTHROW( tsks = matvec->get_tasks() );
  CHECK( tsks.size()==1 );
  auto tsk = tsks.at(0);
  std::vector<message*> msgs;
  // check send messages
  REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
  if (mytid==0 || mytid==ntids-1) {
    CHECK( msgs.size()==2 );
  } else {
    CHECK( msgs.size()==3 );
  }
  auto myfirst = blocked->first_index_r(mycoord),
    mylast = blocked->last_index_r(mycoord);
  for ( auto m : msgs ) {
    INFO( fmt::format("send message: {}",m->as_string()) );
    CHECK( m->get_sender()==mycoord );
    processor_coordinate other;
    REQUIRE_NOTHROW( other = m->get_receiver() );
    auto left = mycoord-1, right = mycoord+1;
    std::shared_ptr<multi_indexstruct> global_struct,local_struct;
    CHECK_NOTHROW( global_struct = m->get_global_struct() );
    CHECK_NOTHROW( local_struct = m->get_local_struct() );
    if (other==mycoord) {
      CHECK( global_struct->volume()==nlocal );
      CHECK( local_struct->volume()==nlocal );
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==mylast );
    } else if (other==left) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we send the first element
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==myfirst );
    } else if (other==right) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we send the last element
      CHECK( global_struct->first_index_r()==mylast );
      CHECK( global_struct->last_index_r()==mylast );
    } else
      throw(fmt::format("{}: strange receiver {}",mycoord.as_string(),other.as_string()));
  }
  // check receive messages
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  if (mytid==0 || mytid==ntids-1) {
    CHECK( msgs.size()==2 );
  } else {
    CHECK( msgs.size()==3 );
  }
  for ( auto m : msgs ) {
    INFO( fmt::format("receive message: {}",m->as_string()) );
    CHECK( m->get_receiver()==mycoord );
    processor_coordinate other;
    REQUIRE_NOTHROW( other = m->get_sender() );
    auto left = mycoord-1, right = mycoord+1;
    std::shared_ptr<multi_indexstruct> global_struct,local_struct;
    CHECK_NOTHROW( global_struct = m->get_global_struct() );
    CHECK_NOTHROW( local_struct = m->get_local_struct() );
    if (other==mycoord) {
      CHECK( global_struct->volume()==nlocal );
      CHECK( local_struct->volume()==nlocal );
      CHECK( global_struct->first_index_r()==myfirst );
      CHECK( global_struct->last_index_r()==mylast );
    } else if (other==left) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we receive the first element
      auto leftbnd = myfirst-1;
      CHECK( global_struct->first_index_r()==leftbnd );
      CHECK( global_struct->last_index_r()==leftbnd );
    } else if (other==right) {
      CHECK( global_struct->volume()==1 );
      CHECK( local_struct->volume()==1 );
      // we receive the last element
      auto rightbnd = mylast+1;
      CHECK( global_struct->first_index_r()==rightbnd );
      CHECK( global_struct->last_index_r()==rightbnd );
    } else
      throw(fmt::format("{}: strange receiver {}",mycoord.as_string(),other.as_string()));
  }

  // initial vector is linear
  { double *xdata;
    REQUIRE_NOTHROW( xdata = xt->get_data(mycoord) );
    for (int i=0; i<nlocal; i++)
      CHECK( xdata[i]==Approx(mytid*nlocal+i) );
  }

  // halo should be linear
  { double *xdata; std::shared_ptr<object> mathalo; index_int hsize,hfirst;
    REQUIRE_NOTHROW( mathalo = matvec->get_beta_object(0) );
    REQUIRE_NOTHROW( xdata = mathalo->get_data(mycoord) );
    REQUIRE_NOTHROW( hsize = mathalo->volume(mycoord) );
    REQUIRE_NOTHROW( hfirst = mathalo->first_index_r(mycoord)[0] );
    for ( index_int i=0; i<hsize; i++) {
      INFO( fmt::format("{} halo @{}+{} = {}",mycoord.as_string(),hfirst,i,xdata[i]) );
      CHECK( xdata[i]==Approx(hfirst+i) );
    }
    // mvp only 1 at the ends
    double *bdata;
    REQUIRE_NOTHROW( bdata = b0->get_data(mycoord) );
    for (int i=0; i<nlocal; i++) {
      INFO( "p=" << mytid << ", i=" << i );
      if (mytid==0 && i==0)
	CHECK( bdata[i]==Approx(-1.) );
      else if (mytid==ntids-1 && i==nlocal-1)
	CHECK( bdata[i]==Approx(ntids*nlocal) );
      else
	CHECK( bdata[i]==Approx(0.) );
    }
  }

  // norm is sqrt(2)
  { double *ndata;
    REQUIRE_NOTHROW( ndata = rr0->get_data(mycoord) );
    CHECK( ndata[0]==Approx(sqrt( ntids*nlocal*ntids*nlocal + 1 )) );
  }

}

TEST_CASE( "compound kernel queue","[kernel][queue][50]" ) {

  // this test is based on the old grouping strategy
  architecture *aa;
  REQUIRE_NOTHROW( aa = env->make_architecture() );
  aa->set_collective_strategy_group();
  decomposition *decomp = new mpi_decomposition(aa);

  int nlocal = 50;
  distribution
    *block = new mpi_block_distribution(decomp,nlocal*ntids),
    *scalar = new mpi_replicated_distribution(decomp);
  auto x = std::shared_ptr<object>( new mpi_object(block) ),
    y = std::shared_ptr<object>( new mpi_object(block) ),
    xy = std::shared_ptr<object>( new mpi_object(scalar) );
  kernel
    *makex = new mpi_origin_kernel(x),
    *makey = new mpi_origin_kernel(y),
    *prod = new mpi_innerproduct_kernel(x,y,xy);
  algorithm *queue = new mpi_algorithm(decomp);
  int inprod_step;

  SECTION( "analyze in steps" ) {

    const char *path;
    SECTION( "kernels in logical order" ) {
      path = "in order";
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      inprod_step = 2;
    }
  
    SECTION( "kernels in wrong order" ) {
      path = "reversed";
      REQUIRE_NOTHROW( queue->add_kernel(prod) );
      REQUIRE_NOTHROW( queue->add_kernel(makex) );
      REQUIRE_NOTHROW( queue->add_kernel(makey) );
      inprod_step = 0;
    }
    INFO( "kernels were added: " << path );

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
	CHECK( tasks.size()==3 ); // this depends on how many tasks the reduction contributes
      else 
	CHECK( tasks.size()==1 );
      REQUIRE_NOTHROW( queue->add_kernel_tasks_to_queue(k) );
    }
    
    REQUIRE_NOTHROW( tasks = queue->get_tasks() );
    for (auto t : tasks ) {
      if (!t->has_type_origin()) {
	std::shared_ptr<object> in; int inn; int ostep;
	CHECK_NOTHROW( in = t->last_dependency()->get_in_object() );
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
  for (auto t : tsks ) {
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
  distribution *local_scalar = new mpi_block_distribution(decomp,1,-1);
  auto local_value = std::shared_ptr<object>( new mpi_object(local_scalar) );
  REQUIRE_NOTHROW( local_value->allocate() );
  REQUIRE_NOTHROW( data = local_value->get_data(mycoord) );
  data[0] = (double)mytid;

  mpi_distribution *replicated;
  REQUIRE_NOTHROW( replicated = new mpi_replicated_distribution(decomp) );
  std::shared_ptr<object> global_sum;
  REQUIRE_NOTHROW( global_sum = std::shared_ptr<object>( new mpi_object(replicated) ) );

  SECTION( "spell out the way to the top" ) {
    mode = "spell it out";
    int
      groupsize = MIN(4,P), mygroupnum = mytid/groupsize, nfullgroups = ntids/groupsize,
      grouped_tids = nfullgroups*groupsize, // how many procs are in perfect groups?
      remainsize = P-grouped_tids, ngroups = nfullgroups+(remainsize>0);
    CHECK( remainsize<groupsize );
    mpi_distribution *locally_grouped;
    { // every proc gets to know all the indices of its group
      parallel_structure *groups;
      REQUIRE_NOTHROW( groups = new parallel_structure(decomp) );
      for (int p=0; p<P; p++) {
	processor_coordinate pcoord;
	REQUIRE_NOTHROW( pcoord = decomp->coordinate_from_linear(p) );
	index_int groupnumber = p/groupsize,
	  f = groupsize*groupnumber,l=MIN(f+groupsize-1,g);
	REQUIRE( l>=f );
	std::shared_ptr<indexstruct> pstruct; std::shared_ptr<multi_indexstruct> mpstruct;
	REQUIRE_NOTHROW
	  ( pstruct = std::shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) ) );
	REQUIRE_NOTHROW( mpstruct = std::shared_ptr<multi_indexstruct>
			 (new multi_indexstruct( pstruct )) );
	REQUIRE_NOTHROW( groups->set_processor_structure(pcoord,mpstruct ) );
      }
      REQUIRE_NOTHROW( groups->set_type( groups->infer_distribution_type() ) );
      // this one works:
      //fmt::print("Locally grouped: each has its group <<{}>>\n",groups->as_string());
      REQUIRE_NOTHROW( locally_grouped = new mpi_distribution(/*decomp,*/groups) );
    }

    mpi_distribution *partially_reduced;
    {
      parallel_structure
	*partials = new parallel_structure(decomp);
      for (int p=0; p<P; p++) {
	processor_coordinate pcoord;
	REQUIRE_NOTHROW( pcoord = decomp->coordinate_from_linear(p) );
	index_int groupnumber = p/groupsize;
	auto pstruct = std::shared_ptr<multi_indexstruct>
	  (new multi_indexstruct
	   ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(groupnumber) ) ));
	REQUIRE_NOTHROW( partials->set_processor_structure(pcoord,pstruct) );
      }
      REQUIRE_NOTHROW( partials->set_type( partials->infer_distribution_type() ) );
      REQUIRE_NOTHROW( partially_reduced = new mpi_distribution(/*decomp,*/partials) );
    }
    std::shared_ptr<object> partial_sums;
    REQUIRE_NOTHROW( partial_sums = std::shared_ptr<object>( new mpi_object(partially_reduced) ) );
    REQUIRE_NOTHROW( partial_sums->allocate() );
  
    // SECTION( "group and sum separate" ) {
    //   std::shared_ptr<object> local_groups;
    //   mpi_kernel *partial_grouping,*local_summing_to_global;

    //   // one kernel for gathering the local values
    //   REQUIRE_NOTHROW( local_groups = new mpi_object(locally_grouped) );
    //   REQUIRE_NOTHROW( local_groups->allocate() );
    //   REQUIRE_NOTHROW( partial_grouping = new mpi_kernel(local_value,local_groups) );
    //   REQUIRE_NOTHROW( partial_grouping->set_localexecutefn( &veccopy ) );
    //   REQUIRE_NOTHROW( partial_grouping->set_explicit_beta_distribution(locally_grouped) );
    //   REQUIRE_NOTHROW( partial_grouping->analyze_dependencies() );
    //   REQUIRE_NOTHROW( partial_grouping->execute() );

    //   std::vector<message*> *msgs; std::vector<std::shared_ptr<task>> tsks; double *data;
    //   REQUIRE_NOTHROW( tsks = partial_grouping->get_tasks() );
    //   int nt = tsks->size(); CHECK( nt==1 );
    //   std::shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks->at(0) );
    //   REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
    //   if (mytid<grouped_tids) {
    // 	CHECK( msgs->size()==groupsize );
    //   } else {
    // 	CHECK( msgs->size()==remainsize );
    //   }
    //   REQUIRE_NOTHROW( msgs = t->get_send_messages() );
    //   CHECK( local_groups->first_index_r(mycoord).coord(0)==mygroupnum*groupsize );
    //   if (mytid<grouped_tids) {
    // 	CHECK( msgs->size()==groupsize );
    // 	CHECK( local_groups->volume(mycoord)==groupsize );
    //   } else {
    // 	CHECK( msgs->size()==remainsize );
    // 	CHECK( local_groups->volume(mycoord)==remainsize );
    //   }
    //   REQUIRE_NOTHROW( data = local_groups->get_data(mycoord) );
    //   for (index_int i=0; i<local_groups->volume(mycoord); i++) {
    // 	index_int ig = (local_groups->first_index_r(mycoord).coord(0)+i);
    // 	INFO( "ilocal=" << i << ", iglobal=" << ig );
    // 	CHECK( data[i]==ig );
    //   }

    //   REQUIRE_NOTHROW( local_summing_to_global = new mpi_kernel(local_groups,partial_sums) );
    //   //REQUIRE_NOTHROW( local_summing_to_global->set_type_local() );
    //   REQUIRE_NOTHROW( local_summing_to_global->set_explicit_beta_distribution
    // 		       (locally_grouped) );
    //   REQUIRE_NOTHROW( local_summing_to_global->set_localexecutefn( &summing ) );
    //   REQUIRE_NOTHROW( local_summing_to_global->analyze_dependencies() );
    //   REQUIRE_NOTHROW( local_summing_to_global->execute() );

    //   // duplicate code with previous section
    //   REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
    //   CHECK( partial_sums->volume(mycoord)==1 );
    //   index_int f = locally_grouped->first_index_r(mycoord).coord(0),
    // 	l = locally_grouped->last_index_r(mycoord).coord(0), s = (l+f)*(l-f+1)/2;
    //   CHECK( data[0]==s );
    // }

    SECTION( "group and sum in one" ) {
      mpi_kernel *partial_summing;
      REQUIRE_NOTHROW( partial_summing = new mpi_kernel(local_value,partial_sums) );
      // depend on the numbers in your group
      REQUIRE_NOTHROW( partial_summing->set_explicit_beta_distribution(locally_grouped) );
      REQUIRE_NOTHROW( partial_summing->set_localexecutefn( &summing ) );
      REQUIRE_NOTHROW( partial_summing->analyze_dependencies() );

      std::vector<message*> msgs; std::vector<std::shared_ptr<task>> tsks; double *data;
      REQUIRE_NOTHROW( tsks = partial_summing->get_tasks() );
      int nt = tsks.size(); CHECK( nt==1 );
      std::shared_ptr<task> t; REQUIRE_NOTHROW( t = tsks.at(0) );
      //INFO( "partial summing task: " << t->as_string() );
      REQUIRE_NOTHROW( msgs = t->get_receive_messages() );
      if (mytid<grouped_tids) { // if I'm in a full group
	CHECK( msgs.size()==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
      }
      REQUIRE_NOTHROW( msgs = t->get_send_messages() );
      CHECK( partial_sums->volume(mycoord)==1 );
      if (mytid<grouped_tids) { // if I'm in a full group
	CHECK( msgs.size()==groupsize );
      } else {
	CHECK( msgs.size()==remainsize );
      }

      REQUIRE_NOTHROW( partial_summing->execute() );

      // duplicate code with previous section
      REQUIRE_NOTHROW( data = partial_sums->get_data(mycoord) );
      CHECK( partial_sums->volume(mycoord)==1 );
      index_int f = locally_grouped->first_index_r(mycoord).coord(0),
	l = locally_grouped->last_index_r(mycoord).coord(0), s = (l+f)*(l-f+1)/2;
      CHECK( data[0]==s );
    }

    mpi_kernel *top_summing;
    REQUIRE_NOTHROW( top_summing = new mpi_kernel(partial_sums,global_sum) );
    parallel_structure *top_beta = new parallel_structure(decomp);
    for (int p=0; p<P; p++)
      REQUIRE_NOTHROW
	( top_beta->set_processor_structure
	  (p, std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,ngroups-1)) ) );
    REQUIRE_NOTHROW( top_beta->set_type( top_beta->infer_distribution_type() ) );
    REQUIRE_NOTHROW( top_summing->set_explicit_beta_distribution
		     ( new mpi_distribution(/*decomp,*/top_beta) ) );
    REQUIRE_NOTHROW( top_summing->set_localexecutefn( &summing ) );
    REQUIRE_NOTHROW( top_summing->analyze_dependencies() );
    REQUIRE_NOTHROW( top_summing->execute() );
  }

  SECTION( "using the reduction kernel" ) {
    mpi_kernel *sumkernel;
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel, send/recv";
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( sumkernel = new mpi_reduction_kernel(local_value,global_sum) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel, grouping";
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( sumkernel = new mpi_reduction_kernel(local_value,global_sum) );
    }
    REQUIRE_NOTHROW( sumkernel->analyze_dependencies() );
    REQUIRE_NOTHROW( sumkernel->execute() );
  }
    
  INFO( "mode: " << mode );
  data = global_sum->get_data(mycoord);
  CHECK( data[0]==(g*(g+1)/2) );
};

TEST_CASE( "explore the reduction kernel","[reduction][kernel][56]" ) {
  mpi_distribution
    *local_scalar = new mpi_block_distribution(decomp,1,-1),
    *global_scalar = new mpi_replicated_distribution(decomp);
  CHECK( local_scalar->volume(mycoord)==1 );
  CHECK( local_scalar->first_index_r(mycoord).coord(0)==mytid );
  CHECK( global_scalar->volume(mycoord)==1 );
  CHECK( global_scalar->first_index_r(mycoord).coord(0)==0 );
  auto local_value = std::shared_ptr<object>( new mpi_object(local_scalar) ),
    global_value = std::shared_ptr<object>( new mpi_object(global_scalar) );
  REQUIRE_NOTHROW( local_value->allocate() );
  double *data = local_value->get_data(mycoord); data[0] = mytid;
  mpi_kernel *reduction;

  int psqrt = sqrt(ntids);
  if (psqrt*psqrt<ntids) {
    printf("Test [56] needs square number of processors\n"); return; }

  printf("collective strategy disabled\n"); return;

  SECTION( "send/recv" ) {
    REQUIRE_NOTHROW( arch->set_collective_strategy_ptp() );
    REQUIRE_NOTHROW( reduction = new mpi_reduction_kernel(local_value,global_value) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    //    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->local_nmessages()==ntids+1 );
  }
  SECTION( "grouping" ) {
    REQUIRE_NOTHROW( arch->set_collective_strategy_group() );
    REQUIRE_NOTHROW( env->set_processor_grouping(psqrt) );
    REQUIRE_NOTHROW( reduction = new mpi_reduction_kernel(local_value,global_value) );
    REQUIRE_NOTHROW( reduction->analyze_dependencies() );
    //    REQUIRE_NOTHROW( reduction->gather_statistics() );
    CHECK( reduction->local_nmessages()==2*psqrt );
  }
}

mpi_sparse_matrix *diffusion1d( distribution *blocked,processor_coordinate mycoord)  { 
  index_int globalsize = blocked->global_size().at(0);
  mpi_sparse_matrix *A = new mpi_sparse_matrix(blocked);
  for (int row=blocked->first_index_r(mycoord).coord(0); row<=blocked->last_index_r(mycoord).coord(0); row++) {
    int col;
    col = row;     A->add_element(row,col,2.);
    col = row+1; if (col<globalsize)
		   A->add_element(row,col,-1.);
    col = row-1; if (col>=0)
		   A->add_element(row,col,-1.);
  }
  return A;
}

TEST_CASE( "cg matvec kernels","[cg][kernel][sparse][60]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ),
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ),
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ),
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ),
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ),
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ),
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  // the sparse matrix
  mpi_sparse_matrix *A;
  REQUIRE_NOTHROW( A = diffusion1d(blocked,mycoord) );  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "matvec" ) {
    kernel *matvec;
    REQUIRE_NOTHROW( A = new mpi_sparse_matrix( blocked ) );

    int test;
    index_int
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
    SECTION( "diagonal matrix" ) {
      test = 1;
      for (index_int row=my_first; row<=my_last; row++) {
	REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
      }
    }
    SECTION( "threepoint matrix" ) {
      test = 2;
      index_int globalsize = blocked->global_size().at(0);
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
    for (index_int row=0; row<blocked->volume(mycoord); row++)
      pdata[row] = 3.;
    REQUIRE_NOTHROW( matvec = new mpi_spmvp_kernel( p,q,A ) );
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
	if (row==0 || row==blocked->global_size().at(0)-1)
	  CHECK( qdata[lrow]==Approx(3.) );
	else
	  CHECK( qdata[lrow]==Approx(0.) );
	break;
      }
    }
  }

}

TEST_CASE( "cg norm kernels","[kernel][cg][norm][61]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ),
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "r_norm squared" ) {
    kernel *r_norm;
    double
      *rdata = r->get_data(mycoord), *rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.;
    }
    REQUIRE_NOTHROW( r_norm = new mpi_normsquared_kernel( r,rnorm ) );
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_size().at(0);
    CHECK( g==nlocal*ntids );
    CHECK( rrdata[0]==Approx(4*g) );
  }
  
  SECTION( "r_norm1" ) {
    kernel *r_norm;
    double
      *rdata = r->get_data(mycoord), *rrdata = rnorm->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.;
    }
    SECTION( "send/recv strategy" ) {
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( r_norm = new mpi_norm_kernel( r,rnorm ) );
    }
    SECTION( "grouping strategy" ) {
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = new mpi_norm_kernel( r,rnorm ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_size().at(0);
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
      REQUIRE_NOTHROW( r_norm = new mpi_norm_kernel( r,rnorm ) );
    }
    SECTION( "grouping strategy" ) {
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( r_norm = new mpi_norm_kernel( r,rnorm ) );
    }
    REQUIRE_NOTHROW( r_norm->analyze_dependencies() );
    REQUIRE_NOTHROW( r_norm->execute() );
    index_int g = r->global_size().at(0);
    CHECK( pow(rrdata[0],2)==Approx(g*(g+1)*(2*g+1)/6.) );
  }
  
}

TEST_CASE( "cg inner product kernels","[kernel][cg][inprod][62]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "rho_inprod" ) {
    const char *mode;
    mpi_innerproduct_kernel *rho_inprod;
    double
      *rdata = r->get_data(mycoord), *zdata = z->get_data(mycoord), *rrdata = rr->get_data(mycoord);
    for (int i=0; i<nlocal; i++) {
      rdata[i] = 2.; zdata[i] = mytid*nlocal+i;
    }
    
    SECTION( "send/recv strategy" ) {
      mode = "reduction kernel uses send/recv";
      arch->set_collective_strategy_ptp();
      REQUIRE_NOTHROW( rho_inprod = new mpi_innerproduct_kernel( r,z,rr ) );
    }
    SECTION( "grouping strategy" ) {
      mode = "reduction kernel uses grouping";
      arch->set_collective_strategy_group();
      REQUIRE_NOTHROW( rho_inprod = new mpi_innerproduct_kernel( r,z,rr ) );
    }
    REQUIRE_NOTHROW( rho_inprod->analyze_dependencies() );
    REQUIRE_NOTHROW( rho_inprod->execute() );
    {
      kernel *prekernel;
      REQUIRE_NOTHROW( prekernel = rho_inprod->get_prekernel() );
      CHECK( prekernel->get_n_in_objects()==2 );
    }
    index_int g = r->global_size().at(0);
    CHECK( rrdata[0]==Approx(g*(g-1)) );
  }
  
}

TEST_CASE( "cg vector kernels","[kernel][cg][axpy][63]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "copy" ) {
    kernel *rrcopy;
    int n;
    double *rdata,*sdata;
    
    SECTION( "scalar" ) {
      n = 1;
      rdata = rr->get_data(mycoord); sdata = rrp->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata[i] = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = new mpi_copy_kernel( rr,rrp ) );
    }
    SECTION( "vector" ) {
      n = nlocal;
      rdata = r->get_data(mycoord); sdata = z->get_data(mycoord);
      for (int i=0; i<n; i++)
	rdata[i] = 2.*(mytid*n+i);
      REQUIRE_NOTHROW( rrcopy = new mpi_copy_kernel( r,z ) );
    }
    
    REQUIRE_NOTHROW( rrcopy->analyze_dependencies() );
    REQUIRE_NOTHROW( rrcopy->execute() );
    for (int i=0; i<n; i++)
      CHECK( sdata[i]==Approx(2.*(mytid*n+i)) );
  }

  SECTION( "add" ) {
    kernel *sum,*makex,*makez;
    REQUIRE_NOTHROW( makex = new mpi_origin_kernel(x) );
    REQUIRE_NOTHROW( makez = new mpi_origin_kernel(z) );
    REQUIRE_NOTHROW( sum = new mpi_sum_kernel(x,z,xnew) );
    algorithm *queue = new mpi_algorithm(decomp);
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
  
}

TEST_CASE( "cg scalar kernels","[kernel][cg][scalar][64]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "scalar kernel error catching" ) {
    kernel *beta_calc;
    std::shared_ptr<object> rr;
    REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new mpi_object(blocked) ) );
    REQUIRE_THROWS( beta_calc = new mpi_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_THROWS( beta_calc = new mpi_scalar_kernel( rrp,"/",rr,beta ) );
    REQUIRE_THROWS( beta_calc = new mpi_scalar_kernel( beta,"/",rrp,rr ) );
  }

  SECTION( "beta_calc" ) {
    kernel *beta_calc;
    CHECK_NOTHROW( rr->get_data(mycoord)[0] = 5. );
    CHECK_NOTHROW( rrp->get_data(mycoord)[0] = 4. );
    REQUIRE_NOTHROW( beta_calc = new mpi_scalar_kernel( rr,"/",rrp,beta ) );
    REQUIRE_NOTHROW( beta_calc->analyze_dependencies() );
    REQUIRE_NOTHROW( beta_calc->execute() );
    double *bd;
    REQUIRE_NOTHROW( bd = beta->get_data(mycoord) );
    CHECK( *bd==Approx(1.25) );
  }

}

TEST_CASE( "cg update kernels","[kernel][cg][update][65]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  {
    double threeval = 3.;
    auto three = std::shared_ptr<object>( new mpi_object(scalar) );
    REQUIRE_NOTHROW( three->allocate() );
    REQUIRE_NOTHROW( three->set_value(threeval) );
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
      three = std::shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_THROWS( pupdate = new mpi_axbyz_kernel
		      ( '+',three,z, '+',beta,pold, p ) );
    }
    SECTION( "pp test s2" ) {
      beta = std::shared_ptr<object>( new mpi_object(blocked) );
      REQUIRE_THROWS( pupdate = new mpi_axbyz_kernel
		      ( '+',three,z, '+',beta,pold, p ) );
    }
    SECTION( "pp" ) {
      REQUIRE_NOTHROW( pupdate = new mpi_axbyz_kernel
		       ( '+',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      std::vector<dependency*> deps;
      REQUIRE_NOTHROW( deps = pupdate->get_dependencies() );
      CHECK( deps.size()==4 );
      std::shared_ptr<task> tsk; REQUIRE_NOTHROW( tsk = pupdate->get_tasks().at(0) );
      auto msgs = tsk->get_receive_messages(); CHECK( msgs.size()==4 );
      for ( auto msg : msgs ) {
	INFO( "message: " << msg->as_string() );
	std::shared_ptr<multi_indexstruct> global,local;
	REQUIRE_NOTHROW( global = msg->get_global_struct() );
	REQUIRE_NOTHROW( local = msg->get_local_struct() );
	index_int siz; REQUIRE_NOTHROW( siz = global->volume() );
	CHECK( siz==local->volume() );
	if (siz==1) {
	} else {
	  CHECK( siz==nlocal );
	}
      }
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(20.) );
    }
    SECTION( "mp" ) {
      REQUIRE_NOTHROW( pupdate = new mpi_axbyz_kernel
		       ( '-',three,z, '+',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(8.) );
    }
    SECTION( "pm" ) {
      REQUIRE_NOTHROW( pupdate = new mpi_axbyz_kernel
		       ( '+',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-8.) );
    }
    SECTION( "mm" ) {
      REQUIRE_NOTHROW( pupdate = new mpi_axbyz_kernel
		       ( '-',three,z, '-',beta,pold, p ) );
      REQUIRE_NOTHROW( pupdate->analyze_dependencies() );
      REQUIRE_NOTHROW( pupdate->execute() );
      for (int i=0; i<nlocal; i++)
	CHECK( pdata[i]==Approx(-20.) );
    }
  }
}

TEST_CASE( "cg preconditioner kernels","[kernel][cg][precon][66]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 1000;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ), 
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pold = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    rnorm = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  double one = 1.;
  
  // the sparse matrix
  mpi_sparse_matrix *A;
  REQUIRE_NOTHROW( A = diffusion1d(blocked,mycoord) );  
  REQUIRE_NOTHROW( r->allocate() );
  REQUIRE_NOTHROW( rr->allocate() );
  REQUIRE_NOTHROW( rrp->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( q->allocate() );
  REQUIRE_NOTHROW( rnorm->allocate() );
  REQUIRE_NOTHROW( alpha->allocate() );
  REQUIRE_NOTHROW( beta->allocate() );

  SECTION( "precon" ) {
    kernel *precon;
    REQUIRE_NOTHROW( precon = new mpi_preconditioning_kernel( r,z ) );
  }
}

TEST_CASE( "neuron kernel","[kernel][sparse][DAG][70]" ) {

  INFO( "mytid=" << mytid );
  int nlocal = 10, g = ntids*nlocal;
  distribution *blocked =
    new mpi_block_distribution(decomp,g);
  auto
    a = std::shared_ptr<object>( new mpi_object(blocked) ),
    b = std::shared_ptr<object>( new mpi_object(blocked) ),
    c1 = std::shared_ptr<object>( new mpi_object(blocked) ), 
    c2 = std::shared_ptr<object>( new mpi_object(blocked) ),
    d = std::shared_ptr<object>( new mpi_object(blocked) );

  mpi_sparse_matrix
    *Anarrow = new mpi_sparse_matrix( blocked ),
    *Awide   = new mpi_sparse_matrix( blocked );
  {
    index_int
      globalsize = blocked->global_size().at(0),
      my_first = blocked->first_index_r(mycoord).coord(0),
      my_last = blocked->last_index_r(mycoord).coord(0);
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
    *make_input = new mpi_origin_kernel(a),
    *fast_mult1 = new mpi_spmvp_kernel(a,b,Anarrow),
    *fast_mult2 = new mpi_spmvp_kernel(b,c1,Anarrow),
    *slow_mult  = new mpi_spmvp_kernel(a,c2,Awide),
    *assemble   = new mpi_sum_kernel(c1,c2,d);

  algorithm *queue = new mpi_algorithm(decomp);
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
      CHECK( a->volume(mycoord)==nlocal );
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

TEST_CASE( "Bilinear laplace","[distribution][multi][81]" ) {

#include "laplace_functions.h"

  if (ntids<4) { printf("need at least 4 procs for grid\n"); return; }

  // two-d decomposition
  int laplace_dim = 2;
  processor_coordinate *layout = arch->get_proc_layout(laplace_dim);
  decomp = new mpi_decomposition(arch,layout);
  processor_coordinate mycoord = decomp->coordinate_from_linear(mytid);

  mpi_distribution *nodes_dist;
  std::shared_ptr<object> nodes_in,nodes_out;
  mpi_kernel *bilinear_op;

  // number of elements (in each direction) is an input parameter
  index_int local_nnodes = 2;

  /* Create distributions */
  //SECTION( "set from global size" ) {
    nodes_dist = new mpi_block_distribution
      (decomp,std::vector<index_int>{local_nnodes*(*layout)[0],local_nnodes*(*layout)[1]});
    //}

  // SECTION( "set from local size" ) {
  //   nodes_dist = new mpi_block_distribution
  //     (decomp,std::vector<index_int>{local_nnodes,local_nnodes},-1);
  // }

  INFO( fmt::format("proc: {} out of decomposition: {}",
		    mycoord.as_string(),layout->as_string()) );

  /* Create the objects */
  nodes_in = std::shared_ptr<object>( new mpi_object(nodes_dist) ); nodes_in->set_name("nodes in");
  nodes_out = std::shared_ptr<object>( new mpi_object(nodes_dist) ); nodes_out->set_name("nodes out");

  stencil_operator *bilinear_stencil = new stencil_operator(2);
  bilinear_stencil->add( 0, 0);
  bilinear_stencil->add( 0,+1);
  bilinear_stencil->add( 0,-1);
  bilinear_stencil->add(-1, 0);
  bilinear_stencil->add(-1,+1);
  bilinear_stencil->add(-1,-1);
  bilinear_stencil->add(+1, 0);
  bilinear_stencil->add(+1,+1);
  bilinear_stencil->add(+1,-1);

  bilinear_op = new mpi_kernel(nodes_in,nodes_out);
  REQUIRE_NOTHROW( bilinear_op->add_sigma_stencil(bilinear_stencil) );
  bilinear_op->set_localexecutefn( &laplace_bilinear_fn );
  // if (mytid==0)
  //   bilinear_op->last_dependency()->set_tracing();
  //fmt::print("bilinear has type {}\n",bilinear_op->last_dependency()->type_as_string());

  algorithm *bilinear = new mpi_algorithm(decomp);
  bilinear->add_kernel( new mpi_setconstant_kernel(nodes_in,1.) );
  bilinear->add_kernel( bilinear_op );
  REQUIRE_NOTHROW( bilinear->analyze_dependencies() );

  bool
    pfirst_i = mycoord.coord(0)==0, pfirst_j = mycoord.coord(1)==0,
    plast_i = mycoord.coord(0)==layout->coord(0)-1,
    plast_j = mycoord.coord(1)==layout->coord(1)-1;
  INFO( fmt::format("Proc {} is first:{},{} last:{},{}",
		    mycoord.as_string(),pfirst_i,pfirst_j,plast_i,plast_j) );

  std::shared_ptr<task> ltask;
  REQUIRE_NOTHROW( ltask = bilinear_op->get_tasks().at(0) );
  std::shared_ptr<object> halo;
  REQUIRE_NOTHROW( halo = bilinear_op->get_beta_object(0) );
  auto pstruct = nodes_out->get_processor_structure(mycoord);
  auto hstruct = halo->get_processor_structure(mycoord);
  auto
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    hfirst = hstruct->first_index_r(), hlast = hstruct->last_index_r();
  INFO( fmt::format("Halo (multi: {}) has endpoints {}--{}",
		    hstruct->is_multi(),hfirst.as_string(),hlast.as_string() ) );
  CHECK( hfirst[0]==pfirst[0]-!pfirst_i );
  CHECK( hfirst[1]==pfirst[1]-!pfirst_j );
  CHECK( hlast[0]==plast[0]+!plast_i );
  CHECK( hlast[1]==plast[1]+!plast_j );
  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = ltask->get_receive_messages() );
  CHECK( msgs.size()==
	 9- 3*pfirst_i - 3*pfirst_j - 3*plast_i - 3*plast_j
	 + (pfirst_i && pfirst_j)
	 + (pfirst_i && plast_j)
	 + (plast_i && pfirst_j)
	 + (plast_i && plast_j)
	 );

  REQUIRE_NOTHROW( bilinear->execute() );

  {
    double *data;
    REQUIRE_NOTHROW( data = nodes_in->get_data(mycoord) );
    int
      ilo = mycoord.coord(0), // ( mycoord.coord(0)==0 ? 0 : 1 ),
      jlo = mycoord.coord(1), // ( mycoord.coord(1)==0 ? 0 : 1 ),
      ihi = local_nnodes, // - ( mycoord.coord(0)==layout->coord(0)-1 ? 0 : 1 ),
      jhi = local_nnodes; // - ( mycoord.coord(1)==layout->coord(1)-1 ? 0 : 1 );
    for (int i=ilo; i<ihi; i++) {
      for (int j=jlo; j<jhi; j++) {
	INFO( i << "," << j );
	CHECK( data[ i*local_nnodes + j ]==Approx(1.) );
      }
    }
  }

  {
    double *data;
    REQUIRE_NOTHROW( data = nodes_out->get_data(mycoord) );
    int
      ilo = ( mycoord.coord(0)==0 ? 1 : 0 ),
      jlo = ( mycoord.coord(1)==0 ? 1 : 0 ),
      ihi = local_nnodes - ( mycoord.coord(0)==layout->coord(0)-1 ? 1 : 0 ),
      jhi = local_nnodes - ( mycoord.coord(1)==layout->coord(1)-1 ? 1 : 0 );
    for (int i=ilo; i<ihi; i++) {
      for (int j=jlo; j<jhi; j++) {
	INFO( fmt::format("local i,j={},{} is global {},{}",i,j,i+pfirst[0],j+pfirst[1]) );
	CHECK( data[ i*local_nnodes + j ]==Approx(0.) );
      }
    }
  }
#if 0
#endif
}

TEST_CASE( "Masked distribution on output","[distribution][mask][111]" ) {
  if( ntids<2 ) { printf("masking requires two procs\n"); return; }

  INFO( " mytid = " << mytid );
  index_int localsize = 5; int alive;
  processor_mask *mask;

  alive = 1; processor_coordinate alive_proc = processor_coordinate1d(alive);
  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  REQUIRE_NOTHROW( mask->add(alive_proc) );


  mpi_distribution *block, *masked_block;
  REQUIRE_NOTHROW( block = new mpi_block_distribution(decomp,localsize,-1) );
  REQUIRE_NOTHROW( masked_block = new mpi_block_distribution(decomp,localsize,-1) );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );
  CHECK( masked_block->lives_on(alive_proc) );

  std::shared_ptr<object> whole_vector, masked_vector;
  REQUIRE_NOTHROW( whole_vector  = std::shared_ptr<object>( new mpi_object(block) ) );
  whole_vector->set_name("whole vector");
  REQUIRE_NOTHROW( masked_vector = std::shared_ptr<object>( new mpi_object(masked_block) ) );
  masked_vector->set_name("masked vector");

  double *data;
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  CHECK( block->lives_on(mycoord) );
  if (mytid==alive) {
    CHECK( masked_block->lives_on(mycoord) );
    CHECK( masked_vector->lives_on(mycoord) );
    REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
  } else {
    CHECK( !masked_block->lives_on(mycoord) );
    CHECK( !masked_vector->lives_on(mycoord) );
    REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
  }

  {
    double *indata,*outdata;
    REQUIRE_NOTHROW( indata = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) indata[i] = 1.;
    if (masked_vector->lives_on(mycoord)) {
      REQUIRE_NOTHROW( outdata = masked_vector->get_data(mycoord) );
      CHECK( outdata!=nullptr );
      for (index_int i=0; i<localsize; i++) outdata[i] = 2.;
    } else {
      REQUIRE_THROWS( outdata = masked_vector->get_data(mycoord) );
    }
  }
  mpi_kernel *copy = new mpi_kernel(whole_vector,masked_vector);
  copy->last_dependency()->set_type_local();
  copy->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy->analyze_dependencies() );

  
  REQUIRE_NOTHROW( copy->execute() );

  if (masked_vector->lives_on(mycoord)) { double *outdata;
    REQUIRE_NOTHROW( outdata = masked_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) 
      CHECK( outdata[i] == 1. );
  } // output vector otherwise not defined
}

TEST_CASE( "Masked distribution on input","[distribution][mask][112]" ) {

  if( ntids<2 ) { printf("masking requires two procs\n"); return; }
  INFO( "mytid: " << mytid );

  index_int localsize = 5;
  processor_mask *mask;

  REQUIRE_NOTHROW( mask = new processor_mask(decomp) );
  processor_coordinate p0 = processor_coordinate1d(0);
  REQUIRE_NOTHROW( mask->add(p0) );

  mpi_distribution
    *block = new mpi_block_distribution(decomp,localsize,-1),
    *masked_block = new mpi_distribution( *block );
  REQUIRE_NOTHROW( masked_block->add_mask(mask) );

  auto
    whole_vector = std::shared_ptr<object>( new mpi_object(block) ),
    masked_vector = std::shared_ptr<object>( new mpi_object(masked_block) );
  double *data;
  REQUIRE_NOTHROW( whole_vector->allocate() );
  REQUIRE_NOTHROW( masked_vector->allocate() );
  CHECK_NOTHROW( data = whole_vector->get_data(mycoord) );
  CHECK( block->lives_on(mycoord) );
  if (mytid==0) {
    CHECK( masked_block->lives_on(mycoord) );
    REQUIRE_NOTHROW( data = masked_vector->get_data(mycoord) );
  } else {
    CHECK( !masked_block->lives_on(mycoord) );
    REQUIRE_THROWS( data = masked_vector->get_data(mycoord) );
  }

  // set the whole output to 1
  {
    double *outdata;
    REQUIRE_NOTHROW( outdata = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++) outdata[i] = 1.;
  }
  // set input to 2, only on the mask
  if (masked_vector->lives_on(mycoord)) {
    double *indata;
    REQUIRE_NOTHROW( indata = masked_vector->get_data(mycoord) );
    CHECK( indata!=nullptr );
    for (index_int i=0; i<localsize; i++) indata[i] = 2.;
  } else {
    REQUIRE_THROWS( masked_vector->get_data(mycoord) );
  }
  mpi_kernel *copy;
  REQUIRE_NOTHROW( copy = new mpi_copy_kernel(masked_vector,whole_vector) );

  {
    dependency *dep;
    REQUIRE_NOTHROW( dep = copy->last_dependency() );
    REQUIRE_NOTHROW( copy->analyze_dependencies() );
    CHECK( !dep->get_beta_object()->has_mask() );
    REQUIRE_NOTHROW( copy->execute() );

    double *outdata;
    REQUIRE_NOTHROW( outdata = whole_vector->get_data(mycoord) );
    if (mytid==0) { // output has copied value
      for (index_int i=0; i<localsize; i++) 
	CHECK( outdata[i] == 2. );
    } else { // output has original value
      for (index_int i=0; i<localsize; i++) 
	CHECK( outdata[i] == 1. );
    }
  }
}

TEST_CASE( "Two masks","[distribution][mask][113]" ) {

  if( ntids<4 ) { printf("test 72 needs at least 4 procs\n"); return; }

  INFO( "mytid=" << mytid );
  index_int localsize = 5;
  processor_mask *mask1,*mask2;

  mask1 = new processor_mask(decomp);
  mask2 = new processor_mask(decomp);
  for (int tid=0; tid<ntids; tid+=2) {
    processor_coordinate ptid = processor_coordinate1d(tid);
    REQUIRE_NOTHROW( mask1->add(ptid) );
  }
  for (int tid=0; tid<ntids; tid+=4) {
    processor_coordinate ptid = processor_coordinate1d(tid);
    REQUIRE_NOTHROW( mask2->add(ptid) );
  }
  
  std::shared_ptr<object>
    whole_vector,masked_vector1,masked_vector2;
  {
    mpi_distribution
      *block = new mpi_block_distribution(decomp,localsize,-1);
    whole_vector = std::shared_ptr<object>( new mpi_object(block) );
    REQUIRE_NOTHROW( whole_vector->allocate() );
    double *data;
    REQUIRE_NOTHROW( data = whole_vector->get_data(mycoord) );
    for (index_int i=0; i<localsize; i++)
      data[i] = 1;
  }
  
  {
    mpi_distribution
      *masked_block1 = new mpi_block_distribution(decomp,localsize,-1);
    REQUIRE_NOTHROW( masked_block1->add_mask(mask1) );
    REQUIRE_NOTHROW( masked_vector1 = std::shared_ptr<object>( new mpi_object(masked_block1) ) );
    REQUIRE_NOTHROW( masked_vector1->allocate() );
    if (masked_vector1->lives_on(mycoord)) {
      double *data;
      REQUIRE_NOTHROW( data = masked_vector1->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++)
	data[i] = 2;
    }
  }
  {
    mpi_distribution
      *masked_block2 = new mpi_block_distribution(decomp,localsize,-1);
    REQUIRE_NOTHROW( masked_block2->add_mask(mask2) );
    REQUIRE_NOTHROW( masked_vector2 = std::shared_ptr<object>( new mpi_object(masked_block2) ) );
    REQUIRE_NOTHROW( masked_vector2->allocate() );
    if (masked_vector2->lives_on(mycoord)) {
      double *data;
      REQUIRE_NOTHROW( data = masked_vector2->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++)
	data[i] = 4;
    }
  }

  mpi_kernel
    *copy1 = new mpi_kernel(whole_vector,masked_vector1);
  copy1->last_dependency()->add_sigma_operator( ioperator("none") );
  copy1->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy1->analyze_dependencies() );
  REQUIRE_NOTHROW( copy1->execute() );

  {
    double *data1;
    if (masked_vector1->lives_on(mycoord)) {
      REQUIRE_NOTHROW( data1 = masked_vector1->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data1[i] == 1. );
    } else
      REQUIRE_THROWS( data1 = masked_vector1->get_data(mycoord) );
  }

  mpi_kernel
    *copy2 = new mpi_kernel(masked_vector1,masked_vector2);
  copy2->last_dependency()->add_sigma_operator( ioperator("none") );
  copy2->set_localexecutefn( &veccopy );
  REQUIRE_NOTHROW( copy2->analyze_dependencies() );
  REQUIRE_NOTHROW( copy2->execute() );

  {
    double *data2;
    if (masked_vector2->lives_on(mycoord)) {
      REQUIRE_NOTHROW( data2 = masked_vector2->get_data(mycoord) );
      for (index_int i=0; i<localsize; i++) 
	CHECK( data2[i] == 1. );
    } else
      REQUIRE_THROWS( data2 = masked_vector2->get_data(mycoord) );
  }
}

#if 0
TEST_CASE( "mask on replicated scalars","[distribution][mask][replicated][114]" ) {

  if (ntids<2) { printf("test 73 needs at least 2\n"); return; }
  INFO( "mytid=" << mytid );

  processor_mask *mask = new processor_mask(decomp);
  mask->add(new processor_coordinate1d(0));
  distribution *scalar = new mpi_replicated_distribution(decomp);
  distribution *scalars = new mpi_replicated_distribution(decomp);
  // mpi_distribution( *scalar ); copying doesn't work!?
  REQUIRE_NOTHROW( scalar->add_mask(mask) );
  CHECK( !scalars->has_mask() );

  std::shared_ptr<object> one,ones;
  REQUIRE_NOTHROW( one = std::shared_ptr<object>( new mpi_object(scalar) ) );
  if (mytid==0) {
    CHECK( one->lives_on(mycoord) );
    CHECK( one->first_index_r(mycoord).coord(0)==0 );
  } else {
    CHECK( !one->lives_on(mycoord) );
    REQUIRE_THROWS( one->first_index_r(mycoord).coord(0) );
  }
  one->allocate();
  double oneval = 1.; REQUIRE_NOTHROW( one->set_value(&oneval) );
  double *in_data;
  if (one->lives_on(mycoord)) {
    REQUIRE_NOTHROW( in_data = one->get_data(mycoord) );
    CHECK( in_data[0]==Approx(1.) );
  } else 
    REQUIRE_THROWS( in_data = one->get_data(mycoord) );

  REQUIRE_NOTHROW( ones = std::shared_ptr<object>( new mpi_object(scalars) ) );
  CHECK( ones->first_index_r(mycoord).coord(0)==0 );
  ones->allocate();

  kernel *bcast; std::shared_ptr<task> tsk;
  REQUIRE_NOTHROW( bcast = new mpi_bcast_kernel(one,ones) );
  REQUIRE_NOTHROW( bcast->set_comm_trace_level(comm_trace_level::EXEC) );
  CHECK( ones->lives_on(mycoord) );
  REQUIRE_NOTHROW( bcast->get_dependencies().at(0)->ensure_beta_distribution(ones) );
  distribution *beta;
  REQUIRE_NOTHROW( beta = bcast->get_dependencies().at(0)->get_beta_distribution() );
  CHECK( beta->volume(mycoord)==1 );
  CHECK( beta->first_index_r(mycoord).coord(0)==0 );

  REQUIRE_NOTHROW( bcast->analyze_dependencies() );
  dependency *dep; REQUIRE_NOTHROW( dep = bcast->last_dependency() );
  std::shared_ptr<object> halo; REQUIRE_NOTHROW( halo = dep->get_beta_object() );
  CHECK( halo->lives_on(mycoord) );

  REQUIRE_NOTHROW( tsk = bcast->get_tasks().at(0) );
  std::vector<message*> msgs;
  REQUIRE_NOTHROW( msgs = tsk->get_receive_messages() );
  CHECK( msgs.size()==1 );
  CHECK( msgs.at(0)->get_receiver()->coord(0)==mytid );
  CHECK( msgs.at(0)->get_sender().coord(0)==0 );
  CHECK( msgs.at(0)->get_global_struct()->local_size()==1 );
  CHECK( msgs.at(0)->get_local_struct()->local_size()==1 );

  REQUIRE_NOTHROW( msgs = tsk->get_send_messages() );
  if (one->lives_on(mycoord)) {
    CHECK( msgs.size()==ntids );
    //for (auto m=msgs->begin(); m!=msgs->end(); ++m) {
    for ( auto m : msgs ) {
      CHECK( m->get_global_struct()->local_size()==1 );
      CHECK( m->get_local_struct()->local_size()==1 );
    }
  } else
    CHECK( msgs.size()==0 );

  REQUIRE_NOTHROW( bcast->execute() );
  double *halo_data;
  REQUIRE_NOTHROW( halo_data = halo->get_data(mycoord) );
  CHECK( halo_data[0]==Approx(1.) );
  double *out_data;
  REQUIRE_NOTHROW( out_data = ones->get_data(mycoord) );
  CHECK( out_data[0]==Approx(1.) );
}
#endif
