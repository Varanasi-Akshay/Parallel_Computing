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
 **** conjugate gradient tests 
 **** (individual kernels are tested in unittest_ops)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "trace kernels","[kernel][1]" ) {
  distribution *scalar; REQUIRE_NOTHROW( scalar = new mpi_replicated_distribution(decomp) );
  std::shared_ptr<object> rr;
  REQUIRE_NOTHROW( rr = std::shared_ptr<object>( new mpi_object(scalar) ) );
  REQUIRE_NOTHROW( rr->allocate() );
  double *data;
  REQUIRE_NOTHROW( data = rr->get_data(mycoord) );
  data[0] = 3.14;

  mpi_algorithm *queue; REQUIRE_NOTHROW( queue = new mpi_algorithm(decomp) );
  REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(rr) ) );
  mpi_kernel *trace; REQUIRE_NOTHROW( trace = new mpi_trace_kernel(rr,std::string("norm")) );
  trace->set_name(std::string("trace-cg-1"));
  REQUIRE_NOTHROW( queue->add_kernel(trace) );
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
}

TEST_CASE( "orthogonality relations","[cg][kernel][ortho][2]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 2;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto
    x = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ),
    zr = std::shared_ptr<object>( new mpi_object(scalar) ),
    xr = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ),
    sbzero = std::shared_ptr<object>( new mpi_object(scalar) ),
    one = std::shared_ptr<object>( new mpi_object(scalar) );
  double one_value = 1.;
  one->set_value( one_value );
  kernel *makeone = new mpi_origin_kernel(one);
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);
  const char *mode;

  {
    // z,r independent: r linear, z constant 1
    double *rdata,*zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      rdata[i] = mytid*nlocal+i+1;
    kernel *maker = new mpi_origin_kernel(r);
    REQUIRE_NOTHROW( zdata = z->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      zdata[i] = 1.;
    kernel *makez = new mpi_origin_kernel(z);

    // rr = r' r
    kernel *rr_inprod;
    REQUIRE_NOTHROW( rr_inprod = new mpi_normsquared_kernel( r,rr ) );    

    // zr = r' z
    kernel *zr_inprod;
    REQUIRE_NOTHROW( zr_inprod = new mpi_innerproduct_kernel( r,z,zr ) );
    
    // alpha = (r'z)/(r'r)
    kernel *coef_compute;
    REQUIRE_NOTHROW( coef_compute = new mpi_scalar_kernel( zr,"/",rr,alpha ) );
    
    // x = z - alpha r
    kernel *update;
    REQUIRE_NOTHROW( update = new mpi_axbyz_kernel( '+',one,z,'-',alpha,r, x ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    kernel *check;
    REQUIRE_NOTHROW( check = new mpi_innerproduct_kernel( x,r,xr ) );
    
    SECTION( "by kernels" ) {
      mode = "by kernels";
      REQUIRE_NOTHROW( rr_inprod->analyze_dependencies() );
      REQUIRE_NOTHROW( zr_inprod->analyze_dependencies() );
      REQUIRE_NOTHROW( coef_compute->analyze_dependencies() );
      REQUIRE_NOTHROW( update->analyze_dependencies() );
      REQUIRE_NOTHROW( check->analyze_dependencies() );

      REQUIRE_NOTHROW( rr_inprod->execute() );      
      REQUIRE_NOTHROW( zr_inprod->execute() );      
      REQUIRE_NOTHROW( coef_compute->execute() );
      REQUIRE_NOTHROW( update->execute() );
      REQUIRE_NOTHROW( check->execute() );
    }
    SECTION( "by task queue" ) {
      mode = "by queue";
      algorithm *queue = new mpi_algorithm(decomp);
      REQUIRE_NOTHROW( queue->add_kernel(makeone) );
      REQUIRE_NOTHROW( queue->add_kernel(maker) );
      REQUIRE_NOTHROW( queue->add_kernel(makez) );
      REQUIRE_NOTHROW( queue->add_kernel(rr_inprod) );
      REQUIRE_NOTHROW( queue->add_kernel(zr_inprod) );
      REQUIRE_NOTHROW( queue->add_kernel(coef_compute) );
      REQUIRE_NOTHROW( queue->add_kernel(update) );
      REQUIRE_NOTHROW( queue->add_kernel(check) );

      REQUIRE_NOTHROW( queue->analyze_dependencies() );
      REQUIRE_NOTHROW( queue->execute() );      
    }

    INFO( "mode: " << mode );

    // check a bunch of intermediate results
    double *quad; int g=blocked->global_size().at(0);
    double zr_value = g*(g+1)/2, rr_value = g*(g+1)*(2*g+1)/6,
      alpha_value = zr_value/rr_value;
    CHECK_NOTHROW( quad = zr->get_data(mycoord) );
    CHECK( quad[0]==Approx( zr_value ) );
    CHECK_NOTHROW( quad = rr->get_data(mycoord) );
    CHECK( quad[0]==Approx( rr_value ) );
    CHECK_NOTHROW( quad = alpha->get_data(mycoord) );
    CHECK( quad[0]==Approx( alpha_value ) );
  }

  // => r' x = 0
  {
    double *isthiszero;
    CHECK( xr->volume(mycoord)==1 );
    REQUIRE_NOTHROW( isthiszero = xr->get_data(mycoord) );
    CHECK( isthiszero[0]==Approx(0.) );
  }
}

TEST_CASE( "A-orthogonality relations","[cg][kernel][ortho][sparse][3]" ) {

  INFO( "mytid: " << mytid );

  int nlocal = 2;
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto 
    x = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) );
  REQUIRE_NOTHROW( x->allocate() );
  REQUIRE_NOTHROW( z->allocate() );
  REQUIRE_NOTHROW( r->allocate() );
  
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ),
    zr = std::shared_ptr<object>( new mpi_object(scalar) ),
    xr = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    sbzero = std::shared_ptr<object>( new mpi_object(scalar) ),
    one = std::shared_ptr<object>( new mpi_object(scalar) );
  double one_value = 1.;
  one->set_value( one_value );
  kernel *makeone = new mpi_origin_kernel(one);
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);
  const char *mode;

  mpi_sparse_matrix *A; int test;
  REQUIRE_NOTHROW( A = new mpi_sparse_matrix( blocked ) );

  index_int globalsize = blocked->global_size().at(0);
  for (int row=my_first; row<=my_last; row++) {
    int col;
    col = row;
    REQUIRE_NOTHROW( A->add_element(row,col,2.) );
    col = row+1;
    if (col<globalsize)
      REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
    col = row-1;
    if (col>=0)
      REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  }

  {
    // z,r independent: r linear, z constant 1
    double *rdata,*zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      rdata[i] = mytid*nlocal+i+1;
    kernel *maker = new mpi_origin_kernel(r);
    REQUIRE_NOTHROW( zdata = z->get_data(mycoord) );
    for (index_int i=0; i<nlocal; i++)
      zdata[i] = 1.;
    kernel *makez = new mpi_spmvp_kernel(r,z,A);

    // rr = r' r
    kernel *rr_inprod;
    REQUIRE_NOTHROW( rr_inprod = new mpi_normsquared_kernel( r,rr ) );    

    // zr = r' z
    kernel *zr_inprod;
    REQUIRE_NOTHROW( zr_inprod = new mpi_innerproduct_kernel( r,z,zr ) );
    
    // alpha = (r'z)/(r'r)
    kernel *coef_compute;
    REQUIRE_NOTHROW( coef_compute = new mpi_scalar_kernel( zr,"/",rr,alpha ) );
    
    // x = z - alpha r
    kernel *update;
    REQUIRE_NOTHROW( update = new mpi_axbyz_kernel( '+',one,z,'-',alpha,r, x ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    kernel *check;
    REQUIRE_NOTHROW( check = new mpi_innerproduct_kernel( x,r,xr ) );
    
    algorithm *queue = new mpi_algorithm(decomp);
    REQUIRE_NOTHROW( queue->add_kernel(makeone) );
    REQUIRE_NOTHROW( queue->add_kernel(maker) );
    REQUIRE_NOTHROW( queue->add_kernel(makez) );
    REQUIRE_NOTHROW( queue->add_kernel(rr_inprod) );
    REQUIRE_NOTHROW( queue->add_kernel(zr_inprod) );
    REQUIRE_NOTHROW( queue->add_kernel(coef_compute) );
    REQUIRE_NOTHROW( queue->add_kernel(update) );
    REQUIRE_NOTHROW( queue->add_kernel(check) );
    
    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );      
  }

  // => r' x = 0
  {
    double *isthiszero;
    CHECK( xr->volume(mycoord)==1 );
    REQUIRE_NOTHROW( isthiszero = xr->get_data(mycoord) );
    CHECK( isthiszero[0]==Approx(0.) );
  }
}

TEST_CASE( "power method","[sparse][10]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, nsteps = 4;
  distribution
    *blocked = new mpi_block_distribution(decomp,nlocal,-1),
    *scalar = new mpi_replicated_distribution(decomp);

  mpi_sparse_matrix *A; int test;
  REQUIRE_NOTHROW( A = new mpi_sparse_matrix( blocked ) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);

  const char *mat;
  SECTION( "diagonal matrix" ) {
    test = 1; mat = "diagonal";
    for (index_int row=my_first; row<=my_last; row++) {
      REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
    }
  }
  SECTION( "threepoint matrix" ) {
    test = 2; mat = "threepoint";
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
  INFO( "matrix test: " << mat );

  // create vectors, sharing storage
  auto xs = std::vector<std::shared_ptr<object>>(2*nsteps);
  for (int step=0; step<nsteps; step++) {
    REQUIRE_NOTHROW( xs[2*step] = std::shared_ptr<object>( new mpi_object(blocked) ) );
    xs[2*step]->set_name(fmt::format("xs[{}]",2*step));

    REQUIRE_NOTHROW( xs[2*step+1] = std::shared_ptr<object>( new mpi_object(blocked) ) );
    xs[2*step+1]->set_name(fmt::format("xs[{}]",2*step+1));
  }
  double *data0;
  REQUIRE_NOTHROW( xs[0]->allocate() ); // starting vector is allocated, everything else in halo
  REQUIRE_NOTHROW( data0 = xs[0]->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) data0[i] = 1.;

  // create lambda values
  auto
    norms = std::vector<std::shared_ptr<object>>(2*nsteps),
    lambdas = std::vector<std::shared_ptr<object>>(nsteps);
  for (int step=0; step<nsteps; step++) {
    std::shared_ptr<object> inobj,outobj;

    REQUIRE_NOTHROW( inobj = std::shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( inobj->allocate() );
    REQUIRE_NOTHROW( inobj->set_name(fmt::format("in-object-{}",step)));
    norms[2*step] = inobj;

    REQUIRE_NOTHROW( outobj = std::shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( outobj->allocate() );
    REQUIRE_NOTHROW( outobj->set_name(fmt::format("out-object-{}",step)));
    norms[2*step+1] = outobj;

    REQUIRE_NOTHROW( lambdas[step] = std::shared_ptr<object>( new mpi_object(scalar) ) );
    REQUIRE_NOTHROW( lambdas[step]->allocate() );
  }
  
  algorithm *queue = new mpi_algorithm(decomp);  
  // originate the first vector
  REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(xs[0]) ) );
  // loop through the rest
  for (int step=0; step<nsteps; step++) {  
    kernel *matvec, *scaletonext,*getnorm,*computelambda;
    // matrix-vector product
    REQUIRE_NOTHROW( matvec = new mpi_spmvp_kernel( xs[2*step],xs[2*step+1],A ) );
    REQUIRE_NOTHROW( matvec->set_name(fmt::format("mvp-{}.",step)) );
    REQUIRE_NOTHROW( queue->add_kernel(matvec) );
    // norms to compare
    REQUIRE_NOTHROW( getnorm = new mpi_norm_kernel( xs[2*step],norms[2*step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( getnorm = new mpi_norm_kernel( xs[2*step+1],norms[2*step+1] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( computelambda =
	    new mpi_scalar_kernel( norms[2*step+1],"/",norms[2*step],lambdas[step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(computelambda) );
    if (step<nsteps-1) {
      // scale down for the next iteration
      REQUIRE_NOTHROW(scaletonext =
	    new mpi_scaledown_kernel( lambdas[step],xs[2*step+1],xs[2*step+2] ) );
      REQUIRE_NOTHROW( queue->add_kernel(scaletonext) );
    }
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  for (int step=0; step<nsteps; step++) {
    INFO( "step " << step << ", power input is " << xs[2*step]->data_status_as_string()
	  << ", power output is " << xs[2*step+1]->data_status_as_string() );
    if (step==0) // starting vector was explicitly allocated
      CHECK( xs[2*step]->has_data_status_allocated() );
    else
      CHECK( ( !arch->get_can_embed_in_beta() ||  xs[2*step]->has_data_status_inherited() ) );
  }

  REQUIRE_NOTHROW( queue->execute() );
  if (test==1) {
    for (int step=0; step<nsteps; step++) {
      INFO( "step: " << step );
      double
	*indata = xs[2*step]->get_data(mycoord),
	*outdata = xs[2*step+1]->get_data(mycoord);
      double *n0,*n1,*l;
      REQUIRE_NOTHROW( n0 = norms[2*step]->get_data(mycoord) );
      REQUIRE_NOTHROW( n1 = norms[2*step+1]->get_data(mycoord) );
      REQUIRE_NOTHROW( l = lambdas[step]->get_data(mycoord) );
      CHECK( *n1==Approx(2*(*n0)) );
      CHECK( *l!=Approx(0.) );
    }
  }
}

TEST_CASE( "power method with data reuse","[reuse][11]" ) {
  INFO( "mytid=" << mytid );

  int nlocal = 10, nsteps = 4;
  distribution
    *blocked = new mpi_block_distribution(decomp,nlocal,-1),
    *scalar = new mpi_replicated_distribution(decomp);

  mpi_sparse_matrix *A; int test;
  REQUIRE_NOTHROW( A = new mpi_sparse_matrix( blocked ) );
  index_int
    my_first = blocked->first_index_r(mycoord).coord(0),
    my_last = blocked->last_index_r(mycoord).coord(0);

  // const char *mat;
  // // SECTION( "diagonal matrix" ) {
  // //   test = 1; mat = "diagonal";
  // //   for (index_int row=my_first; row<=my_last; row++) {
  // //     REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
  // //   }
  // // }
  // // SECTION( "threepoint matrix" ) {
  // {
  //   test = 2; mat = "threepoint";
  //   index_int globalsize = blocked->global_size().at(0);
  //   for (int row=my_first; row<=my_last; row++) {
  //     int col;
  //     col = row;     REQUIRE_NOTHROW( A->add_element(row,col,2.) );
  //     col = row+1; if (col<globalsize)
  // 		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  //     col = row-1; if (col>=0)
  // 		     REQUIRE_NOTHROW( A->add_element(row,col,-1.) );
  //   }
  // }
  // INFO( "matrix test: " << mat );

  // create vectors, sharing storage
  std::shared_ptr<object> xvector,axvector;
  REQUIRE_NOTHROW( xvector = std::shared_ptr<object>( new mpi_object(blocked) ) );
  xvector->set_name("x0");
  REQUIRE_NOTHROW( axvector = std::shared_ptr<object>( new mpi_object(blocked) ) );
  axvector->set_name("Ax0");

  double *data0;
  REQUIRE_NOTHROW( xvector->allocate() );
  REQUIRE_NOTHROW( data0 = xvector->get_data(mycoord) );
  for (int i=0; i<nlocal; i++) data0[i] = 1.;

  // create lambda values
  auto
    norms = std::vector<std::shared_ptr<object>>(2*nsteps),
    lambdas = std::vector<std::shared_ptr<object>>(nsteps);
  double
    *normvalue = new double[2*nsteps],
    *lambdavalue = new double[nsteps];
  for (int step=0; step<nsteps; step++) {
    std::shared_ptr<object> inobj,outobj; fmt::MemoryWriter inname,outname;

    REQUIRE_NOTHROW( inobj = std::shared_ptr<object>( new mpi_object(scalar) ) );
    inname.write("in-object-{}",step);
    REQUIRE_NOTHROW( inobj->set_name(inname.str()) );
    norms[2*step] = inobj;

    REQUIRE_NOTHROW( outobj = std::shared_ptr<object>( new mpi_object(scalar) ) );
    outname.write("out-object-{}",step);
    REQUIRE_NOTHROW( outobj->set_name(outname.str()) );
    norms[2*step+1] = outobj;

    REQUIRE_NOTHROW
      ( lambdas[step] = std::shared_ptr<object>( new mpi_object(scalar,lambdavalue+step) ) );
  }
  
  algorithm *queue = new mpi_algorithm(decomp);  
  // originate the first vector
  REQUIRE_NOTHROW( queue->add_kernel( new mpi_origin_kernel(xvector) ) );
  // loop through the rest
  for (int step=0; step<nsteps; step++) {  
    kernel *matvec, *scaletonext,*getnorm,*computelambda;
    if (step>0) {
      REQUIRE_NOTHROW( axvector = std::shared_ptr<object>( new mpi_object(blocked,axvector) ) );
      xvector->set_name(fmt::format("axvector-{}",step));
    }
    // matrix-vector product
    REQUIRE_NOTHROW( matvec = new mpi_diffusion_kernel( xvector,axvector ) );
    REQUIRE_NOTHROW( matvec->set_name(fmt::format("mvp-{}",step)) );
    REQUIRE_NOTHROW( queue->add_kernel(matvec) );
    // norms to compare
    REQUIRE_NOTHROW( getnorm = new mpi_norm_kernel( xvector,norms[2*step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( getnorm = new mpi_norm_kernel( axvector,norms[2*step+1] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( computelambda =
		     new mpi_scalar_kernel( norms[2*step+1],"/",norms[2*step],lambdas[step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(computelambda) );
    REQUIRE_NOTHROW( queue->analyze_kernel_dependencies() );
    if (step<nsteps-1) {
      // scale down for the next iteration
      REQUIRE_NOTHROW( xvector = std::shared_ptr<object>( new mpi_object(blocked,xvector) ) );
      xvector->set_name(fmt::format("xvector-{}",step));
      REQUIRE_NOTHROW(scaletonext =
		      new mpi_scaledown_kernel( lambdas[step],axvector,xvector ) );
      REQUIRE_NOTHROW( queue->add_kernel(scaletonext) );
    }
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );
  if (mytid==0) {
    printf("Lambda values (version %d): ",test);
    for (int step=0; step<nsteps; step++) printf("%e ",lambdavalue[step]);
    printf("\n");
  }
}

TEST_CASE( "diffusion","[15]" ) {

  int nlocal = 10000, nsteps = 1,nglobal = nlocal*arch->nprocs();

  distribution *blocked = new mpi_block_distribution(decomp,nglobal);
  auto xs = new std::vector<std::shared_ptr<object>>;
  for (int step=0; step<=nsteps; step++) {
    xs->push_back( std::shared_ptr<object>( new mpi_object(blocked) ) );
  }
  // set initial condition to a delta function
  REQUIRE_NOTHROW( xs->at(0)->allocate() );
  double *data = xs->at(0)->get_data(mycoord);
  for (index_int i=0; i<nlocal; i++)
    data[i] = 0.0;
  if (mytid==0)
    data[0] = 1.;

  algorithm *queue;
  queue = new mpi_algorithm(decomp);

  queue->add_kernel( new mpi_origin_kernel(xs->at(0)) );
  for (int step=0; step<nsteps; step++) {
    REQUIRE_NOTHROW( queue->add_kernel( new mpi_diffusion_kernel( xs->at(step),xs->at(step+1) ) ) );
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  auto tsks = queue->get_tasks();
  for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    if (!(*t)->has_type_origin()) {
      auto msgs = (*t)->get_receive_messages();
      if (mytid==0 || mytid==ntids-1)
	CHECK( msgs.size()==2 );
      else
	CHECK( msgs.size()==3 );
      //for (auto m=msgs->begin(); m!=msgs->end(); ++m) {
      for ( auto m : msgs ) {
	auto snd = m->get_sender();
	if (snd.coord(0)==mytid-1 || snd.coord(0)==mytid+1) {
	  CHECK( m->get_global_struct()->local_size(0)==1 );
	}
      }
    }
  }

  REQUIRE_NOTHROW( queue->execute() );
}

TEST_CASE( "cg algorithm","[sparse][20]" ) {

  int nlocal = 10;

  // a bunch of vectors, block distributed
  distribution *blocked = new mpi_block_distribution(decomp,nlocal,-1);
  auto 
    x = std::shared_ptr<object>( new mpi_object(blocked) ), 
    xnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    z = std::shared_ptr<object>( new mpi_object(blocked) ),
    r = std::shared_ptr<object>( new mpi_object(blocked) ),
    rnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    p = std::shared_ptr<object>( new mpi_object(blocked) ), 
    pnew = std::shared_ptr<object>( new mpi_object(blocked) ),
    q = std::shared_ptr<object>( new mpi_object(blocked) ), 
    qold = std::shared_ptr<object>( new mpi_object(blocked) );

  r->allocate();

  // scalars, all redundantly replicated
  distribution *scalar = new mpi_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new mpi_object(scalar) ), 
    rrp = std::shared_ptr<object>( new mpi_object(scalar) ),
    pap = std::shared_ptr<object>( new mpi_object(scalar) ),
    alpha = std::shared_ptr<object>( new mpi_object(scalar) ), 
    beta = std::shared_ptr<object>( new mpi_object(scalar) );
  int n_iterations=5;
  auto 
    rnorms = std::vector<std::shared_ptr<object>>(n_iterations),
    rrzero = std::vector<std::shared_ptr<object>>(n_iterations),
    ppzero = std::vector<std::shared_ptr<object>>(n_iterations);
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = std::shared_ptr<object>( new mpi_object(scalar) );
    rrzero[it] = std::shared_ptr<object>( new mpi_object(scalar) );
    ppzero[it] = std::shared_ptr<object>( new mpi_object(scalar) );
  }

  // the sparse matrix
  mpi_sparse_matrix *A;
  { 
    index_int globalsize = blocked->global_size().at(0);
    int mytid = arch->mytid();
    index_int
      first = blocked->first_index_r(mycoord).coord(0),
      last = blocked->last_index_r(mycoord).coord(0);

    A = new mpi_sparse_matrix(blocked);
    REQUIRE_NOTHROW( x->allocate() ); REQUIRE_NOTHROW( r->allocate() );
    double *xdata = x->get_data(mycoord), *rdata = r->get_data(mycoord);
    for (int row=first; row<=last; row++) {
      xdata[row-first] = 1.; rdata[row-first] = 1.;
      int col;
      col = row;     A->add_element(row,col,2.);
      col = row+1; if (col<globalsize)
    		     A->add_element(row,col,-1.);
      col = row-1; if (col>=0)
    		     A->add_element(row,col,-1.);
    }
  }
  
  double *one_value = new double; one_value[0] = 1.;
  auto one = std::shared_ptr<object>( new mpi_object(scalar,one_value ));
  
  // let's define the steps of the loop body
  algorithm *queue = new mpi_algorithm(decomp);
  queue->add_kernel( new mpi_origin_kernel(one) );
  queue->add_kernel( new mpi_origin_kernel(x) );
  queue->add_kernel( new mpi_origin_kernel(r) );
  queue->add_kernel( new mpi_copy_kernel(r,p) );

  SECTION( "one iteration without copy" ) {
    kernel *precon = new mpi_copy_kernel( r,z );
    queue->add_kernel(precon);

    kernel *rho_inprod = new mpi_innerproduct_kernel( r,z,rr );
    rho_inprod->set_name("compute rho");
    queue->add_kernel(rho_inprod);

    kernel *pisz = new mpi_copy_kernel( z,pnew );
    pisz->set_name("copy z to p");
    queue->add_kernel(pisz);

    kernel *matvec = new mpi_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    kernel *pap_inprod = new mpi_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    kernel *alpha_calc = new mpi_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    kernel *xupdate = new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    kernel *rupdate = new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    kernel *rrtest = new mpi_innerproduct_kernel( z,rnew,rrzero[0] );
    queue->add_kernel(rrtest) ; rrtest->set_name("test rr orthogonality");

    kernel *xcopy = new mpi_copy_kernel( xnew,x );
    queue->add_kernel(xcopy); xcopy     ->set_name("copy x");

    kernel *rcopy = new mpi_copy_kernel( rnew,r );
    queue->add_kernel(rcopy); rcopy     ->set_name("copy r");
  
    kernel *rnorm = new mpi_norm_kernel( r,rnorms[0] );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    queue->analyze_dependencies();
    queue->execute();

    double *data;

    data = z->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++)
      CHECK( data[i]==Approx(1.) );

    data = rr->get_data(mycoord);
    CHECK( data[0]==Approx(ntids*nlocal) );

    data = pnew->get_data(mycoord);
    for (index_int i=0; i<nlocal; i++)
      CHECK( data[i]==Approx(1.) );

    data = q->get_data(mycoord);
    { int lo,hi,i;
      if (mytid==0) lo=1; else lo=0;
      if (mytid==ntids-1) hi=nlocal-2; else hi=nlocal-1;
      for (i=0; i<lo; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data[i]==Approx(1.) );
      }
      for (i=lo; i<=hi; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data[i]==Approx(0.) );
      }
      for (i=hi+1; i<nlocal; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data[i]==Approx(1.) );
      }
    }

    data = pap->get_data(mycoord);
    CHECK( data[0]==Approx(2.) );

    data = alpha->get_data(mycoord);
    CHECK( data[0]==Approx(ntids*nlocal/2.) );

    data = rrzero[0]->get_data(mycoord);
    CHECK( data[0]==Approx(0.) );

  }

  SECTION( "two iterations" ) {

    double *data;

    auto
      x0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      r0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      z0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      p0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      q0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      xnew0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      rnew0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      pnew0 = std::shared_ptr<object>( new mpi_object(blocked) ),
      rr0    = std::shared_ptr<object>( new mpi_object(scalar) ),
      rrp0   = std::shared_ptr<object>( new mpi_object(scalar) ),
      pap0   = std::shared_ptr<object>( new mpi_object(scalar) ),
      alpha0 = std::shared_ptr<object>( new mpi_object(scalar) ),
      beta0  = std::shared_ptr<object>( new mpi_object(scalar) );

    std::shared_ptr<object> x,r, z,p,q, pnew /* xnew,rnew,rrp have to persist */,
      rr,rrp,pap,alpha,beta;
    x = std::shared_ptr<object>( new mpi_object(blocked,x0) );
    r = std::shared_ptr<object>( new mpi_object(blocked,r0) );
    z = std::shared_ptr<object>( new mpi_object(blocked,z0) );
    p = std::shared_ptr<object>( new mpi_object(blocked,p0) );
    q = std::shared_ptr<object>( new mpi_object(blocked,q0) );
    rr    = std::shared_ptr<object>( new mpi_object(scalar,rr0) );
    pap   = std::shared_ptr<object>( new mpi_object(scalar,pap0) );
    alpha = std::shared_ptr<object>( new mpi_object(scalar,alpha0) );
    beta  = std::shared_ptr<object>( new mpi_object(scalar,beta0) );

    kernel *xorigin,*rorigin,
      *rnorm,*precon,*rho_inprod,*pisz,*matvec,*pap_inprod,*alpha_calc,*beta_calc,
      *xupdate,*rupdate,*pupdate, *xcopy,*rcopy,*pcopy,*rrcopy;

    xorigin = new mpi_origin_kernel( x ); xorigin->set_name("origin x0");
    queue->add_kernel(xorigin);
    rorigin = new mpi_origin_kernel( r ); rorigin->set_name("origin r0");
    queue->add_kernel(rorigin);

    REQUIRE_NOTHROW( xnew0->allocate() );
    xnew = std::shared_ptr<object>( new mpi_object(blocked,xnew0) );
    REQUIRE_NOTHROW( rnew0->allocate() );
    rnew = std::shared_ptr<object>( new mpi_object(blocked,rnew0) );
    REQUIRE_NOTHROW( pnew0->allocate() );
    pnew  = std::shared_ptr<object>( new mpi_object(blocked,pnew0) );

    rnorm = new mpi_norm_kernel( r,rnorms[0] );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    precon = new mpi_preconditioning_kernel( r,z );
    queue->add_kernel(precon);

    rho_inprod = new mpi_innerproduct_kernel( r,z,rr );
    queue->add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

      pisz = new mpi_copy_kernel( z,pnew );
      queue->add_kernel(pisz); pisz->set_name("copy z to p");
  
    REQUIRE_NOTHROW( rrp0->allocate() );
    rrp = std::shared_ptr<object>( new mpi_object(scalar,rrp0) );

    matvec = new mpi_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    pap_inprod = new mpi_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = new mpi_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    REQUIRE_NOTHROW( queue->analyze_kernel_dependencies() );

      xcopy = new mpi_copy_kernel( xnew,x );
      queue->add_kernel(xcopy); xcopy->set_name("copy x");
      rcopy = new mpi_copy_kernel( rnew,r );
      queue->add_kernel(rcopy); rcopy->set_name("copy r");
      pcopy = new mpi_copy_kernel( pnew,p );
      queue->add_kernel(pcopy); pcopy->set_name("copy p");

      xnew = std::shared_ptr<object>( new mpi_object(blocked,xnew0) );
      rnew = std::shared_ptr<object>( new mpi_object(blocked,rnew0) );
      pnew  = std::shared_ptr<object>( new mpi_object(blocked,pnew0) );

    rnorm = new mpi_norm_kernel( r,rnorms[1] );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    precon = new mpi_preconditioning_kernel( r,z );
    queue->add_kernel(precon);

    rho_inprod = new mpi_innerproduct_kernel( r,z,rr );
    queue->add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

      beta_calc = new mpi_scalar_kernel( rr,"/",rrp,beta );
      queue->add_kernel(beta_calc); beta_calc ->set_name("compute beta");

      pupdate = new mpi_axbyz_kernel( '+',one,z, '+',beta,p, pnew );
      queue->add_kernel(pupdate); pupdate   ->set_name("update p");

      rrcopy = new mpi_copy_kernel( rr,rrp );
      queue->add_kernel(rrcopy); rrcopy    ->set_name("save rr value");
  
      rrp = std::shared_ptr<object>( new mpi_object(scalar,rrp0) );

    matvec = new mpi_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    pap_inprod = new mpi_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = new mpi_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = new mpi_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = new mpi_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    queue->analyze_dependencies();
    queue->execute();
  }

}

