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
 **** conjugate gradient tests 
 **** (individual kernels are tested in unittest_ops)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_ops.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "orthogonality relations","[cg][kernel][ortho][2]" ) {

  int nlocal = 2, nglobal = nlocal*ntids;
  distribution *blocked;
  REQUIRE_NOTHROW( blocked = new omp_block_distribution(decomp,nglobal) );
  std::shared_ptr<object> x,z,r;
  REQUIRE_NOTHROW( x = std::shared_ptr<object>( new omp_object(blocked) ) );
  REQUIRE_NOTHROW( z = std::shared_ptr<object>( new omp_object(blocked) ) );
  REQUIRE_NOTHROW( r = std::shared_ptr<object>( new omp_object(blocked) ) );

  distribution *scalar = new omp_replicated_distribution(decomp);
  auto 
    rr  = std::shared_ptr<object>( new omp_object(scalar) ),
    zr = std::shared_ptr<object>( new omp_object(scalar) ),
    xr = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) ),
    sbzero = std::shared_ptr<object>( new omp_object(scalar) ),
    one = std::shared_ptr<object>( new omp_object(scalar) );
  double one_value = 1.;
  one->set_value( one_value );
  kernel *makeone = new omp_origin_kernel(one);
  const char *mode;

  omp_sparse_matrix *A; int test;
  REQUIRE_NOTHROW( A = new omp_sparse_matrix( blocked ) );

  index_int globalsize = blocked->global_volume();
  for (int mytid=0; mytid<ntids; mytid++) {
    processor_coordinate mycoord;
    REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
    INFO( "mytid: " << mytid );
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

  algorithm *queue = new omp_algorithm(decomp);
  kernel *maker = new omp_origin_kernel(r),
    *makez = new omp_origin_kernel(z);
  REQUIRE_NOTHROW( queue->add_kernel(makeone) );
  REQUIRE_NOTHROW( queue->add_kernel(maker) );
  REQUIRE_NOTHROW( queue->add_kernel(makez) );

  kernel *rr_inprod, *zr_inprod, *coef_compute, *update,*check;

  SECTION( "regular inner product" ) {
    // z,r independent: r linear, z constant 1
    mode = "regular inner product";
    double *rdata,*zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<globalsize; i++)
      rdata[i] = i+1;
    REQUIRE_NOTHROW( zdata = z->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<globalsize; i++)
      zdata[i] = 1.;

    // rr = r' r
    REQUIRE_NOTHROW( rr_inprod = new omp_normsquared_kernel( r,rr ) );    

    // zr = r' z
    REQUIRE_NOTHROW( zr_inprod = new omp_innerproduct_kernel( r,z,zr ) );
    
    // alpha = (r'z)/(r'r)
    REQUIRE_NOTHROW( coef_compute = new omp_scalar_kernel( zr,"/",rr,alpha ) );
    
    // x = z - alpha r
    REQUIRE_NOTHROW( update = new omp_axbyz_kernel( '+',one,z,'-',alpha,r, x ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    REQUIRE_NOTHROW( check = new omp_innerproduct_kernel( x,r,xr ) );
    
    // REQUIRE_NOTHROW( queue->add_kernel(makeone) );
    // REQUIRE_NOTHROW( queue->add_kernel(maker) );
    // REQUIRE_NOTHROW( queue->add_kernel(makez) );

  }

  SECTION( "A inner product" ) {
    // z = A r
    // rr = r' r
    // beta = r' A r
    // x =

    // z,r independent: r linear, z constant 1
    mode = "A inner product";

    double *rdata,*zdata;
    REQUIRE_NOTHROW( rdata = r->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<globalsize; i++)
      rdata[i] = i+1;
    REQUIRE_NOTHROW( zdata = z->get_data(new processor_coordinate_zero(1)) );
    for (index_int i=0; i<nlocal; i++)
      zdata[i] = 1.;

    // rr = r' r
    REQUIRE_NOTHROW( rr_inprod = new omp_normsquared_kernel( r,rr ) );    

    // zr = r' z
    REQUIRE_NOTHROW( zr_inprod = new omp_innerproduct_kernel( r,z,zr ) );
    
    // alpha = (r'z)/(r'r)
    REQUIRE_NOTHROW( coef_compute = new omp_scalar_kernel( zr,"/",rr,alpha ) );
    
    // x = z - alpha r
    REQUIRE_NOTHROW( update = new omp_axbyz_kernel( '+',one,z,'-',alpha,r, x ) );
    
    // answer = x' r = r' ( z - alpha r ) = rz - zr/rr rr
    REQUIRE_NOTHROW( check = new omp_innerproduct_kernel( x,r,xr ) );
  }

  REQUIRE_NOTHROW( queue->add_kernel(rr_inprod) );
  REQUIRE_NOTHROW( queue->add_kernel(zr_inprod) );
  REQUIRE_NOTHROW( queue->add_kernel(coef_compute) );
  REQUIRE_NOTHROW( queue->add_kernel(update) );
  REQUIRE_NOTHROW( queue->add_kernel(check) );
  
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );      

  if (!strcmp(mode,"regular inner product")) {
    // check a bunch of intermediate results
    double *quad; int g=blocked->global_volume();
    double zr_value = g*(g+1)/2, rr_value = g*(g+1)*(2*g+1)/6,
      alpha_value = zr_value/rr_value;
    CHECK_NOTHROW( quad = zr->get_data(new processor_coordinate_zero(1)) );
    CHECK( quad[0]==Approx( zr_value ) );
    CHECK_NOTHROW( quad = rr->get_data(new processor_coordinate_zero(1)) );
    CHECK( quad[0]==Approx( rr_value ) );
    CHECK_NOTHROW( quad = alpha->get_data(new processor_coordinate_zero(1)) );
    CHECK( quad[0]==Approx( alpha_value ) );
  }

  // => r' x = 0
  INFO( "test: " << mode );
  {
    double *isthiszero;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      CHECK( xr->volume(mycoord)==1 );
    }
    REQUIRE_NOTHROW( isthiszero = xr->get_data(new processor_coordinate_zero(1)) );
    CHECK( isthiszero[0]==Approx(0.) );
  }
}

TEST_CASE( "power method","[3]" ) {

  int nlocal = 10, nglobal = nlocal*ntids, nsteps = 2;
  distribution
    *blocked = new omp_block_distribution(decomp,nglobal),
    *scalar = new omp_replicated_distribution(decomp);

  omp_sparse_matrix *A; int test;
  REQUIRE_NOTHROW( A = new omp_sparse_matrix( blocked ) );

  const char *mat;
  index_int globalsize = blocked->global_volume();
  SECTION( "diagonal matrix" ) {
    test = 1; mat = "diagonal";
    for (index_int row=0; row<globalsize; row++) {
      REQUIRE_NOTHROW( A->add_element( row,row,2.0 ) );
    }
  }
  SECTION( "threepoint matrix" ) {
    test = 2; mat = "threepoint";
    for (int row=0; row<globalsize; row++) {
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
  auto xs = std::vector<std::shared_ptr<object>>(2*nsteps); // VLE why do we allocate too many vectors?
  for (int step=0; step<nsteps; step++) {
    char name_0[7],name_1[7];

    REQUIRE_NOTHROW( xs[2*step] = std::shared_ptr<object>( new omp_object(blocked) ) );
    sprintf(name_0,"xs[%02d]",2*step);
    xs[2*step]->set_name(name_0);

    REQUIRE_NOTHROW( xs[2*step+1] = std::shared_ptr<object>( new omp_object(blocked) ) );
    sprintf(name_1,"xs[%02d]",2*step+1);
    xs[2*step+1]->set_name(name_1);
  }
  double *data0;
  REQUIRE_NOTHROW( data0 = xs[0]->get_data(new processor_coordinate_zero(1)) );
  for (int i=0; i<nlocal; i++) data0[i] = 1.;

  // create lambda values
  auto
    norms = std::vector<std::shared_ptr<object>>(2*nsteps),
    lambdas = std::vector<std::shared_ptr<object>>(nsteps);
  // double
  //   *normvalue = new double[2*nsteps],
  //   *lambdavalue = new double[nsteps];
  for (int step=0; step<nsteps; step++) {
    std::shared_ptr<object> inobj,outobj; fmt::MemoryWriter inname,outname;

    REQUIRE_NOTHROW( inobj = std::shared_ptr<object>( new omp_object(scalar) ) );
    inname.write("in-object-{}",step);
    REQUIRE_NOTHROW( inobj->set_name(inname.str()) );
    norms[2*step] = inobj;

    REQUIRE_NOTHROW( outobj = std::shared_ptr<object>( new omp_object(scalar) ) );
    outname.write("out-object-{}",step);
    REQUIRE_NOTHROW( outobj->set_name(outname.str()) );
    norms[2*step+1] = outobj;

    REQUIRE_NOTHROW( lambdas[step] = std::shared_ptr<object>( new omp_object(scalar) ) );
  }
  
  algorithm *queue = new omp_algorithm(decomp);  
  // originate the first vector
  REQUIRE_NOTHROW( queue->add_kernel( new omp_origin_kernel(xs[0]) ) );
  // loop through the rest
  for (int step=0; step<nsteps; step++) {  
    kernel *matvec, *scaletonext,*getnorm,*computelambda;
    // matrix-vector product
    REQUIRE_NOTHROW( matvec = new omp_spmvp_kernel( xs[2*step],xs[2*step+1],A ) );
    fmt::MemoryWriter w; w.write("mvp-{}.",step);
    REQUIRE_NOTHROW( matvec->set_name(w.str()) );
    REQUIRE_NOTHROW( queue->add_kernel(matvec) );
    // norms to compare
    REQUIRE_NOTHROW( getnorm = new omp_norm_kernel( xs[2*step],norms[2*step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( getnorm = new omp_norm_kernel( xs[2*step+1],norms[2*step+1] ) );
    REQUIRE_NOTHROW( queue->add_kernel(getnorm) );
    REQUIRE_NOTHROW( computelambda =
		     new omp_scalar_kernel( norms[2*step+1],"/",norms[2*step],lambdas[step] ) );
    REQUIRE_NOTHROW( queue->add_kernel(computelambda) );
    if (step<nsteps-1) {
      // scale down for the next iteration
      REQUIRE_NOTHROW(scaletonext =
		      new omp_scaledown_kernel( lambdas[step],xs[2*step+1],xs[2*step+2] ) );
      REQUIRE_NOTHROW( queue->add_kernel(scaletonext) );
    }
  }
  REQUIRE_NOTHROW( queue->analyze_dependencies() );
  REQUIRE_NOTHROW( queue->execute() );

  printf("Lambda values (version %d): ",test);
  for (int step=0; step<nsteps; step++) {
    double *lambdavalue;
    REQUIRE_NOTHROW( lambdavalue = lambdas[step]->get_data(new processor_coordinate_zero(1)) );
    printf("%e ",lambdavalue[0]);
  }
  printf("\n");

  if (test==1) {
    for (int step=0; step<nsteps; step++) {
      INFO( "step: " << step );
      double
	*indata = xs[2*step]->get_data(new processor_coordinate_zero(1)),
	*outdata = xs[2*step+1]->get_data(new processor_coordinate_zero(1));
      // for (int i=0; i<nlocal; i++) {
      // 	INFO( "i=" << i );
      // 	CHECK( outdata[i]==Approx( 2*indata[i] ) );
      // }
      double *n0 = norms[2*step]->get_data(new processor_coordinate_zero(1)), *n1 = norms[2*step+1]->get_data(new processor_coordinate_zero(1)),
	*l = lambdas[step]->get_data(new processor_coordinate_zero(1));
      CHECK( *n1==Approx(2*(*n0)) );
      CHECK( *l!=Approx(0.) );
    }
  }
}

TEST_CASE( "cg algorithm","[4]" ) {

  int nlocal = 10,nglobal = nlocal*ntids;

  // a bunch of vectors, block distributed
  distribution *blocked = new omp_block_distribution(decomp,nglobal);
  auto
    x = std::shared_ptr<object>( new omp_object(blocked) ),
    xnew = std::shared_ptr<object>( new omp_object(blocked) ),
    z = std::shared_ptr<object>( new omp_object(blocked) ),
    r = std::shared_ptr<object>( new omp_object(blocked) ), 
    rnew = std::shared_ptr<object>( new omp_object(blocked) ),
    p = std::shared_ptr<object>( new omp_object(blocked) ), 
    pnew = std::shared_ptr<object>( new omp_object(blocked) ),
    q = std::shared_ptr<object>( new omp_object(blocked) ), 
    qold = std::shared_ptr<object>( new omp_object(blocked) );

  // scalars, all redundantly replicated
  distribution *scalar = new omp_replicated_distribution(decomp);
  auto
    rr  = std::shared_ptr<object>( new omp_object(scalar) ), 
    rrp = std::shared_ptr<object>( new omp_object(scalar) ),
    pap = std::shared_ptr<object>( new omp_object(scalar) ),
    alpha = std::shared_ptr<object>( new omp_object(scalar) ), 
    beta = std::shared_ptr<object>( new omp_object(scalar) );
  int n_iterations=5;
  auto
    rnorms = std::vector<std::shared_ptr<object>>(n_iterations),
    rrzero = std::vector<std::shared_ptr<object>>(n_iterations),
    ppzero = std::vector<std::shared_ptr<object>>(n_iterations);
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = std::shared_ptr<object>( new omp_object(scalar) );
    rrzero[it] = std::shared_ptr<object>( new omp_object(scalar) );
    ppzero[it] = std::shared_ptr<object>( new omp_object(scalar) );
  }

  // the sparse matrix
  omp_sparse_matrix *A;
  { 
    index_int globalsize = blocked->global_volume();

    A = new omp_sparse_matrix(blocked);
    double *xdata = x->get_data(new processor_coordinate_zero(1)), *rdata = r->get_data(new processor_coordinate_zero(1));
    for (int row=0; row<globalsize; row++) {
      xdata[row] = 1.; rdata[row] = 1.;
      int col;
      col = row;     A->add_element(row,col,2.);
      col = row+1; if (col<globalsize)
    		     A->add_element(row,col,-1.);
      col = row-1; if (col>=0)
    		     A->add_element(row,col,-1.);
    }
  }
  
  std::shared_ptr<object> one;
  REQUIRE_NOTHROW( one = std::shared_ptr<object>( new omp_object(scalar) ) );
  { 
    double *one_value;
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord = decomp->coordinate_from_linear(mytid);
      REQUIRE_NOTHROW( one_value = one->get_data(mycoord) );
      REQUIRE_NOTHROW( one_value[0] = 1. );
    }
  }
  
  // let's define the steps of the loop body
  algorithm *queue = new omp_algorithm(decomp);
  queue->add_kernel( new omp_origin_kernel(one) );
  queue->add_kernel( new omp_origin_kernel(x) );
  queue->add_kernel( new omp_origin_kernel(r) );
  queue->add_kernel( new omp_copy_kernel(r,p) );

  SECTION( "one iteration without copy" ) {
    kernel *precon = new omp_copy_kernel( r,z );
    queue->add_kernel(precon);

    kernel *rho_inprod = new omp_innerproduct_kernel( r,z,rr );
    rho_inprod->set_name("compute rho");
    queue->add_kernel(rho_inprod);

    kernel *pisz = new omp_copy_kernel( z,pnew );
    pisz->set_name("copy z to p");
    queue->add_kernel(pisz);

    kernel *matvec = new omp_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    kernel *pap_inprod = new omp_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    kernel *alpha_calc = new omp_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    kernel *xupdate = new omp_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    kernel *rupdate = new omp_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    kernel *rrtest = new omp_innerproduct_kernel( z,rnew,rrzero[0] );
    queue->add_kernel(rrtest) ; rrtest->set_name("test rr orthogonality");

    kernel *xcopy = new omp_copy_kernel( xnew,x );
    queue->add_kernel(xcopy); xcopy     ->set_name("copy x");

    kernel *rcopy = new omp_copy_kernel( rnew,r );
    queue->add_kernel(rcopy); rcopy     ->set_name("copy r");
  
    kernel *rnorm = new omp_norm_kernel( r,rnorms[0] );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    queue->analyze_dependencies();
    queue->execute();

    double *data;

    // z is preconditioned for now just copy of r
    data = z->get_data(new processor_coordinate_zero(1));
    for (index_int i=0; i<nlocal; i++)
      CHECK( data[i]==Approx(1.) );

    // rr = r^tr; with r all ones that comes to Nglobal
    data = rr->get_data(new processor_coordinate_zero(1));
    CHECK( data[0]==Approx(nglobal) );

    // pnew is a copy of z, so again all one
    data = pnew->get_data(new processor_coordinate_zero(1));
    for (index_int i=0; i<nlocal; i++)
      CHECK( data[i]==Approx(1.) );

    // q = Ap, so 1 in the first and last component, zero elsewhere
    data = q->get_data(new processor_coordinate_zero(1));
    for (int mytid=0; mytid<ntids; mytid++) {
      processor_coordinate mycoord;
      REQUIRE_NOTHROW( mycoord = decomp->coordinate_from_linear(mytid) );
      index_int my_first = q->first_index_r(mycoord)[0],my_last = q->last_index_r(mycoord)[0];
      if (mytid==0) {
	CHECK( data[0]==Approx(1.) );
	my_first++;
      } else if (mytid==ntids-1) {
	CHECK( data[my_last]==Approx(1.) );
	my_last--;
      }
      for (int i=my_first; i<=my_last; i++) {
	INFO( "tid: " << mytid << ", i=" << i );
	CHECK( data[i]==Approx(0.) );
      }
    }

    data = pap->get_data(new processor_coordinate_zero(1));
    CHECK( data[0]==Approx(2.) );

    data = alpha->get_data(new processor_coordinate_zero(1));
    CHECK( data[0]==Approx(ntids*nlocal/2.) );

    data = rrzero[0]->get_data(new processor_coordinate_zero(1));
    CHECK( data[0]==Approx(0.) );

  }

  SECTION( "two iterations" ) {

    double *data;

    auto
      x0 = std::shared_ptr<object>( new omp_object(blocked) ),
      r0 = std::shared_ptr<object>( new omp_object(blocked) ),
      z0 = std::shared_ptr<object>( new omp_object(blocked) ),
      p0 = std::shared_ptr<object>( new omp_object(blocked) ),
      q0 = std::shared_ptr<object>( new omp_object(blocked) ),
      xnew0 = std::shared_ptr<object>( new omp_object(blocked) ),
      rnew0 = std::shared_ptr<object>( new omp_object(blocked) ),
      pnew0 = std::shared_ptr<object>( new omp_object(blocked) ),
      rr0    = std::shared_ptr<object>( new omp_object(scalar) ),
      rrp0   = std::shared_ptr<object>( new omp_object(scalar) ),
      pap0   = std::shared_ptr<object>( new omp_object(scalar) ),
      alpha0 = std::shared_ptr<object>( new omp_object(scalar) ),
      beta0  = std::shared_ptr<object>( new omp_object(scalar) );

    std::shared_ptr<object> x,r, z,p,q, pnew /* xnew,rnew,rrp have to persist */,
      rr,rrp,pap,alpha,beta;
    x = std::shared_ptr<object>( new omp_object(blocked,x0) );
    r = std::shared_ptr<object>( new omp_object(blocked,r0) );
    z = std::shared_ptr<object>( new omp_object(blocked,z0) );
    p = std::shared_ptr<object>( new omp_object(blocked,p0) );
    q = std::shared_ptr<object>( new omp_object(blocked,q0) );
    rr    = std::shared_ptr<object>( new omp_object(scalar,rr0) );
    pap   = std::shared_ptr<object>( new omp_object(scalar,pap0) );
    alpha = std::shared_ptr<object>( new omp_object(scalar,alpha0) );
    beta  = std::shared_ptr<object>( new omp_object(scalar,beta0) );

    omp_kernel *xorigin,*rorigin,
      *rnorm,*precon,*rho_inprod,*pisz,*matvec,*pap_inprod,*alpha_calc,*beta_calc,
      *xupdate,*rupdate,*pupdate, *xcopy,*rcopy,*pcopy,*rrcopy;

    xorigin = new omp_origin_kernel( x ); xorigin->set_name("origin x0");
    queue->add_kernel(xorigin);
    rorigin = new omp_origin_kernel( r ); rorigin->set_name("origin r0");
    queue->add_kernel(rorigin);

    REQUIRE_NOTHROW( xnew = std::shared_ptr<object>( new omp_object(blocked,xnew0) ) );
    REQUIRE_NOTHROW( rnew = std::shared_ptr<object>( new omp_object(blocked,rnew0) ) );
    REQUIRE_NOTHROW( pnew  = std::shared_ptr<object>( new omp_object(blocked,pnew0) ) );

    REQUIRE_NOTHROW( rnorm = new omp_norm_kernel( r,rnorms[0] ) );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    precon = new omp_preconditioning_kernel( r,z );
    queue->add_kernel(precon);

    rho_inprod = new omp_innerproduct_kernel( r,z,rr );
    queue->add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

      pisz = new omp_copy_kernel( z,pnew );
      queue->add_kernel(pisz); pisz->set_name("copy z to p");
  
      REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar,rrp0) ) );

    matvec = new omp_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    pap_inprod = new omp_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = new omp_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = new omp_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = new omp_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    xcopy = new omp_copy_kernel( xnew,x );
    queue->add_kernel(xcopy); xcopy->set_name("copy x");
    rcopy = new omp_copy_kernel( rnew,r );
    queue->add_kernel(rcopy); rcopy->set_name("copy r");
    pcopy = new omp_copy_kernel( pnew,p );
    queue->add_kernel(pcopy); pcopy->set_name("copy p");

    REQUIRE_NOTHROW( xnew = std::shared_ptr<object>( new omp_object(blocked,xnew0) ) );
    REQUIRE_NOTHROW( rnew = std::shared_ptr<object>( new omp_object(blocked,rnew0) ) );
    REQUIRE_NOTHROW( pnew  = std::shared_ptr<object>( new omp_object(blocked,pnew0) ) );

    rnorm = new omp_norm_kernel( r,rnorms[1] );
    queue->add_kernel(rnorm); rnorm->set_name("r norm");

    precon = new omp_preconditioning_kernel( r,z );
    queue->add_kernel(precon);

    rho_inprod = new omp_innerproduct_kernel( r,z,rr );
    queue->add_kernel(rho_inprod); rho_inprod->set_name("compute rho");

    beta_calc = new omp_scalar_kernel( rr,"/",rrp,beta );
    queue->add_kernel(beta_calc); beta_calc ->set_name("compute beta");

    pupdate = new omp_axbyz_kernel( '+',one,z, '+',beta,p, pnew );
    queue->add_kernel(pupdate); pupdate   ->set_name("update p");

    rrcopy = new omp_copy_kernel( rr,rrp );
    ;queue->add_kernel(rrcopy); rrcopy    ->set_name("save rr value");
  
    REQUIRE_NOTHROW( rrp = std::shared_ptr<object>( new omp_object(scalar,rrp0) ) );

    matvec = new omp_spmvp_kernel( pnew,q,A );
    queue->add_kernel(matvec);

    pap_inprod = new omp_innerproduct_kernel( pnew,q,pap );
    queue->add_kernel(pap_inprod); pap_inprod->set_name("pap inner product");

    alpha_calc = new omp_scalar_kernel( rr,"/",pap,alpha );
    queue->add_kernel(alpha_calc); alpha_calc->set_name("compute alpha");

    xupdate = new omp_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    queue->add_kernel(xupdate); xupdate->set_name("update x");

    rupdate = new omp_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    queue->add_kernel(rupdate); rupdate->set_name("update r");

    REQUIRE_NOTHROW( queue->analyze_dependencies() );
    REQUIRE_NOTHROW( queue->execute() );
  }

}

