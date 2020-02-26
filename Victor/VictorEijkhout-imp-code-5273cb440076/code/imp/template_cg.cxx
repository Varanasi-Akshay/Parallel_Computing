/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** template_cg.cxx : 
 ****     mode-independent template for conjugate gradients
 ****
 ****************************************************************/

/*!
  \page cg Conjugate Gradients Method

  This is incomplete.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("CG options:\n");
    printf("  -sparse : use sparse matrix rather than stencil\n");
    printf("  -nlocal nnn : set points per processor\n");
    printf("  -steps nnn : set number of iterations\n");
    printf("  -trace : print norms\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("cg");

  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  //  arch->set_can_embed_in_beta(0);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  IMP_decomposition* decomp = new IMP_decomposition(arch);
  
  //  env->set_ir_outputfile("cg");
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int sparse = env->has_argument("sparse");
  int trace = env->has_argument("trace");

  env->print_single
    ( fmt::format("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal) );

  // a bunch of vectors, block distributed
  distribution *blocked = new IMP_block_distribution(decomp,nlocal,-1);
  auto xt    = std::shared_ptr<object>( new IMP_object(blocked) ); xt->set_name(fmt::format("xtrue"));
  xt->allocate(); 
  auto x0    = std::shared_ptr<object>( new IMP_object(blocked) ); x0->set_name(fmt::format("x0"));
  x0->allocate(); 
  auto b0    = std::shared_ptr<object>( new IMP_object(blocked) ); b0->set_name(fmt::format("b0"));
  auto ax0   = std::shared_ptr<object>( new IMP_object(blocked) ); ax0->set_name(fmt::format("ax0"));
  auto r0    = std::shared_ptr<object>( new IMP_object(blocked) ); r0->set_name(fmt::format("r0"));

  // scalars, all redundantly replicated
  distribution *scalar = new IMP_replicated_distribution(decomp);
  auto rnorms = std::vector<std::shared_ptr<object>>(n_iterations);  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = std::shared_ptr<object>( new IMP_object(scalar) ); rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  auto one = std::shared_ptr<object>( new IMP_object(scalar) ); one->set_name("one"); one->set_value(1.);
  auto oNe = std::shared_ptr<object>( new IMP_object(scalar) ); oNe->set_name("oNe"); oNe->set_value(1.);
  
  // Sparse matrix
  IMP_sparse_matrix *sparse_matrix;
  if (sparse) {
    sparse_matrix = new IMP_toeplitz3_matrix(blocked,-1.,2.,-1.);
  }

  // let's define the steps of the loop body
  algorithm *cg = new IMP_algorithm(decomp);
  cg->set_name("Conjugate Gradients Method");

  // initial setup
  { kernel *k = new IMP_origin_kernel(one); k->set_name("origin one");
    cg->add_kernel( k ); }
  { kernel *k = new IMP_origin_kernel(oNe); k->set_name("origin oNe");
    cg->add_kernel( k ); }
  { kernel *xorigin = new IMP_origin_kernel( xt ); xorigin->set_name("origin xtrue");
    xorigin->set_localexecutefn(&vecsetconstantone);
    cg->add_kernel(xorigin); }
  { kernel *xorigin = new IMP_origin_kernel( x0 ); xorigin->set_name("origin x0");
    xorigin->set_localexecutefn(&vecsetconstantzero);
    cg->add_kernel(xorigin); }
  { kernel *borigin;
    if (sparse)
      borigin = new IMP_spmvp_kernel( xt,b0,sparse_matrix );
    else
      borigin = new IMP_centraldifference_kernel( xt,b0 );
    borigin->set_name("b0=A xtrue");
    cg->add_kernel(borigin); }
  if (trace) {
    auto b0norm = std::shared_ptr<object>( new IMP_object(scalar) ); kernel *r0inp = new IMP_norm_kernel( b0,b0norm );
    cg->add_kernel( r0inp );
    cg->add_kernel( new IMP_trace_kernel(b0norm,std::string("A times xtrue")) );
  }
  { kernel *atimesx0;
    if (sparse)
      atimesx0 = new IMP_spmvp_kernel( x0,ax0,sparse_matrix );
    else
      atimesx0 = new IMP_centraldifference_kernel( x0,ax0 );
    atimesx0->set_name("A x0");
    cg->add_kernel(atimesx0); }
  { kernel *rorigin = new IMP_axbyz_kernel( '+',one,ax0, '-',oNe,b0, r0 ); rorigin->set_name("r0=ax0-b");
    cg->add_kernel(rorigin); }
  if (trace) {
    auto rr0 = std::shared_ptr<object>( new IMP_object(scalar) ); kernel *r0inp = new IMP_norm_kernel( r0,rr0 );
    cg->add_kernel( r0inp );
    cg->add_kernel( new IMP_trace_kernel(rr0,std::string("Initial residual norm")) );
  }

  // define objects that need to carry from one iteration to the next
  std::shared_ptr<object> xcarry,rcarry,pcarry,rrp;
  
  for (int it=0; it<n_iterations; it++) {

    // define objects local to the iteration; xcarry,rcarry,rrp have to persist
    auto x = std::shared_ptr<object>( new IMP_object(blocked) ); x->set_name(fmt::format("x{}",it));
    auto r = std::shared_ptr<object>( new IMP_object(blocked) ); r->set_name(fmt::format("r{}",it));
    auto z = std::shared_ptr<object>( new IMP_object(blocked) ); z->set_name(fmt::format("z{}",it));
    auto p = std::shared_ptr<object>( new IMP_object(blocked) ); p->set_name(fmt::format("p{}",it));
    auto q = std::shared_ptr<object>( new IMP_object(blocked) ); q->set_name(fmt::format("q{}",it));
    auto rr    = std::shared_ptr<object>( new IMP_object(scalar) ); rr->set_name(fmt::format("rr{}",it));
    auto pap   = std::shared_ptr<object>( new IMP_object(scalar) ); pap->set_name(fmt::format("pap{}",it));
    auto alpha = std::shared_ptr<object>( new IMP_object(scalar) ); alpha->set_name(fmt::format("alpha{}",it));
    auto beta  = std::shared_ptr<object>( new IMP_object(scalar) ); beta->set_name(fmt::format("beta{}",it));

    if (it==0) {
      kernel *xcopy = new IMP_copy_kernel( x0,x );
      cg->add_kernel(xcopy); xcopy->set_name("start x");
      kernel *rcopy = new IMP_copy_kernel( r0,r );
      cg->add_kernel(rcopy); rcopy->set_name("start r");
    } else {
      kernel *xcopy = new IMP_copy_kernel( xcarry,x );
      cg->add_kernel(xcopy); xcopy->set_name(fmt::format("copy x{}",it));
      kernel *rcopy = new IMP_copy_kernel( rcarry,r );
      cg->add_kernel(rcopy); rcopy->set_name(fmt::format("copy r{}",it));
      kernel *pcopy = new IMP_copy_kernel( pcarry,p );
      cg->add_kernel(pcopy); pcopy->set_name(fmt::format("copy p{}",it));
    }

    xcarry = std::shared_ptr<object>( new IMP_object(blocked) ); xcarry->set_name(fmt::format("xcarry{}",it));
    rcarry = std::shared_ptr<object>( new IMP_object(blocked) ); rcarry->set_name(fmt::format("rcarry{}",it));
    pcarry = std::shared_ptr<object>( new IMP_object(blocked) ); pcarry->set_name(fmt::format("pcarry{}",it));

    //snippet cgtemplate
    kernel *rnorm = new IMP_norm_kernel( r,rnorms[it] );
    cg->add_kernel(rnorm); rnorm->set_name(fmt::format("r norm{}",it));
    if (trace) {
      kernel *trace = new IMP_trace_kernel(rnorms[it],fmt::format("Norm in iteration {}",it));
      cg->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
    }

    kernel *precon = new IMP_preconditioning_kernel( r,z );
    cg->add_kernel(precon); precon->set_name(fmt::format("preconditioning{}",it));

    kernel *rho_inprod = new IMP_innerproduct_kernel( r,z,rr );
    cg->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho{}",it));
    if (trace) {
      kernel *trace = new IMP_trace_kernel(rr,fmt::format("rtz in iteration {}",it));
      cg->add_kernel(trace); trace->set_name(fmt::format("rtz trace {}",it));
    }

    if (it==0) {
      kernel *pisz = new IMP_copy_kernel( z,pcarry );
      cg->add_kernel(pisz); pisz->set_name("copy z to p");
    } else {
      // use rrp object from previous iteration
      kernel *beta_calc = new IMP_scalar_kernel( rr,"/",rrp,beta );
      cg->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta{}",it));

      kernel *pupdate = new IMP_axbyz_kernel( '+',one,z, '+',beta,p, pcarry );
      cg->add_kernel(pupdate); pupdate->set_name(fmt::format("update p{}",it));
    }

    // create new rrp, and immediately copy rr into it
    rrp = std::shared_ptr<object>( new IMP_object(scalar) ); rrp->set_name(fmt::format("rho{}p",it));
    kernel *rrcopy = new IMP_copy_kernel( rr,rrp );
    cg->add_kernel(rrcopy); rrcopy->set_name(fmt::format("save rr value{}",it));

    // matvec for now through 1d central difference
    kernel *matvec;
    if (sparse)
      matvec = new IMP_spmvp_kernel( pcarry,q,sparse_matrix );
    else
      matvec = new IMP_centraldifference_kernel( pcarry,q );
    cg->add_kernel(matvec); matvec->set_name(fmt::format("spmvp{}",it));

    kernel *pap_inprod = new IMP_innerproduct_kernel( pcarry,q,pap );
    cg->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap inner product{}",it));

    kernel *alpha_calc = new IMP_scalar_kernel( rr,"/",pap,alpha );
    cg->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha{}",it));

    kernel *xupdate = new IMP_axbyz_kernel( '+',one,x, '-',alpha,pcarry, xcarry );
    cg->add_kernel(xupdate); xupdate->set_name(fmt::format("update x{}",it));

    kernel *rupdate = new IMP_axbyz_kernel( '+',one,r, '-',alpha,q, rcarry );
    cg->add_kernel(rupdate); rupdate->set_name(fmt::format("update r{}",it));
    //snippet end
  }

  try {
    cg->analyze_dependencies();
    cg->execute();
  } catch (std::string c) { fmt::print("{}\n",c); return -1; }
  
  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    auto data = rnorms[it]->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  //env->print_summary();
  delete env;

  return 0;
}
