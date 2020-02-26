/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** template_gropp.cxx : 
 ****     mode-independent template for conjugate gradients according to Gropp
 ****
 ****************************************************************/

/*! \page gropp Conjugate Gradients Method by Bill Gropp

  This is based on a presentation by Bill Gropp. However, I think
  his algorithm is wrong, since it contains two matrix vector products
  per iteration.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/

//! \test There is a test for a CG method with overlap of computation/communication. See \subpage gropp.

int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Pipelined CG options:\n");
    printf("  -nlocal nnn : set points per processor\n");
    printf("  -steps nnn : set number of iterations\n");
    printf("  -trace : print norms\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("gropp");
  
  architecture *arch = env->get_architecture();
  //  arch->set_collective_strategy_group();
  IMP_decomposition* decomp = new IMP_decomposition(arch);
  
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int trace = env->has_argument("trace");
  
  // a bunch of vectors, block distributed
  distribution *blocked = new IMP_block_distribution(decomp,nlocal,-1);
  object *xt,*x0,*b0,*r0,*z0,*ax0;
  xt    = new IMP_object(blocked); xt->set_name(fmt::format("xtrue"));
  xt->allocate(); xt->set_value(1.);
  x0    = new IMP_object(blocked); x0->set_name(fmt::format("x0"));
  x0->allocate(); x0->set_value(0.);
  b0    = new IMP_object(blocked); b0->set_name(fmt::format("b0"));
  ax0   = new IMP_object(blocked); ax0->set_name(fmt::format("ax0"));
  r0    = new IMP_object(blocked); r0->set_name(fmt::format("r0"));
  z0    = new IMP_object(blocked); z0->set_name(fmt::format("z0"));

  // scalars, all redundantly replicated
  distribution *scalar = new IMP_replicated_distribution(decomp);
  IMP_object **rnorms = new IMP_object*[n_iterations];  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = new IMP_object(scalar); rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  object *one = new IMP_object(scalar); one->set_name("one"); one->set_value(1.);
  object *oNe = new IMP_object(scalar); oNe->set_name("oNe"); oNe->set_value(1.);
  
  // let's define the steps of the loop body
  algorithm *pipe_cg = new IMP_algorithm(decomp);
  pipe_cg->set_name("Pipelined Conjugate Gradients");
  
  // initial setup
  { kernel *k = new IMP_origin_kernel(one); k->set_name("origin one");
    pipe_cg->add_kernel( k ); }
  { kernel *k = new IMP_origin_kernel(oNe); k->set_name("origin oNe");
    pipe_cg->add_kernel( k ); }
  { kernel *xorigin = new IMP_origin_kernel( xt ); xorigin->set_name("origin xtrue");
    pipe_cg->add_kernel(xorigin); }
  { kernel *xorigin = new IMP_origin_kernel( x0 ); xorigin->set_name("origin x0");
    pipe_cg->add_kernel(xorigin); }
  { kernel *borigin = new IMP_centraldifference_kernel( xt,b0 ); borigin->set_name("b0=A xtrue");
    pipe_cg->add_kernel(borigin); }
  { kernel *atimesx0 = new IMP_centraldifference_kernel( x0,ax0 ); atimesx0->set_name("A x0");
    pipe_cg->add_kernel(atimesx0); }
  { kernel *rorigin = new IMP_axbyz_kernel( '+',one,ax0, '-',oNe,b0, r0 ); rorigin->set_name("r0=ax0-b");
    pipe_cg->add_kernel(rorigin); }
  if (trace) {
    object *rr0 = new IMP_object(scalar); kernel *r0inp = new IMP_norm_kernel( r0,rr0 ); pipe_cg->add_kernel( r0inp );
    pipe_cg->add_kernel( new IMP_trace_kernel(rr0,std::string("Initial residual norm")) );
  }
  { kernel *precr0 = new IMP_preconditioning_kernel( r0,z0 ); precr0->set_name("z0=Mr0");
    pipe_cg->add_kernel(precr0); }

  // define objects that need to carry from one iteration to the next
  object *xcarry,*rcarry,*zcarry, *pcarry,*qcarry, *rrp;
  
  for (int it=0; it<n_iterations; it++) {
    
    object *x,*r,*z,*az,*p,*q,*mq, /* xcarry,rcarry,pcarry, rrp have to persist */
      *rr{nullptr},*rrp{nullptr},*pap{nullptr},*alpha{nullptr},*beta{nullptr};

    x = new IMP_object(blocked); x->set_name(fmt::format("x{}",it));
    r = new IMP_object(blocked); r->set_name(fmt::format("r{}",it));
    z = new IMP_object(blocked); z->set_name(fmt::format("z{}",it));
    az = new IMP_object(blocked); az->set_name(fmt::format("az{}",it));
    p = new IMP_object(blocked); p->set_name(fmt::format("p{}",it));
    q = new IMP_object(blocked); q->set_name(fmt::format("q{}",it));
    mq = new IMP_object(blocked); mq->set_name(fmt::format("mq{}",it));

    rr    = new IMP_object(scalar); rr->set_name(fmt::format("rr{}",it));
    pap   = new IMP_object(scalar); pap->set_name(fmt::format("pap{}",it));
    alpha = new IMP_object(scalar); alpha->set_name(fmt::format("alpha{}",it));
    beta  = new IMP_object(scalar); beta->set_name(fmt::format("beta{}",it));

    if (it==0) {
      { kernel *xcopy = new IMP_copy_kernel( x0,x );
	pipe_cg->add_kernel(xcopy); xcopy->set_name(fmt::format("start x-{}",it)); }
      { kernel *rcopy = new IMP_copy_kernel( r0,r );
	pipe_cg->add_kernel(rcopy); rcopy->set_name(fmt::format("start r-{}",it)); }
      { kernel *zcopy = new IMP_copy_kernel( z0,z );
	pipe_cg->add_kernel(zcopy); zcopy->set_name(fmt::format("start z-{}",it)); }
    } else {
      kernel *xcopy = new IMP_copy_kernel( xcarry,x );
      pipe_cg->add_kernel(xcopy); xcopy->set_name(fmt::format("copy x-{}",it));
      kernel *rcopy = new IMP_copy_kernel( rcarry,r );
      pipe_cg->add_kernel(rcopy); rcopy->set_name(fmt::format("copy r-{}",it));
      kernel *zcopy = new IMP_copy_kernel( zcarry,z );
      pipe_cg->add_kernel(zcopy); zcopy->set_name(fmt::format("copy z-{}",it));
    }

    xcarry = new IMP_object(blocked); xcarry->set_name(fmt::format("x{}p",it));
    rcarry = new IMP_object(blocked); rcarry->set_name(fmt::format("r{}p",it));
    zcarry = new IMP_object(blocked); zcarry->set_name(fmt::format("z{}p",it));

    //snippet gropptemplate
    kernel *rnorm = new IMP_norm_kernel( r,rnorms[it] );
    pipe_cg->add_kernel(rnorm); rnorm->set_name(fmt::format("rnorm-{}",it));
    if (trace) {
      kernel *trace = new IMP_trace_kernel(rnorms[it],fmt::format("Norm in iteration {}",it));
      pipe_cg->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
    }

    rrp = new IMP_object(scalar); rrp->set_name(fmt::format("rho{}",it));

    { kernel *rho_inprod = new IMP_innerproduct_kernel( r,z,rr );
      pipe_cg->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho-{}",it)); }

    { kernel *matvec = new IMP_centraldifference_kernel( z,az );
      pipe_cg->add_kernel(matvec); matvec->set_name(fmt::format("z matvec-{}",it)); }

    if (it==0) { // initialize z<-p, az<-s

      kernel *pisz = new IMP_copy_kernel( z,p );
      pipe_cg->add_kernel(pisz); pisz->set_name(fmt::format("copy z to p-{}",it));
      kernel *qisaz = new IMP_copy_kernel( az,q );
      pipe_cg->add_kernel(qisaz); qisaz->set_name(fmt::format("copy az to q-{}",it));

    } else { // update p,q and copy rr

      kernel *beta_calc = new IMP_scalar_kernel( rr,"/",rrp,beta );
      pipe_cg->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta-{}",it));
      kernel *rrcopy = new IMP_copy_kernel( rr,rrp );
      pipe_cg->add_kernel(rrcopy); rrcopy->set_name(fmt::format("save rr value-{}",it));

      kernel *pupdate = new IMP_axbyz_kernel( '+',one,z, '+',beta,pcarry, p );
      pipe_cg->add_kernel(pupdate); pupdate   ->set_name(fmt::format("update p-{}",it));
      kernel *qupdate = new IMP_axbyz_kernel( '+',one,az, '+',beta,qcarry, q ); // s = Ap?
      pipe_cg->add_kernel(qupdate); qupdate->set_name(fmt::format("update q-{}",it));

    }
  
    { kernel *precon = new IMP_preconditioning_kernel( q,mq );
      pipe_cg->add_kernel(precon); precon->set_name(fmt::format("q precon-{}",it)); }

    { kernel *pap_inprod = new IMP_innerproduct_kernel( p,q,pap );
      pipe_cg->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap innprod-{}",it)); }

    { kernel *alpha_calc = new IMP_scalar_kernel( rr,"/",pap,alpha );
      pipe_cg->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha-{}",it)); }

    { kernel *xupdate = new IMP_axbyz_kernel( '+',one,x, '-',alpha,p, xcarry );
      pipe_cg->add_kernel(xupdate); xupdate->set_name(fmt::format("update x-{}",it)); }

    { kernel *rupdate = new IMP_axbyz_kernel( '+',one,r, '-',alpha,q, rcarry );
      pipe_cg->add_kernel(rupdate); rupdate->set_name(fmt::format("update r-{}",it)); }

    { kernel *zupdate = new IMP_axbyz_kernel( '+',one,z, '-',alpha,mq, zcarry );
      pipe_cg->add_kernel(zupdate); zupdate->set_name(fmt::format("update z-{}",it)); }

    pcarry = new IMP_object(blocked); pcarry->set_name(fmt::format("p{}p",it));
    qcarry = new IMP_object(blocked); qcarry->set_name(fmt::format("s{}p",it));

    { kernel *pcopy = new IMP_copy_kernel( p,pcarry ); // copy in #1, pupdate later
      pipe_cg->add_kernel(pcopy); pcopy->set_name(fmt::format("copy p-{}",it)); }
    { kernel *qcopy = new IMP_copy_kernel( q,qcarry ); // copy in #1, qupdate later
      pipe_cg->add_kernel(qcopy); qcopy->set_name(fmt::format("copy q-{}",it)); }
    //snippet end
  }

  pipe_cg->analyze_dependencies();
  pipe_cg->execute();

  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    double *data = rnorms[it]->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  delete env;

  return 0;
}
