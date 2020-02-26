/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** template_cgm.cxx : 
 ****     mode-independent template for conjugate gradients
 ****     with repeated queue invocation
 ****
 ****************************************************************/

#include "template_common_header.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("CG options:\n");
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
  int trace = env->has_argument("trace");

  env->print_single
    ( fmt::format("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal) );

  // a bunch of vectors, block distributed
  distribution *blocked = new IMP_block_distribution(decomp,nlocal,-1);
  object *xt,*x0,*b0,*r0,*ax0;
  xt    = new IMP_object(blocked); xt->set_name(fmt::format("xtrue"));
  xt->allocate(); //xt->set_value(1.);
  x0    = new IMP_object(blocked); x0->set_name(fmt::format("x0"));
  x0->allocate(); //x0->set_value(0.);
  b0    = new IMP_object(blocked); b0->set_name(fmt::format("b0"));
  ax0   = new IMP_object(blocked); ax0->set_name(fmt::format("ax0"));
  r0    = new IMP_object(blocked); r0->set_name(fmt::format("r0"));

  // scalars, all redundantly replicated
  distribution *scalar = new IMP_replicated_distribution(decomp);
  IMP_object **rnorms = new IMP_object*[n_iterations];  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = new IMP_object(scalar); rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  object *one = new IMP_object(scalar); one->set_name("one"); one->set_value(1.);
  object *oNe = new IMP_object(scalar); oNe->set_name("oNe"); oNe->set_value(1.);
  object *x,*r;
  x = new IMP_object(blocked); x->set_name(fmt::format("x"));
  r = new IMP_object(blocked); r->set_name(fmt::format("r"));

  // create initial x & r
  algorithm *cg_setup = new IMP_algorithm(decomp);
  
  try {
    { kernel *k = new IMP_origin_kernel(one,std::string("origin one"));
      cg_setup->add_kernel( k ); }
    { kernel *k = new IMP_origin_kernel(oNe,std::string("origin oNe"));
      cg_setup->add_kernel( k ); }
    { kernel *xorigin = new IMP_origin_kernel( xt,std::string("origin xtrue"));
      xorigin->set_localexecutefn(&vecsetconstantone);
      cg_setup->add_kernel(xorigin); }
    { kernel *xorigin = new IMP_origin_kernel( x ,std::string("origin x"));
      xorigin->set_localexecutefn(&vecsetconstantzero);
      cg_setup->add_kernel(xorigin); }
    { kernel *borigin = new IMP_centraldifference_kernel( xt,b0 ); borigin->set_name("b0=A xtrue");
      cg_setup->add_kernel(borigin); }
    if (trace) {
      object *b0norm = new IMP_object(scalar); kernel *r0inp = new IMP_norm_kernel( b0,b0norm );
      cg_setup->add_kernel( r0inp );
      cg_setup->add_kernel( new IMP_trace_kernel(b0norm,std::string("A times xtrue")) );
    }
    { kernel *atimesx0 = new IMP_centraldifference_kernel( x,ax0 ); atimesx0->set_name("A x0");
      cg_setup->add_kernel(atimesx0); }
    { kernel *rorigin = new IMP_axbyz_kernel( '+',one,ax0, '-',oNe,b0, r0 ); rorigin->set_name("r0=ax0-b");
      cg_setup->add_kernel(rorigin); }
    if (trace) {
      object *rr0 = new IMP_object(scalar); kernel *r0inp = new IMP_norm_kernel( r0,rr0 );
      cg_setup->add_kernel( r0inp );
      cg_setup->add_kernel( new IMP_trace_kernel(rr0,std::string("Initial residual norm")) );
    }
  } catch (std::string c) {
    fmt::print("Error <<{}>> creating cg setup\n",c); throw(-1); }
  
  int it = 0;
  
  object *z,*p,*q, 
    *rr,*pap,*alpha,*beta,
    *xnew,*rnew,*pnew,*rrp,
    *rnorm;

  x = new IMP_object(blocked); x->set_name(fmt::format("x"));
  r = new IMP_object(blocked); r->set_name(fmt::format("r"));
  z = new IMP_object(blocked); z->set_name(fmt::format("z"));
  p = new IMP_object(blocked); p->set_name(fmt::format("p"));
  q = new IMP_object(blocked); q->set_name(fmt::format("q"));
  rr    = new IMP_object(scalar); rr->set_name(fmt::format("rr"));
  pap   = new IMP_object(scalar); pap->set_name(fmt::format("pap"));
  alpha = new IMP_object(scalar); alpha->set_name(fmt::format("alpha"));
  beta  = new IMP_object(scalar); beta->set_name(fmt::format("beta"));

  xnew = new IMP_object(blocked); xnew->set_name(fmt::format("xnew"));
  rnew = new IMP_object(blocked); rnew->set_name(fmt::format("rnew"));
  pnew = new IMP_object(blocked); pnew->set_name(fmt::format("pnew"));
  rrp  = new IMP_object(scalar); rrp->set_name(fmt::format("rrp"));

  rnorm = new IMP_object(scalar); rnorm->set_name(fmt::format("rnorm"));

  algorithm *cg = new IMP_algorithm(decomp);
  try {
    cg->set_name("Conjugate Gradients Method");

    cg->add_kernel( new IMP_origin_kernel(x) );
    cg->add_kernel( new IMP_origin_kernel(r) );
    { kernel *k = new IMP_origin_kernel(one,std::string("origin one"));
      cg->add_kernel( k ); }
    { kernel *k = new IMP_origin_kernel(oNe,std::string("origin oNe"));
      cg->add_kernel( k ); }

    // since we can not test for iteration zero, we have to make p update that always works
    { kernel *initial_p = new IMP_origin_kernel(p);
      initial_p->set_localexecutefn( &vecsetconstantzero );
      cg->add_kernel(initial_p);
    }
    { kernel *initial_rrp = new IMP_origin_kernel(rrp);
      initial_rrp->set_localexecutefn( &vecsetconstantone );
      cg->add_kernel(initial_rrp);
    }

    //snippet cgtemplate
    kernel *normofr = new IMP_norm_kernel( r,rnorm );
    cg->add_kernel(normofr); normofr->set_name(fmt::format("r norm{}",it));
    if (trace) {
      kernel *trace = new IMP_trace_kernel(rnorm,fmt::format("Norm in iteration {}",it));
      cg->add_kernel(trace); trace->set_name(fmt::format("normofr trace {}",it));
    }

    kernel *precon = new IMP_preconditioning_kernel( r,z );
    cg->add_kernel(precon); precon->set_name(fmt::format("preconditioning{}",it));

    kernel *rho_inprod = new IMP_innerproduct_kernel( r,z,rr );
    cg->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho{}",it));
    if (trace) {
      kernel *trace = new IMP_trace_kernel(rr,fmt::format("rtz in iteration {}",it));
      cg->add_kernel(trace); trace->set_name(fmt::format("rtz trace {}",it));
    }

    // if (it==0) {
    //   kernel *pisz = new IMP_copy_kernel( z,pnew );
    //   cg->add_kernel(pisz); pisz->set_name("copy z to p");
    // } else {
    {
      // use rrp object from previous iteration
      kernel *beta_calc = new IMP_scalar_kernel( rr,"/",rrp,beta );
      cg->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta{}",it));

      kernel *pupdate = new IMP_axbyz_kernel( '+',one,z, '+',beta,p, pnew );
      cg->add_kernel(pupdate); pupdate->set_name(fmt::format("update p{}",it));
    }

    // create new rrp, and immediately copy rr into it
    rrp = new IMP_object(scalar); rrp->set_name(fmt::format("rho{}p",it));
    kernel *rrcopy = new IMP_copy_kernel( rr,rrp );
    cg->add_kernel(rrcopy); rrcopy->set_name(fmt::format("save rr value{}",it));

    // matvec for now through 1d central difference
    kernel *matvec = new IMP_centraldifference_kernel( pnew,q );
    cg->add_kernel(matvec); matvec->set_name(fmt::format("spmvp{}",it));

    kernel *pap_inprod = new IMP_innerproduct_kernel( pnew,q,pap );
    cg->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap inner product{}",it));

    kernel *alpha_calc = new IMP_scalar_kernel( rr,"/",pap,alpha );
    cg->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha{}",it));

    kernel *xupdate = new IMP_axbyz_kernel( '+',one,x, '-',alpha,pnew, xnew );
    cg->add_kernel(xupdate); xupdate->set_name(fmt::format("update x{}",it));
    
    kernel *rupdate = new IMP_axbyz_kernel( '+',one,r, '-',alpha,q, rnew );
    cg->add_kernel(rupdate); rupdate->set_name(fmt::format("update r{}",it));
  } catch (std::string c) {
    fmt::print("Error <<{}>> creating cg algorithm\n",c); throw(-1); }

  //snippet end


  algorithm *copy_to_next_iteration = new IMP_algorithm(decomp);
  
  try {
    copy_to_next_iteration->add_kernel( new IMP_origin_kernel(xnew) );
    copy_to_next_iteration->add_kernel( new IMP_origin_kernel(rnew) );
    copy_to_next_iteration->add_kernel( new IMP_origin_kernel(pnew) );
    copy_to_next_iteration->add_kernel( new IMP_origin_kernel(rr) );
    copy_to_next_iteration->add_kernel( new IMP_copy_kernel(xnew,x) );
    copy_to_next_iteration->add_kernel( new IMP_copy_kernel(rnew,r) );
    copy_to_next_iteration->add_kernel( new IMP_copy_kernel(pnew,p) );
    copy_to_next_iteration->add_kernel( new IMP_copy_kernel(rr,rrp) );
  } catch (std::string c) {
    fmt::print("Error <<{}>> creating cg copy algorithm\n",c); throw(-1); }

  try {
    cg_setup->analyze_dependencies();
    cg->analyze_dependencies();
    copy_to_next_iteration->analyze_dependencies();

#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    double walltime = MPI_Wtime();
#endif
    cg_setup->execute();
    double rnorms[n_iterations];
    for (int it=0; it<n_iterations; it++) {
      rnorm->place_data( rnorms+it );
      cg->execute();
      if (it<n_iterations-1) {
	copy_to_next_iteration->execute();
	cg->clear_has_been_executed();
	copy_to_next_iteration->clear_has_been_executed();
      }
    }
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    walltime = MPI_Wtime()-walltime;
    { int procno; MPI_Comm_rank(MPI_COMM_WORLD,&procno);
      if (procno==0) {
	printf("Wall time: %7.3f\n",walltime);
	printf("Norms: ");
	for (int it=0; it<n_iterations; it++)
	  printf("%d: %e",it,rnorms[it]);
	printf("\n");
      }
    }
#endif
  } catch (std::string c) { fmt::print("{}\n",c); return -1; }
  
  // env->print_single( std::string("Norms:\n") );
  // for (int it=0; it<n_iterations; it++) {
  //   double *data = rnorms[it]->get_raw_data();
  //   env->print_single(fmt::format("{}:{}, ",it,data[0]));
  // }
  // env->print_single(std::string("\n"));

  delete env;

  return 0;
}
