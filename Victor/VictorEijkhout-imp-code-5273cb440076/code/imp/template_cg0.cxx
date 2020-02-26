/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** template_cg.cxx : 
 ****     mode-independent template for conjugate gradients
 ****
 ****************************************************************/

/*! \page cg Conjugate Gradients Method

  This is incomplete.
*/

#include "template_common_header.h"
//#include "cg_kernel.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

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
  
  env->set_ir_outputfile("cg");
  int nlocal = env->iargument("nlocal",100); // points per processor
  int n_iterations = env->iargument("steps",20);
  int trace = env->has_argument("trace");
  env->print_single
    ( fmt::format("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal) );
  fmt::print("CG solve for {} iterations on {} points/proc\n",n_iterations,nlocal);

  // a bunch of vectors, block distributed
  distribution *blocked = new IMP_block_distribution(decomp,nlocal,-1);
  object *x0,*r0;
  x0    = new IMP_object(blocked); x0->set_name(fmt::format("x0"));
  r0    = new IMP_object(blocked); r0->set_name(fmt::format("r0"));

  // scalars, all redundantly replicated
  distribution *scalar = new IMP_replicated_distribution(decomp);
  IMP_object **rnorms = new IMP_object*[n_iterations];  
  for (int it=0; it<n_iterations; it++) {
    rnorms[it] = new IMP_object(scalar); rnorms[it]->set_name(fmt::format("rnorm{}",it));
    rnorms[it]->allocate();
  }

  // the linear system
  //  IMP_sparse_matrix *A = new IMP_toeplitz3_matrix(blocked,-1,2,-1);
  x0->set_value(0.); r0->set_value(1.);

  object *one = new IMP_object(scalar); one->set_name("one"); one->set_value(1.);
  
  // let's define the steps of the loop body
  algorithm *queue = new IMP_algorithm(decomp);
  kernel *k;
  k = new IMP_origin_kernel(one); k->set_name("origin one");
  queue->add_kernel( k );
  kernel *xorigin = new IMP_origin_kernel( x0 ); xorigin->set_name("origin x0");
  queue->add_kernel(xorigin);
  kernel *rorigin = new IMP_origin_kernel( r0 ); rorigin->set_name("origin r0");
  queue->add_kernel(rorigin);

  object **x,**r,**z,**p,**q, **rr,**pap,**alpha,**beta;
  
  x = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    x[it] = new IMP_object(blocked); x[it]->set_name(fmt::format("x{}",it)); }
  r = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    r[it] = new IMP_object(blocked); r[it]->set_name(fmt::format("r{}",it)); }
  z = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    z[it] = new IMP_object(blocked); z[it]->set_name(fmt::format("z{}",it)); }
  p = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    p[it] = new IMP_object(blocked); p[it]->set_name(fmt::format("p{}",it)); }
  q = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    q[it] = new IMP_object(blocked); q[it]->set_name(fmt::format("q{}",it)); }

  rr = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    rr[it] = new IMP_object(scalar); rr[it]->set_name(fmt::format("rr{}",it)); }
  pap = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    pap[it] = new IMP_object(scalar); pap[it]->set_name(fmt::format("pap{}",it)); }
  alpha = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    alpha[it] = new IMP_object(scalar); alpha[it]->set_name(fmt::format("alpha{}",it)); }
  beta = new object*[n_iterations];
  for (int it=0; it<n_iterations; it++) {
    beta[it] = new IMP_object(scalar); beta[it]->set_name(fmt::format("beta{}",it)); }

  try {

    for (int it=0; it<n_iterations; it++) {
      double *data;
      kernel *precon,*rho_inprod, *xupdate,*rupdate;

      //snippet cgtemplate
      if (it==0) {
	precon = new IMP_preconditioning_kernel( r0,z[it] );
	rho_inprod = new IMP_innerproduct_kernel( r0,z[it],rr[it] );
      } else {
	precon = new IMP_preconditioning_kernel( r[it-1],z[it] );
	rho_inprod = new IMP_innerproduct_kernel( r[it-1],z[it],rr[it] );
      }
      queue->add_kernel(precon); precon->set_name(fmt::format("preconditioning{}",it));
      queue->add_kernel(rho_inprod); rho_inprod->set_name(fmt::format("compute rho{}",it));

      if (it==0) {
	kernel *pisz = new IMP_copy_kernel( z[it],p[it] );
	queue->add_kernel(pisz); pisz->set_name("copy z to p");
      } else {
	// use rrp object from previous iteration
	kernel *beta_calc = new IMP_scalar_kernel( rr[it],"/",rr[it-1],beta[it] );
	queue->add_kernel(beta_calc); beta_calc ->set_name(fmt::format("compute beta{}",it));
	kernel *pupdate = new IMP_axbyz_kernel( '+',one,z[it], '+',beta[it],p[it-1], p[it] );
	queue->add_kernel(pupdate); pupdate->set_name(fmt::format("update p{}",it));
      }

      //    kernel *matvec = new IMP_spmvp_kernel( pnew,q,A );
      kernel *matvec = new IMP_centraldifference_kernel( p[it],q[it] );
      queue->add_kernel(matvec); matvec->set_name(fmt::format("spmvp{}",it));

      kernel *pap_inprod = new IMP_innerproduct_kernel( p[it],q[it],pap[it] );
      queue->add_kernel(pap_inprod); pap_inprod->set_name(fmt::format("pap inner product{}",it));

      kernel *alpha_calc = new IMP_scalar_kernel( rr[it],"/",pap[it],alpha[it] );
      queue->add_kernel(alpha_calc); alpha_calc->set_name(fmt::format("compute alpha{}",it));

      if (it==0) {
	xupdate = new IMP_axbyz_kernel( '+',one,x0, '-',alpha[it],p[it], x[it] );
	rupdate = new IMP_axbyz_kernel( '+',one,r0, '-',alpha[it],q[it], r[it] );
      } else {
	xupdate = new IMP_axbyz_kernel( '+',one,x[it-1], '-',alpha[it],p[it], x[it] );
	rupdate = new IMP_axbyz_kernel( '+',one,r[it-1], '-',alpha[it],q[it], r[it] );
      }
      xupdate->set_name(fmt::format("update x{}",it)); queue->add_kernel(xupdate);
      rupdate->set_name(fmt::format("update r{}",it)); queue->add_kernel(rupdate);
      //snippet end

      kernel *rnorm = new IMP_norm_kernel( r[it],rnorms[it] );
      queue->add_kernel(rnorm); rnorm->set_name(fmt::format("r norm{}",it));
      if (trace) {
	kernel *trace = new IMP_trace_kernel(rnorms[it],fmt::format("Norm in iteration {}",it));
	queue->add_kernel(trace); trace->set_name(fmt::format("rnorm trace {}",it));
      }
    }

  } catch (std::string c) { fmt::print("Error <<{}>> during cg construction\n",c);
  } catch (...) { fmt::print("Unknown error during cg iteration\n"); }

  try {
    //  env->kernels_to_dot_file();
    queue->analyze_dependencies();
    queue->execute();
  } catch (std::string c) { fmt::print("Error <<{}>> during cg execution\n",c);
  } catch (...) { fmt::print("Unknown error during cg execution\n"); }


  env->print_single( std::string("Norms:\n") );
  for (int it=0; it<n_iterations; it++) {
    double *data = rnorms[it]->get_raw_data();
    env->print_single(fmt::format("{}:{}, ",it,data[0]));
  }
  env->print_single(std::string("\n"));

  delete env;

  return 0;
}
