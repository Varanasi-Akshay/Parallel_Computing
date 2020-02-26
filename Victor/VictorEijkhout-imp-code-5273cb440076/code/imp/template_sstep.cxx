/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** template_sstep.cxx : 
 ****     mode-independent template for s-step methods
 ****
 ****************************************************************/

#include "template_common_header.h"
#include "unittest_functions.h"

/*! \page sstep s-Step time evolution

*/

class sstep_environment : public IMP_environment {
protected:
  virtual void print_options() override {
    printf("Sstep equation options:\n");
    printf("  -nglobal nnnn : set global problem size\n");
    printf("  -nlocal nnnn : set per processor problem size\n");
    printf("  -steps nn : set number of iterations\n");
    printf("  -block nn : set iteration blocksize\n");
    IMP_environment::print_options();
  };
public:
  sstep_environment(int argc,char **argv) : IMP_environment(argc,argv) {
    if (has_argument("help")) print_options(); // this is broken
  };
};

/****
 **** Main program
 ****/

//! \test We have a test for a sstep equation without collectives. See \subpage sstep.
//! \todo make the data setting mode-independent
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("****************************************************************\n");
    printf("****                                                        ****\n");
    printf("**** sstep : a blocked central difference code              ****\n");
    printf("****                                                        ****\n");
    printf("**** if blocked, halos are recursively calculated, which    ****\n");
    printf("**** should lead to only one communication per block        ****\n");
    printf("****                                                        ****\n");
    printf("****************************************************************\n");
    printf("Sstep equation options:\n");
    printf("  -nlocal nnn  : set points per processor\n");
    printf("  -nglobal nnn : set global number of points\n");
    printf("  -steps nnn   : set number of iterations\n");
    printf("  -xps n       : set number of experiments\n");
    printf("  -block       : set iteration blocking\n");
    printf("  -trace       : print norms\n");
    printf("  -reuse       : reuse xvector data\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  try {
    IMP_environment *env = new sstep_environment(argc,argv);
    env->set_name("sstep");
  
    IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
    arch->set_can_embed_in_beta(1);

#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    int mytid = arch->mytid();
#endif
    int ntids = arch->nprocs();
    IMP_decomposition* decomp = new IMP_decomposition(arch);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
#endif

    index_int nglobal = env->iargument("nglobal",1000000);
    {
      int nl = env->iargument("nlocal",-1);
      if (nl>0) nglobal = nl*arch->nprocs();
    }
    int
      nsteps = env->iargument("steps",10),
      nexperiments = env->iargument("xps",1);
    int
      blockd = env->has_argument("block"),
      trace = env->has_argument("trace"),
      reuse = env->has_argument("reuse");

    // define a three-point operator
    auto
      noop = std::shared_ptr<ioperator>( new ioperator("none") ),
      left = std::shared_ptr<ioperator>( new ioperator("<=1") ),
      right = std::shared_ptr<ioperator>( new ioperator(">=1") );

    // we use this only to derive the explicit betas
    signature_function *sigma_f = new signature_function();
    sigma_f->add_sigma_oper( noop );
    sigma_f->add_sigma_oper( left);
    sigma_f->add_sigma_oper( right );

    // we start and end with a disjoint block distribution
    distribution
      *block_dist = new IMP_block_distribution(decomp,nglobal);
    auto global_structure = block_dist->get_global_structure();
    // in between we have non-disjoint distributions
    distribution **halo_dists = new distribution*[nsteps];
    for (int istep=nsteps-1; istep>=0; istep--) {
      distribution *out_dist;
      if (istep==nsteps-1)
	out_dist = block_dist;
      else out_dist = halo_dists[istep+1];
      if (blockd>0)
	halo_dists[istep] = new IMP_distribution
	  (sigma_f->derive_beta_structure(out_dist,global_structure));
      else 
	halo_dists[istep] = block_dist;
    }

    object **objects = new object*[nsteps+1];
    objects[0] = new IMP_object(block_dist);
    objects[nsteps] = new IMP_object(block_dist);
    for (int istep=1; istep<nsteps; istep++)
      objects[istep] = new IMP_object(halo_dists[istep]);

    algorithm *sstep = new IMP_algorithm(decomp);
    sstep->add_kernel( new IMP_origin_kernel(objects[0]) );
    kernel **kernels = new kernel*[nsteps];
    for (int istep=0; istep<nsteps; istep++) {
      kernel *k = new IMP_kernel(objects[istep],objects[istep+1]);
      k->set_localexecutefn( &threepointsumbump );
      k->set_explicit_beta_distribution(halo_dists[istep]);
      kernels[istep] = k;
      sstep->add_kernel(k);
    }
    sstep->analyze_dependencies();

    IMP_algorithm *copy_back = new IMP_algorithm(decomp);
    copy_back->add_kernel( new IMP_origin_kernel(objects[nsteps]) );
    copy_back->add_kernel( new IMP_copy_kernel(objects[nsteps],objects[0]) );
    copy_back->analyze_dependencies();

#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    for (int iexperiment=0; iexperiment<nexperiments; iexperiment++) {
      sstep->execute();
      copy_back->execute();
      if (iexperiment<nexperiments-1) {
	sstep->clear_has_been_executed();
	copy_back->clear_has_been_executed();
      }
    }

    delete env;

  } catch (std::string c) {
    fmt::print("Program sstep throw exception <<{}>>\n",c);
  }
  
  return 0;
}
