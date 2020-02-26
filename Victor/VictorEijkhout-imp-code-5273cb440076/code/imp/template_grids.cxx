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
#include "laplace_functions.h"

/*! \page grid Grid updates with overlap

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
    printf("**** if blocked, computations are arranged to allow overlap ****\n");
    printf("****                                                        ****\n");
    printf("****************************************************************\n");
    printf("Sstep equation options:\n");
    printf("  -nlocal nnn  : set points per processor per side\n");
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
    // two-d decomposition
    int laplace_dim = 2;
    processor_coordinate *layout = arch->get_proc_layout(laplace_dim);
    IMP_decomposition *decomp = new IMP_decomposition(arch,layout);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
    processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
#endif

    int
      nlocal = env->iargument("nlocal",100),
      nsteps = env->iargument("steps",10),
      nexperiments = env->iargument("xps",1);
    bool
      blockd = env->has_argument("block"),
      trace = env->has_argument("trace"),
      reuse = env->has_argument("reuse");

    /* Create distributions */
    IMP_distribution *nodes_dist = new IMP_block_distribution
      (decomp,std::vector<index_int>{nlocal*(*layout)[0],nlocal*(*layout)[1]});


    /* Create the objects */
    //    std::vector< std::shared_ptr<IMP_object> > sequence(nsteps);
    std::vector<IMP_object*> sequence(nsteps);
    for (int step=0; step<nsteps; step++) {
      //sequence.at(step) = std::shared_ptr<IMP_object>( new IMP_object(nodes_dist) );
      sequence.at(step) = new IMP_object(nodes_dist);
    }

    /* Create algorithm, and initialize the sequences */
    algorithm *grid_evolve = new IMP_algorithm(decomp);
    //grid_evolve->add_kernel( new IMP_setconstant_kernel( sequence.at(0).get(),1.) );
    grid_evolve->add_kernel( new IMP_setconstant_kernel( sequence.at(0),1.) );
    if (blockd)
      grid_evolve->add_kernel( new IMP_setconstant_kernel( sequence.at(1),1.) );
    //grid_evolve->add_kernel( new IMP_setconstant_kernel( sequence.at(1).get(),1.) );

    /* Define stencil operation */
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

    for (int step=0; step<nsteps/2-1; step++) {
      kernel *k;
      int in1,in2,out1,out2;
      if (blockd) {
	in1 = 2*step; in2 = 2*step+1; out1 = 2*step+2; out2 = 2*step+3;
      } else {
	in1 = 2*step; out1 = 2*step+1; in2 = 2*step+1; out2 = 2*step+2;
      }
      
      //k = new IMP_kernel( sequence.at(in1).get(),sequence.at(out1).get() );
      k = new IMP_kernel( sequence.at(in1),sequence.at(out1) );
      k->add_sigma_stencil(bilinear_stencil);
      k->set_localexecutefn( &laplace_bilinear_fn );
      grid_evolve->add_kernel(k);

      //k = new IMP_kernel( sequence.at(in2).get(),sequence.at(out2).get() );
      k = new IMP_kernel( sequence.at(in2),sequence.at(out2) );
      k->add_sigma_stencil(bilinear_stencil);
      k->set_localexecutefn( &laplace_bilinear_fn );
      grid_evolve->add_kernel(k);
    }

    grid_evolve->analyze_dependencies();
    grid_evolve->optimize();
    grid_evolve->execute();

    delete env;

  } catch (std::string c) {
    fmt::print("Program sstep throw exception <<{}>>\n",c);
  }
  
  return 0;
}
