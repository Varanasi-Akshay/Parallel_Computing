/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** template_heat.cxx : 
 ****     mode-independent template for heat equation without collectives
 ****
 ****************************************************************/

#include "template_common_header.h"

/*! \page heat Heat equation

  The one-dimensional heat equation is just about the simplest test you can do.
  Since we have no collectives the MPI performance is predictable, and 
  should be fully identical to hand-coded MPI.
  However, in product mode we should get latency hiding.
*/

class heat_environment : public IMP_environment {
protected:
  virtual void print_options() override {
    printf("Heat equation options:\n");
    printf("  -nglobal nnnn : set global problem size\n");
    printf("  -nlocal nnnn : set per processor problem size\n");
    printf("  -steps nnn : set number of iterations\n");
    IMP_environment::print_options();
  };
public:
  heat_environment(int argc,char **argv) : IMP_environment(argc,argv) {
    if (has_argument("help")) print_options(); // this is broken
  };
};

/****
 **** Main program
 ****/

//! \test We have a test for a heat equation without collectives. See \subpage heat.
//! \todo make the data setting mode-independent
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Heat equation options:\n");
    printf("  -nlocal nnn  : set points per processor\n");
    printf("  -nglobal nnn : set global number of points\n");
    printf("  -steps nnn   : set number of iterations\n");
    printf("  -trace       : print norms\n");
    printf("  -reuse       : reuse xvector data\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new heat_environment(argc,argv);
  env->set_name("heat");
  
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  IMP_decomposition* decomp = new IMP_decomposition(arch);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
#endif

  index_int nglobal = env->iargument("nglobal",1000000); int nsteps = env->iargument("steps",20);
  {
    int nl = env->iargument("nlocal",-1);
    if (nl>0) nglobal = nl*arch->nprocs();
  }
  int trace = env->has_argument("trace"), reuse = env->has_argument("reuse");

  distribution
    *blocked = new IMP_block_distribution(decomp,nglobal),
    *scalar = new IMP_replicated_distribution(decomp);
  auto xs = new std::vector<object*>; auto ys = new std::vector<object*>;
  for (int step=0; step<=nsteps; step++) {
    IMP_object *step_object,*out_object;
    if (reuse && step>0) {
      xs->at(0)->allocate(); ys->at(0)->allocate();
      step_object = new IMP_object(blocked,xs->at(0)->get_raw_data());
      out_object = new IMP_object(blocked,ys->at(0)->get_raw_data());
    } else {
      step_object = new IMP_object(blocked);
      out_object = new IMP_object(blocked);
    }
    step_object->set_name(fmt::format("xobject{}",step));
    xs->push_back(step_object);
    out_object->set_name(fmt::format("yobject{}",step));
    ys->push_back(out_object);
  }
  // set initial condition to a delta function
  double *data = xs->at(0)->get_raw_data();
  for (index_int i=0; i<xs->at(0)->local_allocation(); i++)
    data[i] = 0.0;
#if defined(IMPisOMP)
  data[0] = 1.;
#else
  if (mytid==0)
    data[0] = 1.;
#endif

  //snippet heatmainloop
  algorithm *heat;
  heat = new IMP_algorithm(decomp);

  IMP_kernel *initialize = new IMP_origin_kernel(xs->at(0));
  domain_coordinate *deltaloc = new domain_coordinate( std::vector<index_int>{0} );
  initialize->set_localexecutefn
    ( [deltaloc] (int step,processor_coordinate *p,std::vector<object*> *in,object *out,double *flopcount) -> void {
      vecdelta(step,p,in,out,flopcount,*deltaloc); } );
  heat->add_kernel( initialize );
  for (int step=0; step<nsteps; step++) {
    heat->add_kernel( new IMP_diffusion_kernel( xs->at(step),ys->at(step) ) );
    if (trace) {
      object *nrm = new IMP_object(scalar);
      kernel *xnrm = new IMP_norm_kernel( xs->at(step+1),nrm );
      heat->add_kernel( xnrm );
      heat->add_kernel( new IMP_trace_kernel(nrm,fmt::format("Norm at step {}",step+1)) );
    }
    heat->add_kernel( new IMP_copy_kernel( ys->at(step),xs->at(step+1) ) );
  }
  heat->analyze_dependencies();
  heat->execute();
  //snippet end

  delete env;

  return 0;
}
