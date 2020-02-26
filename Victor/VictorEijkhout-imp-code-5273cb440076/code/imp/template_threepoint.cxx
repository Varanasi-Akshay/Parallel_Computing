/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** template_threepoint.cxx : 
 **** mode-independent template for threepoint averaging
 ****
 ****************************************************************/

/*! \page threepoint Threepoint averaging

  We have included a sample application modeled on the one-dimensional heat equation
  \f[ \frac{\delta}{\delta t}u = -\frac{\delta^2}{\delta x^2} u \f]
  with explicit time-stepping. In the discrete formulation this becomes
  \f[ u(t+1,x) = 2u(t,x)-u(t,x-1)-u(t,x+1) \f]
  Denoting the distribution of the output as \f$ \gamma\f$, 
  we write this in global terms as
  \f[ u(t+1,\gamma) = 2u(t,\gamma)-u(t,\gamma\ll 1)-u(t,\gamma\gg 1) \f]
  Obtaining the left and right shifted inputs requires a single message
  from the left and right processor, assuming a block distribution.

  The code as currently written preserves all timesteps; it is possible
  to make this more efficient.
*/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#if defined(IMPisOMP)
#include <omp.h>
#endif

#if defined(IMPisMPI)
#include <mpi.h>
#endif

#include "IMP_base.h"
#include "IMP_ops.h"
#define MPI_VARS_HERE
#include "IMP_static_vars.h"
#include "threepoint_kernel.h"

int object::count = 0;
int task::count = 0;

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);

  /* Print help information if the user specified "-h" argument */
  if (env->has_argument("h")) {
    printf("Usage: %s [-d] -s nsteps -n size\n",argv[0]);
    return -1;
  }
      
  int
    nsteps = env->iargument("s",1),
    globalsize = env->iargument("n",2*env->get_architecture()->nprocs());
  nsteps = 10; globalsize = 1000; //*env->nrocs();
  printf("Doing threepoint for %d steps on domain size %d\n",nsteps,globalsize);

  /* Define objects to hold all IMP information:
     - array of objects; in this case nsteps+1, generally 2*nsteps
     - array of kernels
     - for OMP a task queue object; for MPI this is a dummy
  */
  object **all_objects = new object*[nsteps+1];
  algorithm* queue = new IMP_algorithm(decomp);

  /* Create the distributions and distribution operators.
     These come from a repertoire that does not depend on this problem.
   */
  distribution
    *blocked = new IMP_block_distribution(env,globalsize);

  /* Declare an initial kernel to create the first object from scratch;
     execution will come later.
  */
  {
    //    env->dprint("Generation step\n");
    int step = 0;
    object *gen_object = new IMP_object( blocked );
    all_objects[step] = gen_object;
    kernel *gen_step = new IMP_kernel(gen_object);
    gen_step->set_localexecutefn( &gen_kernel_execute );
    queue->add_kernel( gen_step );
  }

  /* For each update step create a kernel.
     - The output vector is created as a new vector
     - There are three input (beta) vectors:
     - 1 the previous vector
     - 2 that vector shifted left
     - 3 and shifted right
   */
  for (int step=1; step<=nsteps; ++step) {
    //    env->dprint("Defining step %d\n",step);
    // allocate new vector as output, and store
    object
      *output_vector = new IMP_object( blocked ),
      *input_vector = all_objects[step-1];
    all_objects[step] = output_vector;
    kernel *update_step = 
      new IMP_kernel(input_vector,output_vector);
    update_step->set_localexecutefn( &threepoint_execute );
    update_step->add_sigma_oper( new ioperator(">>1") );
    update_step->add_sigma_oper( new ioperator("<<1") );
    update_step->add_sigma_oper( new ioperator("none") );
    queue->add_kernel( update_step );
  }

  /* Analyze data dependencies.
     - OMP: generate tasks for all kernels
     - MPI: generate VecScatter objects
  */
  queue->analyze_dependencies();

  /* Execute the queue;
     for timing purposes we may do this multiple times
  */
  for (int r=0; r<env->iargument("r",1); r++) {
// #if defined(IMPisOMP)
//     double t = omp_get_wtime();
// #endif
    queue->execute();
// #if defined(IMPisOMP)
//     t = omp_get_wtime()-t;
//     printf("Execute time: %e\n",t);
// #endif
  }

#if 0
  /* Maybe print input and output vector */
  if (env->dodebug() & DEBUG_VECTORS) {
    env->print("Starting point:\n");
    all_objects[0]->print();
    env->print("Final result:\n");
    all_objects[nsteps]->print();
  }

  /* Print statistics */
  env->print_stats();
#endif

  delete queue;
  delete env;

  return 0;
}
