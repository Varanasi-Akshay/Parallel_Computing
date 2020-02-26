/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2016
 ****
 **** template_prk_transpose.cxx : 
 ****     mode-independent template for PRK transpose
 ****
 ****************************************************************/

/*! \page prktranspose PRK Transpose

  https://github.com/ParRes/Kernels/
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("prk_transpose");

  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  IMP_decomposition* decomp = new IMP_decomposition(arch);
  
  IMP_algorithm *queue = new IMP_algorithm(decomp);
  queue->set_name("Matrix transpose");
  
  queue->analyze_dependencies();
  queue->execute();

#if defined(IMPisMPI)
  if (mytid==0) {
    printf("only mytid 0\n");
#endif
    printf("Norms: ");
    for (int it=0; it<n_iterations; it++) {
      double *data = rnorms[it]->get_data(0);
      printf("%8.4e ",data[0]);
    } printf("\n");
      printf("Norms reduction: %e\n",
	     rnorms[n_iterations-2]->get_data(0)[0]
	     /
	     rnorms[0]->get_data(0)[0]
	     );
#if defined(IMPisMPI)
  }
#endif

  delete env;

  return 0;
}
