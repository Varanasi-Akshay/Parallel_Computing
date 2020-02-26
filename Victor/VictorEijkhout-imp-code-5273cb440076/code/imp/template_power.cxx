/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** General template for power method
 ****
 ****************************************************************/

/*! \page power Power method

  We run a number of iterations of the power method 
  on a diagonal and tridiagonal matrix. This exercises
  the sparse matrix vector product and the collective routines.
*/

#include "template_common_header.h"

/****
 **** Main program
 ****/

//! \test We have a test for a power metohd. See \subpage power.
//! \todo this uses mytid, so is only for MPI. extend.
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("power");
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  int mytid = arch->mytid();
  IMP_decomposition* decomp = new IMP_decomposition(arch);
  processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
  
  int nlocal = 10000, nsteps = 20;
  //snippet powerobjects
  distribution
    *blocked = new IMP_block_distribution(decomp,nlocal,-1),
    *scalar = new IMP_replicated_distribution(decomp);

  // create vectors, sharing storage
  object **xs = new object*[2*nsteps];
  double *data0,*data1;
  xs[0] = new IMP_object(blocked);
  xs[1] = new IMP_object(blocked);

  // data0 = xs[0]->get_data(mytid);
  // data1 = xs[1]->get_data(mytid);
  for (int i=0; i<nlocal; i++) {
    data0[i] = 1.;
  }
  for (int step=1; step<nsteps; step++) {
    xs[2*step] = new IMP_object(blocked); //,data0);
    xs[2*step+1] = new IMP_object(blocked); //,data1);
  }

  // create lambda values
  object **lambdas = new object*[nsteps];
  double **lambdavalue = new double*[nsteps];
  for (int step=0; step<nsteps; step++) {
    lambdas[step] = new IMP_object(scalar);
    lambdas[step]->allocate();
    lambdavalue[step] = lambdas[step]->get_raw_data().get();
  }
  //snippet end
  
  algorithm *queue;
  IMP_sparse_matrix *A; int test;
  index_int
    my_first = blocked->first_index_r(mycoord)[0],
    my_last = blocked->last_index_r(mycoord)[0];

  for (test=1; test<=2; test++) {

    // need to recreate the queue and matrix for each test
    A = new IMP_sparse_matrix( blocked );
    queue = new IMP_algorithm(decomp);

    if (test==1) { // diagonal matrix
      for (index_int row=my_first; row<=my_last; row++) {
	A->add_element( row,row,2.0 );
      }
    } else if (test==2) { // threepoint matrix
      index_int globalsize = blocked->global_volume();
      for (int row=my_first; row<=my_last; row++) {
	int col;
	col = row;     A->add_element(row,col,2.);
	col = row+1; if (col<globalsize)
		       A->add_element(row,col,-1.);
	col = row-1; if (col>=0)
		       A->add_element(row,col,-1.);
      }
    }

    //snippet powerqueue
    queue->add_kernel( new IMP_origin_kernel(xs[0]) );
    for (int step=0; step<nsteps; step++) {  
      kernel *matvec, *scaletonext,*getlambda;
      // matrix-vector product
      matvec = new IMP_spmvp_kernel( xs[2*step],xs[2*step+1],A );
      queue->add_kernel(matvec);
      // inner product with previous vector
      getlambda = new IMP_innerproduct_kernel( xs[2*step],xs[2*step+1],lambdas[step] );
      queue->add_kernel(getlambda);
      if (step<nsteps-1) {
	// scale down for the next iteration
        scaletonext = new IMP_scaledown_kernel( lambdavalue[step],xs[2*step+1],xs[2*step+2] );
        queue->add_kernel(scaletonext);
      }
    }
    //snippet end

    queue->analyze_dependencies();

    queue->execute();

    // printf("Lambda values (version %d): ",test);
    // for (int step=0; step<nsteps; step++) printf("%e ",lambdavalue[step]);
    // printf("\n");
  }

  delete env;
  return 0;

}

