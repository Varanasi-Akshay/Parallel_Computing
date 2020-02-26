/****************************************************************
 ****
 **** Overlap 0 :
 **** sequential code for stencil update
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N 1000
#define STEPS 50
// indexing in input array with halo
#define INDEXi(i,j,n) (i+1)*(n)+(j)
// indexing in output array
#define INDEXo(i,j,n) (i)*(n)+(j)

int main() {

  double *inputs[STEPS], *outputs[STEPS];
  for (int step=0; step<STEPS; step++) {
    inputs[step] = malloc( (N+2)*N*sizeof(double) );
    outputs[step] = malloc( N*N*sizeof(double) );
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
	inputs[step][INDEXi(i,j,N)] = 1.;
	outputs[step][INDEXo(i,j,N)] = 1.;
      }
    }
  }

  for (int step=0; step<STEPS-1; step++) {
    // compute a new output
    for (int i=0; i<N; i++) {
      int j;
      j = 0;
      outputs[step][INDEXo(i,j,N)] =
	( inputs[step][INDEXi(i-1,j,N)] + inputs[step][INDEXi(i,j,N)] + inputs[step][INDEXi(i+1,j,N)] )/2;
      j = N-1;
      outputs[step][INDEXo(i,j,N)] =
	( inputs[step][INDEXi(i-1,j,N)] + inputs[step][INDEXi(i,j,N)] + inputs[step][INDEXi(i+1,j,N)] )/2;
      for (int j=1; j<N-1; j++)
	outputs[step][INDEXo(i,j,N)] =
	  4 * inputs[step][INDEXi(i,j,N)] -
	  ( inputs[step][INDEXi(i-1,j,N)] + inputs[step][INDEXi(i+1,j,N)] +
	    inputs[step][INDEXi(i,j-1,N)] + inputs[step][INDEXi(i,j+1,N)] );
    }
  }

  printf("check %e\n",outputs[STEPS-1][INDEXo(0,0,N)]);

  return 0;
}
