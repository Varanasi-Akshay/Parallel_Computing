/****************************************************************
 ****
 **** Overlap 4h :
 **** MPI code for stencil update of two grids
 **** using Isend/Irecv, overlapping all grids
 **** using threading to hide the packing overhead
 **** one pure MPI thread, synchronize with barriers
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifndef SYSTEM
#define SYSTEM none
#endif
#if SYSTEM==hikari
#include <string.h>
#endif

#include <mpi.h>
#include <omp.h>

#define LAP "4x"

#ifndef N
#define N 5000
#endif

#define STEPS 10
#include "lapdefs.c"

int main() {

  {
    int supported, requested=MPI_THREAD_MULTIPLE;
    MPI_Init_thread(0,0,requested,&supported);
    if (supported<requested) {
      printf("Not enough threading support: %d s/b %d\n",
	     supported,requested);
      return 1;
    }
  }
  MPI_Comm comm = MPI_COMM_WORLD;
  int nprocs,procno, side,proci,procj;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procno);
  side = pow(1.1*nprocs,0.5);
  if (side*side<nprocs) {
    printf("Can not square %d procs\n",nprocs);
    MPI_Finalize();
    return 1;
  }
  proci = procno / side;
  procj = procno % side;

#include "laptypes.c"
			   
#include "lapalloc.c"

  int
    proc_top = PROC( proci-1,procj,side ), proc_bot = PROC( proci+1,procj,side ),
    proc_left = PROC( proci,procj-1,side ), proc_right = PROC( proci,procj+1,side );
  if (proc_top<0 || proc_top>=nprocs) { printf("top wrong\n"); return 1; }
  if (proc_bot<0 || proc_bot>=nprocs) { printf("bot wrong\n"); return 1; }
  if (proc_left<0 || proc_left>=nprocs) { printf("left wrong\n"); return 1; }
  if (proc_right<0 || proc_right>=nprocs) { printf("right wrong\n"); return 1; }

  MPI_Request requests1[8], requests2[8];
  double runtime, flops = 0.;

  /*
   * Get MPI initialization out of the way
   */
  {
    int step = STEPS-1;
#include "post1.c"
#include "post2.c"
    MPI_Waitall(8,requests1,MPI_STATUSES_IGNORE);
    MPI_Waitall(8,requests2,MPI_STATUSES_IGNORE);
  }

  MPI_Barrier(comm);
  runtime = MPI_Wtime();

#pragma omp parallel 
  {
    int tnum = omp_get_thread_num();
    if (tnum==0) {
      int step=0;
      // post isend/irecv for sequence 1
#include "post1.c"
    }
    // make sure that 1 is posted
#pragma omp barrier
  
    /*
     * Big time stepping loop
     */
    for (int step=0; step<STEPS-1; step++) {

      if (tnum==0) {
	// post isend/irecv for sequence 2
#include "post2.c"
      }
      if (tnum==1) {
	// wait for 1 and compute a new output
	MPI_Waitall(8,requests1,MPI_STATUSES_IGNORE);
#include "update1.c"
	flops += 4.*N*N;
      }

      // barrier wait for results of 1 so that we can post
#pragma omp barrier

      if (tnum==0) {
	// post 1 for the next iteration if there is one
        if (step<STEPS-2) {
#include "post1p.c"
        }
      }
      if (tnum==1) {
          // wait for 2 and compute a new output
          MPI_Waitall(8,requests2,MPI_STATUSES_IGNORE);
#include "update2.c"
	flops += 4.*N*N;
      }
    // wait for results of 2 so that we can post
#pragma omp barrier
    } // end of for loop
  } // end of parallel region

  MPI_Barrier(comm);
  runtime = MPI_Wtime()-runtime;
  if (procno==0)
    printf("overlap4x runtime %e for %e flops\n",runtime,flops);

  if (procno==0)
    printf("check %e\n",outputs1[STEPS-1][INDEXo(0,0,N)]);

#ifdef TIME
#include "lapwrite.c"
#endif

  MPI_Type_free(&topsend);
  MPI_Type_free(&botsend);
  MPI_Type_free(&leftsend);
  MPI_Type_free(&rightsend);
  MPI_Type_free(&toprecv);
  MPI_Type_free(&botrecv);
  MPI_Type_free(&leftrecv);
  MPI_Type_free(&rightrecv);

  for (int step=0; step<STEPS; step++) {
    free(inputs1[step]); free(outputs1[step]);
    free(inputs2[step]); free(outputs2[step]);
  }

  MPI_Finalize();

  return 0;
}
