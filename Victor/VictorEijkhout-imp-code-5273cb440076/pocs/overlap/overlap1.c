/****************************************************************
 ****
 **** Overlap 1 :
 **** MPI code for stencil update,
 **** all actual MPI removed to measure only scalar time
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

#define LAP "1"

#ifndef N
#define N 5000
#endif

#define STEPS 10
#include "lapdefs.c"

int main() {

  MPI_Init(0,0);
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

  double runtime,flops=0.;
  MPI_Barrier(comm);
  runtime = MPI_Wtime();

  for (int step=0; step<STEPS-1; step++) {

    //if (procno==0) printf("Iteration %d\n",step);

    // compute a new output
#include "update1.c"
    flops += 4.*N*N;

#include "update2.c"
    flops += 4.*N*N;

  }

  MPI_Barrier(comm);
  runtime = MPI_Wtime()-runtime;
  if (procno==0)
    printf("overlap1 runtime %e for %e\n",runtime,flops);

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

  MPI_Finalize();

  return 0;
}
