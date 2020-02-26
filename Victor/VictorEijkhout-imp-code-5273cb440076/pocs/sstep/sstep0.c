#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc,char **argv) {
  MPI_Comm comm;
  MPI_Init(&argc,&argv);
  comm = MPI_COMM_WORLD;
  int nprocs,procno;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procno);

  int side,nsteps;
  if (argc==1)
    side = 100;
  else
    side = atoi(argv[1]);
  MPI_Bcast(&side,1,MPI_INT,0,comm);
  if (argc>2)
    nsteps = atoi(argv[2]);
  else 
    nsteps = 30;
  MPI_Bcast(&nsteps,1,MPI_INT,0,comm);

  double
    *domains[nsteps+1], *betas[nsteps+1]; // beta0 goes unused
  for (int istep=0; istep<=nsteps; istep++) {
    domains[istep] = (double*)malloc(side*sizeof(double));
    for (int i=0; i<side; i++)
      domains[istep][i] = i*i;
    betas[istep] = (double*)malloc((side+2)*sizeof(double));
    for (int i=0; i<side+2; i++)
      betas[istep][i] = 0.;
  }

  int
    left = procno>0 ? procno-1 : MPI_PROC_NULL,
    right = procno<nprocs-1 ? procno+1 : MPI_PROC_NULL;
  MPI_Request requests[4];
  /*
   * Iterate
   */
  if (procno==0)
    printf("Running %d steps with %d local points\n",nsteps,side);
  MPI_Barrier(comm);
  double t = MPI_Wtime();
  for (int istep=0; istep<nsteps; istep++) {
    // initiate transfer
    MPI_Isend(&(domains[istep][0]),1,MPI_DOUBLE, left,0,comm,requests+0);
    MPI_Isend(&(domains[istep][side-1]),1,MPI_DOUBLE, right,0,comm,requests+1);
    MPI_Irecv(&(betas[istep+1][0]),1,MPI_DOUBLE, left,0,comm,requests+2);
    MPI_Irecv(&(betas[istep+1][side+1]),1,MPI_DOUBLE, right,0,comm,requests+3);
    // copy domain into next beta
    for (int i=0; i<side; i++)
      betas[istep+1][i+1] = domains[istep][i];
    // wait for boundary points
    MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);
    // threepoint averaging
    for (int i=0; i<side; i++)
      domains[istep+1][i] =
	2*betas[istep+1][i+1] -betas[istep+1][i] - betas[istep+1][i+2];
  }
  MPI_Barrier(comm);
  t = MPI_Wtime()-t;
  if (procno==0)
    printf("Elapsed time: %6.4f\n",t);

  // prevent optimization, so to speak
  double s=0;
  for (int i=0; i<side; i++)
    s += domains[nsteps][i];
  if (s==10.) printf("%f\n",s);

  MPI_Finalize();
  return 0;
}
