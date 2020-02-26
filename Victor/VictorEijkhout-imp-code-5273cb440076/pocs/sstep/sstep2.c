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
  int domainsize[nsteps+1], betasize[nsteps+1];
  domainsize[0] = betasize[0] = side;
  for (int istep=1; istep<=nsteps; istep++) {
    domainsize[istep] = side + 2*(nsteps-istep);
    betasize[istep] = side + 2*(nsteps-istep + 1);
  }
  for (int istep=0; istep<=nsteps; istep++) {
    domains[istep] = (double*)malloc( domainsize[istep] *sizeof(double));
    for (int i=0; i<domainsize[istep]; i++)
      domains[istep][i] = i*i;
    betas[istep] = (double*)malloc( betasize[istep] *sizeof(double));
    for (int i=0; i<betasize[istep]; i++)
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

  // initiate transfer
  MPI_Isend(&(domains[0][0]),nsteps,MPI_DOUBLE, left,0,comm,requests+0);
  MPI_Isend(&(domains[0][side-nsteps]),nsteps,MPI_DOUBLE, right,0,comm,requests+1);
  MPI_Irecv(&(betas[1][0]),nsteps,MPI_DOUBLE, left,0,comm,requests+2);
  MPI_Irecv(&(betas[1][side+nsteps /* ?? */]),nsteps,MPI_DOUBLE, right,0,comm,requests+3);

  // step 0 copy is a special case:
  for (int i=0; i<side; i++)
    betas[1][i+nsteps] = domains[0][i];
  // now iterate
  for (int istep=0; istep<nsteps; istep++) {
    if (istep>0) {
      // copy domain into next beta
      for (int i=nsteps; i<betasize[istep]-2*nsteps; i++)
	betas[istep+1][i] = domains[istep][i];
    }
    // threepoint averaging
    for (int i=nsteps; i<domainsize[istep+1]-2*nsteps; i++)
      domains[istep+1][i] =
	2*betas[istep+1][i+1] -betas[istep+1][i] - betas[istep+1][i+2];
  }

  // wait for boundary points
  MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);

  // fill in the missing bits
  for (int istep=0; istep<nsteps; istep++) {
    if (istep>0) {
      // copy domain into next beta
      for (int i=0; i<nsteps; i++)
	betas[istep+1][i] = domains[istep][i];
      for (int i=betasize[istep+1]-nsteps; i<betasize[istep+1]; i++)
	betas[istep+1][i] = domains[istep][i];
    }
    // threepoint averaging
    for (int i=0; i<nsteps; i++)
      domains[istep+1][i] =
	2*betas[istep+1][i+1] -betas[istep+1][i] - betas[istep+1][i+2];
    for (int i=domainsize[istep+1]-nsteps; i<domainsize[istep+1]; i++)
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
