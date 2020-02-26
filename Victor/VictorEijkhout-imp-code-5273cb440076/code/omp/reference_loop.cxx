#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"

#define THREADS 16

int main(int argc,char **argv) {
  double flopcount=0;
  int magic=201;
  double *ashared, *bshared;
  double dtime,ttime;

  if (hasarg_from_argcv("h",argc,argv)) {
    printf("Usage: %s -h -d -s steps -n size\n",argv[0]);
    return -1;
  }

  int
    debug = hasarg_from_argcv("d",argc,argv),
    nsteps = iarg_from_argcv("s",1,argc,argv),
    globalsize = iarg_from_argcv("n",0,argc,argv);

  ashared = (double*) malloc(globalsize*sizeof(double));
  if (!ashared) {
    printf("Malloc problem\n"); return 2;
  }
  bshared = (double*) malloc(globalsize*sizeof(double));
  if (!bshared) {
    printf("Malloc problem\n"); return 2;
  }

  for (int i=0; i<globalsize; i++) {
    ashared[i] = magic;
    bshared[i] = 0.;
  }

#pragma omp parallel
  {
    int n = omp_get_num_threads();
    int t = omp_get_thread_num();
    if (t==0) {
      printf("Running on %d threads\n",n);
    }
  }

  ttime = omp_get_wtime();

  for (int step=0; step<nsteps; step++) {
    int i;
#pragma omp parallel for private(i)
    for (i=1; i<globalsize-1; i++)
      bshared[i] = ashared[i]+ashared[i+1]+ashared[i-1];
#pragma omp parallel for private(i)
    for (i=0; i<globalsize; i++)
      ashared[i] = bshared[i];
    flopcount += globalsize*2;
  }

  dtime = (omp_get_wtime()-ttime); 

  printf("Elapsed time: %7.3f, Mega-Op count %d, GFlop rate: %7.5f\n",
	 dtime,(int)(1.e-6*flopcount),flopcount*1.e-9/dtime);

  return 0;   
}

