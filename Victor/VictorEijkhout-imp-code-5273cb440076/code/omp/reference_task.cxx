#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "utils.h"
#include "threepoint_local.h"

#define Indextwo(i,j,n) ( (i)*n+j )

int main(int argc,char **argv) {
  int magic=201, nthreads;
  double *ashared, *bshared;
  double dtime,ttime;

  if (hasarg_from_argcv("h",argc,argv)) {
    printf("Usage: %s -h -d -s steps -n size\n",argv[0]);
    return -1;
  }

#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
    int t = omp_get_thread_num();
    if (t==0) {
      printf("Running on %d threads\n",nthreads);
    }
  }

  int
    debug = hasarg_from_argcv("d",argc,argv),
    nsteps = iarg_from_argcv("s",1,argc,argv);
  index_int
    globalsize = (index_int)iarg_from_argcv("n",0,argc,argv),
    *localsizes = new index_int[nthreads],
    *offsets = new index_int[nthreads+1];

  psizes_from_global(localsizes,nthreads,globalsize);
  offsets[0] = 0;
  for (int i=0; i<nthreads; i++)
    offsets[i+1] = offsets[i]+localsizes[i];

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

  double total_ops;

  class task;
  task **tasks = new task*[(nsteps+1)*nthreads];
  double *data = new double[(nsteps+1)*globalsize];

  class task {
  private:
    int step,domain,done;
  public:
    task(int s,int d) { step = s; domain = d; done = 0; };
    void execute(task **tasks,double *data,index_int *offsets,int nthreads,int globalsize,
		 double *ops) {
      int status;      
#pragma omp atomic capture
      {
	status = this->done; this->done += 1;
      }
      *ops = 0;
      if (status==0) {
        if (step==0) {
          //printf("gen %d %d\n",step,domain);
	  gen_local( data,offsets[domain],offsets[domain+1] );
	} else {
	  double no_ops;
	  //printf("avg %d %d\n",step,domain);
#pragma omp task 
	  tasks[Indextwo( (step-1), MOD(domain-1,nthreads),nthreads )]->
	    execute(tasks,data,offsets,nthreads,globalsize,&no_ops);
#pragma omp task 
	  tasks[Indextwo( (step-1), MOD(domain  ,nthreads),nthreads )]->
	    execute(tasks,data,offsets,nthreads,globalsize,&no_ops);
#pragma omp task 
	  tasks[Indextwo( (step-1), MOD(domain+1,nthreads),nthreads )]->
	    execute(tasks,data,offsets,nthreads,globalsize,&no_ops);
#pragma omp taskwait
	  avg_local( data+(step-1)*globalsize, data+step*globalsize,
		     offsets[domain],offsets[domain+1],globalsize,ops);
	}
      } // else printf("Task %d:%d was already done\n",step,domain);
    }; // execute
  };

  ttime = omp_get_wtime();

  // generate tasks
  for (int step=0; step<=nsteps; step++)
    for (int dom=0; dom<nthreads; dom++)
      tasks[ Indextwo( step,dom,nthreads) ] = new task(step,dom);

  // execute tasks
double ops;
#pragma omp parallel private(ops)
  {
    for (int step=0; step<=nsteps; step++) {
      for (int dom=0; dom<nthreads; dom++) {
	tasks[ Indextwo( step,dom,nthreads) ]->
	  execute(tasks,data,offsets,nthreads,globalsize,&ops);
	//#pragma omp atomic capture
#pragma omp critical (count_ops)
	{ total_ops += ops; }
      }
    }
  }

  dtime = (omp_get_wtime()-ttime); 

  printf("Elapsed time: %7.3f, Mega-Op count %d, GFlop rate: %7.5f\n",
	 dtime,(int)(1.e-6*total_ops),total_ops*1.e-9/dtime);

  return 0;   
}

