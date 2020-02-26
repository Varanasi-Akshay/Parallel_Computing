#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "utils.h"
#include "threepoint_local.h"

class MPI_Env {
private:
public:
  MPI_Comm comm;
  int mytid,ntids,left,right;
  MPI_Request requests[4];
  MPI_Env(MPI_Comm thecomm) {
    comm = thecomm;
    MPI_Comm_size(comm,&ntids);
    MPI_Comm_rank(comm,&mytid);
    if (mytid==0)
      left = ntids-1;
    else
      left = mytid-1;
    if (mytid==ntids-1)
      right = 0;
    else
      right = mytid+1;
  };
  void print(char *msg) {
    if (mytid==0)
      printf(msg);
    return;
  };
};

void onesidedget( double in_data[],double tmp_data[],int localsize,
		  MPI_Win window,MPI_Env *mpienv,double *nmsgs);
void communicate( double in_data[],double tmp_data[],int localsize,
		  MPI_Env *mpienv,double *nmsgs);

int main (int argc,char **argv) {
  MPI_Env *mpienv;
  char msg[1000];

  MPI_Init(&argc,&argv);
  mpienv = new MPI_Env(MPI_COMM_WORLD);

  if (hasarg_from_argcv("h",argc,argv)) {
    sprintf(msg,"Usage: %s -h -d -s steps -n size\n",argv[0]);
    mpienv->print(msg);
    MPI_Finalize();
    return -1;
  }

  int
    debug = hasarg_from_argcv("d",argc,argv),
    nsteps = iarg_from_argcv("s",1,argc,argv);
  index_int
    globalsize = (index_int)iarg_from_argcv("n",0,argc,argv),
    localsize,first,last,
    side = 2; //atoi(argv[1])
  MPI_Bcast(&debug,1,MPI_INT,0,mpienv->comm);
  MPI_Bcast(&nsteps,1,MPI_INT,0,mpienv->comm);
  MPI_Bcast(&globalsize,1,MPI_INDEX_INT,0,mpienv->comm);
  {
    int P,me;
    MPI_Comm_size(mpienv->comm,&P);
    MPI_Comm_rank(mpienv->comm,&me);
    index_int *proc_data_sizes = new index_int[P];
    if (globalsize>0) {
      psizes_from_global(proc_data_sizes,P,globalsize);
      localsize = proc_data_sizes[me];
    } else {
      localsize = -globalsize;
      for (int p=0; p<P; p++)
	proc_data_sizes[p] = localsize;
    }
    first = 0;
    for (int p=0; p<me; p++)
      first += proc_data_sizes[p];
    last = first+localsize;
  }
  sprintf(msg,"Localsize in bytes: %e\n",(double)(8*localsize));
  mpienv->print(msg);
  double* in_data = new double[localsize];
  double* out_data = new double[localsize];
  double* tmp_data = new double[localsize+2];

  MPI_Win window;
  if (side==1) {
    MPI_Win_create(&in_data,localsize,sizeof(int),MPI_INFO_NULL,
		   mpienv->comm,&window);
  }
  
  double global_nops,nops = 0, global_nmsgs,nmsgs = 0;
  double wtime = MPI_Wtime();
  gen_local(in_data,first,last);
  for (int iter=0; iter<nsteps; iter++) {
    double ops=0, msgs;
    if (side==1) {
      onesidedget(in_data,tmp_data,localsize,window,mpienv,&msgs);
    } else {
      communicate(in_data,tmp_data,localsize,mpienv,&msgs);
      nmsgs += msgs;
    }
    avg_local(tmp_data,out_data, first,last,&ops);
    nops += ops;
  }
  wtime = MPI_Wtime()-wtime;
  MPI_Reduce(&nmsgs,&global_nmsgs,1,MPI_DOUBLE,MPI_SUM,0,mpienv->comm);
  MPI_Reduce(&nops,&global_nops,1,MPI_DOUBLE,MPI_SUM,0,mpienv->comm);
  delete in_data; delete out_data; delete tmp_data;
  if (side==1) {
    MPI_Win_free(&window);
  }
  
  {
    char msg[100];
    sprintf(msg,
	    "Execution time %8.2e for %8.2e ops and %8.2e msgs\nfor Gflop rate %8.2e\n",
	    wtime,global_nops,global_nmsgs,
	    global_nops/wtime*1.e-9);
    mpienv->print(msg);
  }

  MPI_Finalize();

  return 0;
}

void onesidedget( double in_data[],double tmp_data[],int localsize,
		  MPI_Win window,MPI_Env *mpienv,double *msgs) {
  MPI_Win_fence(0,window);
  // get from the left
  MPI_Get(tmp_data+0,1,MPI_INT,
	  mpienv->left,localsize-1,1,MPI_INT,
	  window);
  // get from the right
  MPI_Get(tmp_data+localsize+1,1,MPI_INT,
	  mpienv->right,0,1,MPI_INT,
	  window);
  MPI_Win_fence(0,window);
  *msgs = 2;
  return;
}

void communicate( double in_data[],double tmp_data[],int localsize,
		  MPI_Env *mpienv,double *msgs) {
  MPI_Isend(in_data+0,1,MPI_DOUBLE,
	    mpienv->left,0,mpienv->comm,mpienv->requests+0);
  MPI_Isend(in_data+localsize-1,1,MPI_DOUBLE,
	    mpienv->right,0,mpienv->comm,mpienv->requests+1);
  MPI_Irecv(tmp_data+0,1,MPI_DOUBLE,
	    mpienv->left,0,mpienv->comm,mpienv->requests+2);
  MPI_Irecv(tmp_data+localsize+1,1,MPI_DOUBLE,
	    mpienv->right,0,mpienv->comm,mpienv->requests+3);
  for (int i=0; i<localsize; i++) {
    tmp_data[i+1] = in_data[i];
  }
  MPI_Waitall(4,mpienv->requests,MPI_STATUSES_IGNORE);
  *msgs = 2.;
  return;
}
