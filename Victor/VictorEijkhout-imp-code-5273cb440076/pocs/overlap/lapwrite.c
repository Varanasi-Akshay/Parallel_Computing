#include <memory.h>

#define QUOTE(name) #name
#define STRING(name) QUOTE(name)
#define SYSTEMSTRING STRING(SYSTEM)

FILE *time_file;
if (procno==0) {
  char lapfile[60];
  memset(lapfile,0,60);
  sprintf(lapfile,"%s/laptime-%s-%d-%d.out",SYSTEMSTRING,LAP,N,THICK);
  time_file = fopen(lapfile,"w");

  if (time_file==NULL) {
    printf("Could not open file: %s\n",lapfile);
    MPI_Abort(comm,-2);
  }
  int msec = (int)(runtime*1000.);
  fprintf(time_file,"%d %d\n",nprocs,msec);
  for (int step=0; step<STEPS-1; step++)
    fprintf(time_file,"%lld %lld %lld %lld %lld %lld %lld %lld\n",
	    post1start_t[step],post1stop_t[step],work1start_t[step],work1stop_t[step],
	    post2start_t[step],post2stop_t[step],work2start_t[step],work2stop_t[step]
	    );
  for (int p=1; p<nprocs; p++) {
    MPI_Recv(post1start_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(post1stop_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(work1start_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(work1stop_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(post2start_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(post2stop_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(work2start_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    MPI_Recv(work2stop_t,STEPS-1,MPI_LONG,p,0,comm,MPI_STATUS_IGNORE);
    fprintf(time_file,"\n");
    for (int step=0; step<STEPS-1; step++) {
      fprintf(time_file,"%lld %lld %lld %lld %lld %lld %lld %lld\n",
	      post1start_t[step],post1stop_t[step],work1start_t[step],work1stop_t[step],
	      post2start_t[step],post2stop_t[step],work2start_t[step],work2stop_t[step]
	      );
    }
  }
  fclose(time_file);
 } else {
  MPI_Send(post1start_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(post1stop_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(work1start_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(work1stop_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(post2start_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(post2stop_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(work2start_t,STEPS-1,MPI_LONG,0,0,comm);
  MPI_Send(work2stop_t,STEPS-1,MPI_LONG,0,0,comm);
 }
