#ifdef TRACE
if (procno==0)
  printf("post 2 for step=%d\n",step);
#endif

#ifdef TIME
post2start_t[step] = tsc();
#endif

#ifdef FAKE
MPI_Isend( outputs2[step]+0*N*THICK,1,topsend, proc_top,0,comm, requests2+0 );
MPI_Isend( outputs2[step]+1*N*THICK,1,botsend, proc_bot,0,comm, requests2+1 );
MPI_Isend( outputs2[step]+2*N*THICK,1,leftsend, proc_left,0,comm, requests2+2 );
MPI_Isend( outputs2[step]+3*N*THICK,1,rightsend, proc_right,0,comm, requests2+3 );

MPI_Irecv( inputs2[step]+0*N*THICK,1,toprecv, proc_top,0,comm, requests2+4 );
MPI_Irecv( inputs2[step]+1*N*THICK,1,botrecv, proc_bot,0,comm, requests2+5 );
MPI_Irecv( inputs2[step]+2*N*THICK,1,leftrecv, proc_left,0,comm, requests2+6 );
MPI_Irecv( inputs2[step]+3*N*THICK,1,rightrecv, proc_right,0,comm, requests2+7 );
#else
MPI_Isend( outputs2[step],1,topsend, proc_top,0,comm, requests2+0 );
MPI_Isend( outputs2[step],1,botsend, proc_bot,0,comm, requests2+1 );
MPI_Isend( outputs2[step],1,leftsend, proc_left,0,comm, requests2+2 );
MPI_Isend( outputs2[step],1,rightsend, proc_right,0,comm, requests2+3 );

MPI_Irecv( inputs2[step],1,toprecv, proc_top,0,comm, requests2+4 );
MPI_Irecv( inputs2[step],1,botrecv, proc_bot,0,comm, requests2+5 );
MPI_Irecv( inputs2[step],1,leftrecv, proc_left,0,comm, requests2+6 );
MPI_Irecv( inputs2[step],1,rightrecv, proc_right,0,comm, requests2+7 );
#endif

#ifdef TIME
post2stop_t[step] = tsc();
#endif
