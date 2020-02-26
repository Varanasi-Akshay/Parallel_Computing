/****************************************************************
 ****
 **** laptypes.c :
 **** include file for the overlap*.c
 ****
 ****************************************************************/

#ifndef THICK
#define THICK 1
#endif

  MPI_Datatype topsend,botsend,leftsend,rightsend,
    toprecv,botrecv,leftrecv,rightrecv;
  {
    // int insize[2] = {N+2,N+2}, outsize[2] = {N,N};

  /*
   * Sending from the out array
   */
  // top
  {
    // int subsize[2] = {THICK,N}; int start[2] = {0,0};
    /* MPI_Type_create_subarray */
    /*   (2, outsize,subsize,start,   MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&topsend);
  }
  // bot
  {
    // int subsize[2] = {THICK,N}; int start[2] = {N-THICK,0};
    /* MPI_Type_create_subarray */
    /*   (2, outsize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&botsend);
  }
  // left
  {
    // int subsize[2] = {N,THICK}; int start[2] = {0,0};
    /* MPI_Type_create_subarray */
    /*   (2, outsize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&leftsend);
  }
  // right
  {
    // int subsize[2] = {N,THICK}; int start[2] = {0,N-THICK};
    /* MPI_Type_create_subarray */
    /*   (2, outsize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&rightsend);
  }
			   

  /*
   * Receiving to the in array
   */
  // top
  {
    // int subsize[2] = {THICK,N}; int start[2] = {0,1};
    /* MPI_Type_create_subarray */
    /*   (2, insize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&toprecv);
  }
  // bot
  {
    // int subsize[2] = {THICK,N}; int start[2] = {N+2-THICK,1};
    /* MPI_Type_create_subarray */
    /*   (2, insize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&botrecv);
  }
  // left
  {
    // int subsize[2] = {N,THICK}; int start[2] = {1,0};
    /* MPI_Type_create_subarray */
    /*   (2, insize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&leftrecv);
  }
  // right
  {
    // int subsize[2] = {N,THICK}; int start[2] = {1,N+2-THICK};
    /* MPI_Type_create_subarray */
    /*   (2, insize,subsize,start,  MPI_ORDER_C, */
    MPI_Type_contiguous(N*THICK,
       MPI_DOUBLE,&rightrecv);
  }
  }
			   
  MPI_Type_commit(&topsend);
  MPI_Type_commit(&botsend);
  MPI_Type_commit(&leftsend);
  MPI_Type_commit(&rightsend);
  MPI_Type_commit(&toprecv);
  MPI_Type_commit(&botrecv);
  MPI_Type_commit(&leftrecv);
  MPI_Type_commit(&rightrecv);
    
