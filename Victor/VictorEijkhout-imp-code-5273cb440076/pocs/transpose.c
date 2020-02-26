#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc,char **argv) {

  int nexperiment = 10;
  int patch_size = 2;
  if (argc>=2) {
    patch_size = atoi(argv[1]);
  }

  MPI_Init(0,0);
  MPI_Comm comm = MPI_COMM_WORLD;

  /*
   * Set up a processor grid
   */
  int nprocs,procno;
  MPI_Comm_size(comm,&nprocs);
  MPI_Comm_rank(comm,&procno);
  int nprocs_per_side;
  for (int isqrt=nprocs/2; isqrt>=2; isqrt--) {
    if (nprocs%isqrt==0) {
      nprocs_per_side = nprocs/isqrt;
      if (nprocs_per_side>=isqrt) break;
    }
  }
  if (nprocs_per_side*nprocs_per_side!=nprocs) {
    if (procno==0)
      printf("Number of processors is not a perfect square\n");
    return 1; }
  if (procno==0)
    printf("Processors divided as square with side %d\n",nprocs_per_side);

  int
    procno_i = procno / nprocs_per_side,
    procno_j = procno % nprocs_per_side;
  int other_proc = procno_j * nprocs_per_side + procno_i;

  int nthreads, nthreads_per_side;
#pragma omp parallel
#pragma omp single
  nthreads = omp_get_num_threads();
  if (nthreads==1)
    nthreads_per_side = 1;
  else {
    for (int isqrt=nthreads/2; isqrt>=2; isqrt--) {
      if (nthreads%isqrt==0) {
	nthreads_per_side = nthreads/isqrt;
	if (nthreads_per_side>=isqrt) break;
      }
    }
    if (nthreads_per_side*nthreads_per_side!=nthreads) {
      if (procno==0)
	printf("Number of threads is not a perfect square\n");
      return 1; }
  }
  if (procno==0)
    printf("Threads divided as square with side %d\n",nthreads_per_side);

  /*
   * Create local data
   */
  int
    local_matrix_side = nthreads_per_side * patch_size,
    local_matrix_elements = local_matrix_side * local_matrix_side;
  if (procno==0)
    printf("Local matrix elements: %d\n",local_matrix_elements);
  double
    *local_matrix = (double*) malloc(local_matrix_elements*sizeof(double)),
    *trans_matrix = (double*) malloc(local_matrix_elements*sizeof(double));
  if (local_matrix==NULL) {
    printf("Could not allocate local matrix\n"); return 1; }
  if (trans_matrix==NULL) {
    printf("Could not allocate transpose matrix\n"); return 1; }
#define THREADNO( it,jt ) ( (it)*nthreads_per_side + jt )
  for (int it=0; it<nthreads_per_side; it++)
    for (int jt=0; jt<nthreads_per_side; jt++)
      for (int i=0; i<patch_size; i++)
	for (int j=0; j<patch_size; j++)
	  local_matrix[ (it*nthreads_per_side+i) * nthreads_per_side*patch_size
			+ jt*patch_size + j ] = 1.*THREADNO(it,jt) ;


  double ntransposed=0,ncheck=0, t_elapsed;

  MPI_Barrier(comm);
  t_elapsed = MPI_Wtime();
  for (int iexperiment=0; iexperiment<nexperiment; iexperiment++) {

    /*
     * Do the MPI transposition
     */
    MPI_Sendrecv
      ( local_matrix,local_matrix_elements,MPI_DOUBLE, other_proc,0,
	trans_matrix,local_matrix_elements,MPI_DOUBLE, other_proc,0,
	comm, MPI_STATUS_IGNORE );
    ntransposed += local_matrix_elements;
    MPI_Barrier(comm);

    /* 
     * Do the OpenMP transposition
     */
#if 0
#define INDEX(i,j) (i)*local_matrix_side+(j)
#pragma omp parallel for schedule(guided,8)
    for ( int i=0; i<local_matrix_side; i++ )
      for ( int j=i+1; j<local_matrix_side; j++ )
	trans_matrix[ INDEX(i,j) ] = trans_matrix[ INDEX(j,i) ];
#endif
    // Prevent compiler optimizations
    ncheck += trans_matrix[0];

  }
  MPI_Barrier(comm);
  t_elapsed = MPI_Wtime()-t_elapsed;
  double global_transposed;
  MPI_Reduce(&ntransposed,&global_transposed,1,MPI_DOUBLE,MPI_SUM,0,comm);
  if (procno==0)
    printf("Global number of elements transposed: %e in %e\n",
	   global_transposed,t_elapsed);

#if 0
  if (procno==0) {
    double *global_matrix = (double*)malloc
      ( local_matrix_elements*nprocs*sizeof(double) );
    if (global_matrix==NULL) {
      printf("Could not allocate global matrix\n"); MPI_Abort(comm,0); }
    MPI_Gather
      (local_matrix,local_matrix_elements,MPI_DOUBLE,
       global_matrix,local_matrix_elements,MPI_DOUBLE,
       0,comm);
    printf("Input matrix:\n");
    for (int i=0; i<local_matrix_elements*nprocs; i++)
      printf("%5.3f ",global_matrix[i]);
    printf("\n");
    MPI_Gather
      (trans_matrix,local_matrix_elements,MPI_DOUBLE,
       global_matrix,local_matrix_elements,MPI_DOUBLE,
       0,comm);
    printf("Transposed matrix:\n");
    for (int i=0; i<local_matrix_elements*nprocs; i++)
      printf("%5.3f ",global_matrix[i]);
    printf("\n");
  } else {
    MPI_Gather
      (local_matrix,local_matrix_elements,MPI_DOUBLE,
       NULL,local_matrix_elements,MPI_DOUBLE,
       0,comm);
    MPI_Gather
      (trans_matrix,local_matrix_elements,MPI_DOUBLE,
       NULL,local_matrix_elements,MPI_DOUBLE,
       0,comm);
  }
#endif

  MPI_Finalize();

  return 0;
}
