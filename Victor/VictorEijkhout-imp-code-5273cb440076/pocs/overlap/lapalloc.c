/****************************************************************
 ****
 **** lapalloc.c :
 **** include file for the overlap*.c
 ****
 ****************************************************************/

// initialize
double
  *inputs1[STEPS], *outputs1[STEPS],
  *inputs2[STEPS], *outputs2[STEPS];
#ifdef TIME
unsigned long long 
  post1start_t[STEPS], post1stop_t[STEPS],
  post2start_t[STEPS], post2stop_t[STEPS],
  work1start_t[STEPS], work1stop_t[STEPS], 
  work2start_t[STEPS], work2stop_t[STEPS];
#endif

size_t
  outsize = (size_t)N*N*sizeof(double),
  insize = (size_t)(N+2)*(size_t)(N+2)*sizeof(double);
if (outsize<=0) {
  printf("Invalid outsize %ld\n",outsize); return 1; }
if (insize<=0) {
  printf("Invalid insize %ld\n",insize); return 1; }
if (procno==0)
  printf("outsize: %ld, insize: %ld\n",outsize,insize);

double total_alloc = 0;
for (int step=0; step<STEPS; step++) {
  void *malloc_result;

  total_alloc += insize;
  malloc_result = malloc(insize);
  if (!malloc_result) {
    printf("Failed to malloc 1 @ %d\n",step); return 1; }
  inputs1[step] = malloc_result;
  
  total_alloc += outsize;
  malloc_result = malloc(outsize);
  if (!malloc_result) {
    printf("Failed to malloc 2 @ %d\n",step); return 1; }
  outputs1[step] = malloc_result;
  
  total_alloc += insize;
  malloc_result = malloc(insize);
  if (!malloc_result) {
    printf("Failed to malloc 3 @ %d\n",step); return 1; }
  inputs2[step] = malloc_result;
  
  total_alloc += outsize;
  malloc_result = malloc(outsize);
  if (!malloc_result) {
    printf("Failed to malloc 4 @ %d\n",step); return 1; }
  outputs2[step] = malloc_result;
  
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      outputs1[step][INDEXo(i,j,N)] = 1.;
      outputs2[step][INDEXo(i,j,N)] = 1.;
    }
  }
  for (int i=-1; i<N+1; i++) {
    for (int j=-1; j<N+1; j++) {
      inputs1[step][INDEXi(i,j,N)] = 1.;
      inputs2[step][INDEXi(i,j,N)] = 1.;
    }
  }
 }
if (procno==0)
  printf("Total allocation %f\n",total_alloc);

