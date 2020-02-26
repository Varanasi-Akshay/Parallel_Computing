#ifdef TIME
work1start_t[step] = tsc();
#endif

for (int i=0; i<N; i++) {
  for (int j=0; j<N; j++)
    outputs1[step][INDEXo(i,j,N)] =
      4 * inputs1[step][INDEXi(i,j,N)] -
      ( inputs1[step][INDEXi(i-1,j,N)] + inputs1[step][INDEXi(i+1,j,N)] +
	inputs1[step][INDEXi(i,j-1,N)] + inputs1[step][INDEXi(i,j+1,N)] );
 }

#ifdef TIME
work1stop_t[step] = tsc();
#endif
