#ifdef TIME
work2start_t[step] = tsc();
#endif

for (int i=0; i<N; i++) {
  for (int j=0; j<N; j++)
    outputs2[step][INDEXo(i,j,N)] =
      4 * inputs2[step][INDEXi(i,j,N)] -
      ( inputs2[step][INDEXi(i-1,j,N)] + inputs2[step][INDEXi(i+1,j,N)] +
	inputs2[step][INDEXi(i,j-1,N)] + inputs2[step][INDEXi(i,j+1,N)] );
 }

#ifdef TIME
work2stop_t[step] = tsc();
#endif
