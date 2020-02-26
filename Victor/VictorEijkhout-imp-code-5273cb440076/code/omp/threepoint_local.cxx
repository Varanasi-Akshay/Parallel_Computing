#include <stdlib.h>
#include <stdio.h>
#include "threepoint_local.h"
#include "../imp/utils.h"

void gen_local(double *outdata,index_int first,index_int last) {

  for (int i=first; i<last; i++)
    outdata[i] = (double)i;

  return;
}

void avg_local(double *indata,double *outdata,
	       index_int first,index_int last,index_int globalsize,double *nops) {

  // shift 0
  for (int i=first; i<last; i++)
    outdata[i] = indata[i];
  // shift to the right
  for (int i=first+1; i<last; i++)
    outdata[i] += indata[MOD(i-1,globalsize)];
  // shift to the left
  for (int i=first; i<last-1; i++)
    outdata[i] += indata[MOD(i+1,globalsize)];

  *nops = 2.*(last-first);
}
