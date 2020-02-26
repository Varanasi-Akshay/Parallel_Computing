#ifndef THREEPOINT_LOCAL_H
#define THREEPOINT_LOCAL_H

#include "utils.h"

void gen_local(double *outdata,
	       index_int first,index_int last);
void avg_local(double *indata,double *outdata,
	       index_int first,index_int last,index_int gsize,double *nops);

#endif
