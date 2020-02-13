#include <math.h>
#include <stdio.h>
#include <omp.h>

#define N 30000000

double gtod_timer(void);    // timer prototype
int c_setaffinity(int);     // affinity prototype

int main() {

  int i, nt=1, niter=0;
  double a[N], error=0.0, sum;

  double time, t0, t1;

#ifdef _OPENMP
#pragma omp parallel private(nt)
{ nt = omp_get_num_threads(); if(nt<1) printf("NO print, OMP warmup.\n"); }
#endif

   for(i = 0; i < N-1; i+=2) {a[i]   = 0.0; a[i+1] = 1.0;}
    
   t0 = gtod_timer();
    
   do {

      for (i = 1; i < N;   i+=2) a[i] = (a[i] + a[i-1]) / 2.0;
      for (i = 0; i < N-1; i+=2) a[i] = (a[i] + a[i+1]) / 2.0;
      
       
      error=0.0; niter++;

      for (i = 0; i < N-1; i++) error = error + fabs(a[i] - a[i+1]);
       
   } while (error >= 1.0);
 
   t1 = gtod_timer();
   time  = t1 - t0;

   printf("%lf\n",time);
}
