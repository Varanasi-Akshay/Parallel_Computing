// indexing in input array with halo
#define INDEXi(i,j,n) (i+1)*(n+2)+(j+1)
// indexing in output array
#define INDEXo(i,j,n) (i)*(n)+(j)
// linear proc from i,j
#define PROC(i,j,side) ( (i+side)%side )*side + ( (j+side)%side )

static __inline unsigned long long tsc(void){
  unsigned long a, d;
  unsigned long long d2;

  __asm__ __volatile__ ("rdtsc" : "=a" (a), "=d" (d));

  d2 = d;
  return (unsigned long long) a | (d2 << 32);
};
