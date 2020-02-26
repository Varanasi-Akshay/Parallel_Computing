#ifndef UTILS_H
#define UTILS_H

#define MOD(x,y) ( ( x+y ) % (y) )
#ifndef MAX
#define MAX(a,b) \
  ({ __typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a > _b ? _a : _b; })
#endif
#ifndef MIN
#define MIN(a,b) \
  ({ __typeof__ (a) _a = (a); \
  __typeof__ (b) _b = (b); \
  _a < _b ? _a : _b; })
#endif
#ifndef ABS
#define ABS(a) \
  ({ __typeof__ (a) _a = (a); \
  _a < 0 ? -_a : _a; })
#endif

typedef long long int index_int; // VLE we should test somewhere.....
#define MPI_INDEX_INT MPI_LONG_LONG_INT

#define detect(t,m) { if (t) throw(m); }

bool hasarg_from_argcv(const char *name,int nargs,char **the_args);
int iarg_from_argcv(const char *name,int vdef,int nargs,char **the_args);
const char* sarg_from_argcv(const char *name,const char *vdef,int nargs,char **the_args);

#endif
