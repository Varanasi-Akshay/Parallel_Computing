#include <stdio.h>
#include <spawn.h>

int f90_setaffinity_( int *icore){
    cpu_set_t        mask;

    CPU_ZERO(       &mask);
    CPU_SET(*icore, &mask);
    return( sched_setaffinity( (pid_t) 0 , sizeof(mask), &mask ) );
}
int c_setaffinity( int icore){
    cpu_set_t        mask;

    CPU_ZERO(       &mask);
    CPU_SET(icore, &mask);
    return( sched_setaffinity( (pid_t) 0 , sizeof(mask), &mask ) );
}
