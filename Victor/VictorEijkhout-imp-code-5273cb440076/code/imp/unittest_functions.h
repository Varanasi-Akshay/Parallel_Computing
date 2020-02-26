#ifndef UFUNCTIONS_H
#define UFUNCTIONS_H 1

#include "imp_base.h"

void vector_gen(kernel_function_types);
void vecset(kernel_function_types);

void vecshiftleftbump(kernel_function_types);
void vecshiftleftmodulo(kernel_function_types);
void vecshiftrightbump(kernel_function_types);
void vecshiftrightmodulo(kernel_function_types);

void vecscalebytwo(kernel_function_types);
void ksumming(kernel_function_types);
void threepointsummod(kernel_function_types);
void threepointsumbump(kernel_function_types);

int pointfunc33(int i,int my_first);

// testing stuff
void test_globalsize(kernel_function_types,index_int);
void test_nprocs(kernel_function_types,int);
void test_distr_nprocs(kernel_function_types,int);

// nbody
void nb_level_down(kernel_function_types);

#endif
