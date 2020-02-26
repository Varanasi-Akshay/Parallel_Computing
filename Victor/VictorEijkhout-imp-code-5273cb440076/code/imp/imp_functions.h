/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_functions.cxx : header file for imp support functions
 ****
 ****************************************************************/

#ifndef IMP_FUNCTIONS_H
#define IMP_FUNCTIONS_H

//snippet numa123index
#define INDEX1D( i,offsets,nsize ) \
  i-offsets[0]
#define INDEX1Dk( i,offsets,nsize,k )		\
  (k)*(i-offsets[0])
#define INDEX2D( i,j,offsets,nsize ) \
  (i-offsets[0])*nsize[1]+j-offsets[1]
#define INDEX3D( i,j,k,offsets,nsize ) \
  ( (i-offsets[0])*nsize[1]+j-offsets[1] )*nsize[2] + k-offsets[2]
#define COORD1D( i,gsize ) \
  ( i )
#define COORD2D( i,j,gsize ) \
  ( (i)*gsize[1] + j )
#define COORD3D( i,j,k,gsize ) \
  ( ( (i)*gsize[1] + j )*gsize[2] + k )
//snippet end

#include <memory>
#include <vector>
class object;
class processor_coordinate;
class domain_coordinate;

index_int INDEXanyD(domain_coordinate &i,domain_coordinate &off,domain_coordinate &siz,int d);

#include "utils.h"

#define kernel_function_proto void(int,processor_coordinate&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*)
#define kernel_function_args int step,processor_coordinate &p,std::vector<std::shared_ptr<object>> &invectors,std::shared_ptr<object> outvector,double *flopcount
#define kernel_function_types int,processor_coordinate&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*

#define kernel_function_call step,p,invectors,outvector,flopcount

typedef void(*kernel_function)(int,processor_coordinate&,std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,double*);

typedef struct {const char *op; std::shared_ptr<object> obj; } char_object_struct;
typedef struct {double *s1; double *s2; std::shared_ptr<object>obj; } doubledouble_object_struct;
typedef struct {char c1; std::shared_ptr<object> s1;
  char c2; std::shared_ptr<object>s2;
  std::shared_ptr<object> obj; } charcharxyz_object_struct;

void vecnoset( kernel_function_types );
void vecsetlinear( kernel_function_types );
void vecsetlinear2d( kernel_function_types );
void vecdelta( kernel_function_types, domain_coordinate&);
void vecsetconstant( kernel_function_types, double);
void vecsetconstantzero( kernel_function_types );
void vecsetconstantone( kernel_function_types );
void vecsetconstantp( kernel_function_types );
void veccopy( kernel_function_types );
void crudecopy( kernel_function_types );

void vecscaleby( kernel_function_types );
void vecscalebytwo( kernel_function_types );
void vecscalebyc( kernel_function_types,double );
void vecscaledownby( kernel_function_types );
void vecscaledownbyc( kernel_function_types,double );

void vectorsum( kernel_function_types );
void vectorroot( kernel_function_types );
void vecaxbyz( kernel_function_types,void* );
void summing( kernel_function_types );
void rootofsumming( kernel_function_types );
void local_inner_product( kernel_function_types );
void local_norm( kernel_function_types );
void local_normsquared( kernel_function_types );
void local_sparse_matrix_vector_multiply( kernel_function_types,void* );
void sparse_matrix_multiply(processor_coordinate&,std::shared_ptr<object> invec,std::shared_ptr<object> outvec,double*);

void char_scalar_op( kernel_function_args,void* );

void print_trace_message( kernel_function_types,void * );

// nbody stuff
class indexstruct;
std::shared_ptr<indexstruct> doubleinterval(index_int i);
std::shared_ptr<indexstruct> halfinterval(index_int i);
void scansum( kernel_function_types );
void scansumk( kernel_function_types,int );
void scanexpand( kernel_function_types );

// cg stuff
void central_difference_damp( kernel_function_types,double );
void central_difference( kernel_function_types );
void central_difference_anyd( kernel_function_types );
void local_diffusion( kernel_function_types );

#endif
