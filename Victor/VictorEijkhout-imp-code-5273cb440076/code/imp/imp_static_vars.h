/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Statically defined variables for general use
 ****
 ****************************************************************/

#ifndef STATIC_VARS_H
#define STATIC_VARS_H

#ifdef STATIC_VARS_HERE
#define EXTERN
#else
#define EXTERN extern
#endif

EXTERN int vt_copy_kernel;
EXTERN int message_tag_admin_threshold;

#endif
