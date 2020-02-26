/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** laplace-bilinear.cxx : 
 ****     mode-independent template for 9point stencil
 ****
 ****************************************************************/

/*! \page laplace9 Laplace bilinear

  This is incomplete.
*/

/*
 * Headers and static vars
 */
// static vars need to go before cxx header files
#define STATIC_VARS_HERE
#include "imp_static_vars.h"

#include "IMP_specific_header.h"
// this one defines class-static vars
#include "template_common_header.h"
//#include "cg_kernel.h"
#include "laplace_functions.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("cg");

  architecture *arch = env->get_architecture();
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
#endif
  int ntids = arch->nprocs();
  
  // two-d decomposition
  int laplace_dim = 2;
  processor_coordinate *layout = arch->get_proc_layout(laplace_dim);
  decomp = new mpi_decomposition(arch,layout);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
#endif
  fmt::print("decomposition: {} \n",decomp->as_string());
  
  IMP_distribution *nodes_dist;
  IMP_object *nodes_in,*nodes_out;
  IMP_kernel *bilinear_op;

  // number of elements (in each direction) is an input parameter
  index_int local_nnodes = env->iargument("nodes",100);

  try {
    /* Create distributions */
    nodes_dist = new IMP_block_distribution
      (decomp,std::vector<index_int>{local_nnodes*(*layout)[0],local_nnodes*(*layout)[1]});
      //(decomp,std::vector<index_int>{local_nnodes,local_nnodes},-1); // VLE 

    /* Create the objects */
    nodes_in = new IMP_object(nodes_dist); nodes_in->set_name("nodes in");
    nodes_out = new IMP_object(nodes_dist); nodes_out->set_name("nodes out");

    multi_shift_operator
      *up = new multi_shift_operator(std::vector<index_int>{ 0,+1}),
      *dn = new multi_shift_operator(std::vector<index_int>{ 0,-1}),
      *lt = new multi_shift_operator(std::vector<index_int>{-1, 0}),
      *rt = new multi_shift_operator(std::vector<index_int>{+1, 0}),
      *lu = new multi_shift_operator(std::vector<index_int>{-1,+1}),
      *ru = new multi_shift_operator(std::vector<index_int>{+1,-1}),
      *ld = new multi_shift_operator(std::vector<index_int>{-1,-1}),
      *rd = new multi_shift_operator(std::vector<index_int>{-1,-1});

    bilinear_op = new IMP_kernel(nodes_in,nodes_out);
    bilinear_op->add_sigma_oper(up); bilinear_op->add_sigma_oper(dn);
    bilinear_op->add_sigma_oper(lt); bilinear_op->add_sigma_oper(rt);
    bilinear_op->add_sigma_oper(lu); bilinear_op->add_sigma_oper(ru);
    bilinear_op->add_sigma_oper(ld); bilinear_op->add_sigma_oper(rd);
    bilinear_op->set_localexecutefn( &laplace_bilinear_fn );
    //fmt::print("bilinear has type {}\n",bilinear_op->last_dependency()->type_as_string());

    algorithm *bilinear = new IMP_algorithm(decomp);
    bilinear->add_kernel( new IMP_origin_kernel(nodes_in) );
    bilinear->add_kernel( bilinear_op );
    bilinear->analyze_dependencies();
    bilinear->execute();

  } catch (std::string c) {
    fmt::print("Error <<{}>> running laplace-bilinear\n",c);
    throw(1);
  }


  
#if defined(IMPisMPI)
  if (mytid==0) {
    printf("only mytid 0\n");
#endif
#if defined(IMPisMPI)
  }
#endif

  delete env;

  return 0;
}
