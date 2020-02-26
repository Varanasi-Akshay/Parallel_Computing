/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** lulesh.cxx : 
 ****     mode-independent template for LULESH proxy-app
 ****
 ****************************************************************/

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
#include "lulesh_functions.h"

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
  
  //env->set_ir_outputfile("lulesh");
  decomp = new mpi_decomposition(arch);

  int lulesh_dim = 1;
  
  // number of elements (in each direction) is an input parameter
  index_int local_nelements = env->iargument("elements",100);
  mpi_distribution *elements_dist;
  elements_dist = new mpi_block_distribution(decomp,local_nelements,-1);

  // each element has 4 local nodes.
  index_int local_nnodes; 
  mpi_distribution *local_nodes_dist; 
  if (lulesh_dim==1) {
    local_nnodes = 4*local_nelements;
    local_nodes_dist = new mpi_block_distribution(decomp,local_nnodes,-1);
  } else throw(0);

  // 2 nodes per element, except the last which gets 4
  mpi_distribution *global_nodes_dist;
  index_int local_gnodes;
  if (lulesh_dim==1) {
    local_gnodes = 2*local_nelements;
    if (mytid==ntids-1)
      local_gnodes += 2;
    global_nodes_dist = new mpi_block_distribution(decomp,local_gnodes,-1);
  } else throw(0);

  /*
   * Create the objects
   */
  mpi_object *elements,*local_nodes,*global_nodes;
  elements = new mpi_object(elements_dist);
  local_nodes = new mpi_object(local_nodes_dist);
  global_nodes = new mpi_object(global_nodes_dist);

  /*
   * Initialize the element values
   */
  mpi_kernel *init_elements,*init_local_nodes,*init_global_nodes;
  init_elements = new mpi_kernel(elements);
  init_elements->set_localexecutefn( &vecsetlinear );

  init_elements->analyze_dependencies();
  init_elements->execute();

  /*
   * element -> local node is a local replication
   */
  mpi_kernel *element_to_local_nodes;
  element_to_local_nodes = new mpi_kernel(elements,local_nodes);
  if (lulesh_dim==1) {
    element_to_local_nodes->set_signature_function_function(&signature_divby4);
    element_to_local_nodes->set_localexecutefn(&bcast_by4);
  } else throw(0);
  element_to_local_nodes->analyze_dependencies();
  element_to_local_nodes->execute();

  /*
   * local node summing to global needs a little trick
   */
  mpi_kernel *local_to_global_nodes;
  local_to_global_nodes = new mpi_kernel(local_nodes,global_nodes);
  index_int g = global_nodes->global_last_index();
  if (lulesh_dim==1) {
    local_to_global_nodes->set_signature_function_function
      ( [g] (index_int i) -> indexstruct* { return signature_n2to4(i,(void*)&g); } );
    local_to_global_nodes->set_localexecutefn(&sum_mod2);
  } else throw(0);
  local_to_global_nodes->analyze_dependencies();
  local_to_global_nodes->execute();

  /*
   * Summing nodes back to elements
   */
  mpi_kernel *nodes_to_elements;
  nodes_to_elements = new mpi_kernel(global_nodes,elements);
  nodes_to_elements->set_signature_function_function(&signature_n2e);
  nodes_to_elements->set_localexecutefn(&sum_mod4);
  nodes_to_elements->analyze_dependencies();
  nodes_to_elements->execute();


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
