/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** lulesh-2d.cxx : 
 ****     mode-independent template for 2D LULESH proxy-app
 ****
 ****************************************************************/

/*! \page lulesh LULESH

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
  
  // two-d decomposition
  int lulesh_dim = 2;
  processor_coordinate *layout = arch->get_proc_layout(lulesh_dim);
  decomp = new mpi_decomposition(arch,layout);
#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  processor_coordinate *mycoord = decomp->coordinate_from_linear(mytid);
#endif
  
  /*
   * Create distributions
   */
  IMP_distribution *elements_dist, *local_nodes_dist, *global_nodes_dist;

  // number of elements (in each direction) is an input parameter
  index_int local_nelements = env->iargument("elements",100);
  std::vector<index_int> elements_domain;
  for (int id=0; id<lulesh_dim; id++)
    elements_domain.push_back( local_nelements*(layout->coord(id)) );
  elements_dist = new mpi_block_distribution(decomp,elements_domain);

  // each element has 2 local nodes in each direction.
  std::vector<index_int> local_nodes_domain;
  for (int id=0; id<lulesh_dim; id++ )
    local_nodes_domain.push_back( 2*elements_domain[id] );
  local_nodes_dist = new mpi_block_distribution(decomp,local_nodes_domain);

  // 2 nodes per element, except the last which gets 4
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
  elements = new mpi_object(elements_dist); elements->set_name("elements");
  local_nodes = new mpi_object(local_nodes_dist); local_nodes->set_name("local_nodes");
  global_nodes = new mpi_object(global_nodes_dist); global_nodes->set_name("global nodes");

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
  element_to_local_nodes->last_dependency()->set_signature_function_function
		   ( new multi_sigma_operator( 2, &signature_everyd_divby2 ) );
  element_to_local_nodes->set_localexecutefn(&bcast_by2d);
  element_to_local_nodes->set_localexecutectx(&lulesh_dim);

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
