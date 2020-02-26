/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** template_lulesh.cxx : 
 ****     mode-independent template LULESH
 ****
 ****************************************************************/

#include "template_common_header.h"
#include "lulesh_functions.h"

/*! \page lulesh LULESH Solver

*/

class lulesh_environment : public IMP_environment {
protected:
public:
  lulesh_environment(int argc,char **argv) : IMP_environment(argc,argv) {
    if (has_argument("help")) print_options(); // this is broken
  };
};

/****
 **** Main program
 ****/

//! \test We have a test for a lulesh equation without collectives. See \subpage lulesh.
//! \todo make the data setting mode-independent
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Lulesh solver options:\n");
    printf("  -dim n : space dimension\n");
    printf("  -elocal nnn: local number of elements per dimension\n");
    printf("  -trace : print stuff\n");
    printf("\n");
  };

  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new lulesh_environment(argc,argv);
  env->set_name("lulesh");
  int elocal = env->iargument("elocal",100); // element per processor per dimension
  int dim = env->iargument("dim",2);   
  int trace = env->has_argument("trace");

  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  auto layout = arch->get_proc_layout(dim);
  IMP_decomposition *decomp = new IMP_decomposition(arch,layout);

#if defined(IMPisMPI) || defined(IMPisPRODUCT)
  int mytid = arch->mytid();
  auto mycoord = decomp->coordinate_from_linear(mytid);
#endif
  int ntids = arch->nprocs();

  /****
   **** Make objects:
   ****/
  object *elements,*elements_back, *local_nodes,*local_nodes_back, *global_nodes;

  {
    /*
     * Elements
     */
    std::vector<index_int> elements_domain; distribution *elements_dist;
    for (int id=0; id<dim; id++)
      elements_domain.push_back( elocal*(layout->coord(id)) );
    elements_dist = new IMP_block_distribution(decomp,elements_domain);
    elements = new IMP_object(elements_dist);
    elements->set_name(fmt::format("elements{}D",dim));
    elements_back = new IMP_object(elements_dist);
    elements_back->set_name(fmt::format("elements_back{}D",dim));

    /*
     * Local nodes
     */
    std::vector<index_int> local_nodes_domain; distribution *local_nodes_dist;
    for (int id=0; id<dim; id++ )
      local_nodes_domain.push_back( 2*elements_domain[id] );
    local_nodes_dist = new IMP_block_distribution(decomp,local_nodes_domain);
    local_nodes = new IMP_object(local_nodes_dist);
    local_nodes->set_name("local_nodes");
    local_nodes_back = new IMP_object(local_nodes_dist);
    local_nodes_back->set_name("local_nodes_back");

    std::vector<index_int> global_nodes_domain; distribution *global_nodes_dist;
    for (int id=0; id<dim; id++)
      global_nodes_domain.push_back( elocal*layout->coord(id)+1 );
    global_nodes_dist = new IMP_block_distribution(decomp,global_nodes_domain);
    global_nodes = new IMP_object(global_nodes_dist);
    global_nodes->set_name("global_nodes");
  }

  /****
   **** Kernels
   ****/
  IMP_kernel *init_elements, *element_to_local_nodes, *local_to_global_nodes,
    *global_to_local_nodes;

  init_elements = new IMP_origin_kernel(elements);
  init_elements->set_localexecutefn( &vecsetlinear );

  /*
   * Elements to local nodes
   */
  element_to_local_nodes = new IMP_kernel(elements,local_nodes);
  element_to_local_nodes->set_name("element_to_local_nodes");
  element_to_local_nodes->last_dependency()->set_signature_function_function
    ( new multi_sigma_operator( dim, &signature_struct_element_to_local ) ); // coordinate?
  element_to_local_nodes->set_localexecutefn
    ( [dim] (int step,processor_coordinate &p,
	     std::vector<object*> *inobjects,object *outobject,double *cnt)
      -> void {
          return element_to_local_function(step,p,inobjects,outobject,dim,cnt); } );

  /*
   * Local nodes to global nodes
   */
  local_to_global_nodes = new IMP_kernel(local_nodes,global_nodes);
  local_to_global_nodes->last_dependency()->set_signature_function_function
    ( new multi_sigma_operator
      ( dim, [local_nodes]
	(std::shared_ptr<multi_indexstruct> g) -> std::shared_ptr<multi_indexstruct> {
	return signature_local_from_global( g,local_nodes->get_enclosing_structure() ); } ) );
  local_to_global_nodes->set_localexecutefn
    ( [local_nodes] (int step,processor_coordinate &p,std::vector<object*> *in,object *out,
		     double *flopcount) -> void {
      local_to_global_function
	(step,p,in,out,local_nodes->get_enclosing_structure(),flopcount); } );

  /*
   * Global nodes back to local nodes
   */
  global_to_local_nodes = new IMP_kernel(global_nodes,local_nodes_back);
  {
    global_to_local_nodes->last_dependency()->set_signature_function_function
      ( new multi_sigma_operator(dim,&signature_global_node_to_local) );
    auto 
      local_nodes_global_domain = local_nodes->get_enclosing_structure(),
      global_nodes_global_domain = global_nodes->get_enclosing_structure();
    global_to_local_nodes->set_localexecutefn
      ( [local_nodes_global_domain] (int step,processor_coordinate &p,
				     std::vector<object*> *in,object *out,double *ct)
	-> void {
	function_global_node_to_local(step,p,in,out,local_nodes_global_domain,ct); } );
  }

  /****
   **** Lulesh algorithm
   ****/
  
  //snippet lulesh algorithm
  algorithm *lulesh;
  lulesh = new IMP_algorithm(decomp);

  lulesh->add_kernel(init_elements);
  lulesh->add_kernel(element_to_local_nodes);
  lulesh->add_kernel(local_to_global_nodes);
  lulesh->add_kernel(global_to_local_nodes);

  if (trace) {
    lulesh->add_kernel( new IMP_trace_kernel(elements,"elements") );
    lulesh->add_kernel( new IMP_trace_kernel(local_nodes,"local_nodes") );
    lulesh->add_kernel( new IMP_trace_kernel(global_nodes,"global_nodes") );
    lulesh->add_kernel( new IMP_trace_kernel(local_nodes_back,"local_nodes_back") );
  }

  lulesh->analyze_dependencies();
  lulesh->execute();

  //snippet end

  //  delete env;

  return 0;
}
