/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** template_threepoint.cxx : 
 **** mode-independent template for threepoint averaging
 ****
 ****************************************************************/

/*! \page kmeans k-Means clustering

  This is incomplete.
*/

#include "template_common_header.h"
#include "kmeans_functions.h"

/****
 **** Main program
 ****/
int main(int argc,char **argv) {

  environment::print_application_options =
    [] () {
    printf("Kmeans options:\n");
    printf("  -n nnn : global number of points\n");
    printf("  -k nnn : number of clusters\n");
    printf("  -s nnn : number of steps\n");
    printf(":\n");
  };
  
  /* The environment does initializations, argument parsing, and customized printf
   */
  IMP_environment *env = new IMP_environment(argc,argv);
  env->set_name("kmeans");
  
  /* Print help information if the user specified "-h" argument */
  if (env->has_argument("h")) {
    printf("Usage: %s [-d] [-s nsteps] [-n size] [-k clusters]\n",argv[0]);
    return -1;
  }
      
  IMP_architecture *arch = dynamic_cast<IMP_architecture*>(env->get_architecture());
  IMP_decomposition* decomp = new IMP_decomposition(arch);

  int
    dim = 2,
    nsteps = env->iargument("s",1),
    ncluster = env->iargument("k",5),
    globalsize = env->iargument("n",env->get_architecture()->nprocs());

  /*
   * Define data
   */

  // centers are 2k replicated reals

  //snippet kmeanscenter
  IMP_distribution 
    *kreplicated = new IMP_replicated_distribution(decomp,dim,ncluster);
  IMP_object
    *centers = new IMP_object( kreplicated );
  //snippet end

  // coordinates are Nx2 with the N distributed
  //snippet kmeanscoord
  IMP_distribution
    *twoblocked = new IMP_block_distribution(decomp,dim,-1,globalsize);
  IMP_object
    *coordinates = new IMP_object( twoblocked );
  //snippet end

  // calculate Nxk distances, with the N distributed
  IMP_distribution
    *kblocked = new IMP_block_distribution(decomp,ncluster,-1,globalsize);
  IMP_object
    *distances = new IMP_object( kblocked );

  // grouping should be N integers, just use reals
  IMP_distribution
    *blocked = new IMP_block_distribution(decomp,-1,globalsize);
  IMP_object
    *grouping = new IMP_object( blocked );

  IMP_distribution
    *kdblocked = new IMP_block_distribution
        (decomp,ncluster*(dim+1),-1,globalsize);
  IMP_object
    *masked_coordinates = new IMP_object( kdblocked );

  /*
   * The kmeans algorithm
   */
  algorithm *kmeans = new IMP_algorithm( decomp );
  kmeans->set_name("K-means clustering");
  
  IMP_kernel
    *initialize_centers = new IMP_kernel(centers);
  initialize_centers->set_localexecutefn
    ( [] ( kernel_function_args ) -> void {
      set_initial_centers(outvector,p); } );
  kmeans->add_kernel( initialize_centers );

  IMP_kernel
    *set_initial_coordinates = new IMP_kernel( coordinates );
  set_initial_coordinates->set_name("set initial coordinates");
  set_initial_coordinates->set_localexecutefn(generate_random_coordinates);
  kmeans->add_kernel( set_initial_coordinates );

  IMP_kernel
    *calculate_distances = new IMP_kernel( coordinates,distances );
  calculate_distances->set_name("calculate distances");
  calculate_distances->set_localexecutefn(distance_calculation);
  calculate_distances->set_explicit_beta_distribution(coordinates);
  calculate_distances->add_in_object( centers);
  calculate_distances->set_explicit_beta_distribution( centers );
  kmeans->add_kernel( calculate_distances );

  IMP_kernel
    *find_nearest_center = new IMP_kernel( distances,grouping );
  find_nearest_center->set_name("find nearest center");
  find_nearest_center->set_localexecutefn( &group_calculation );
  find_nearest_center->set_explicit_beta_distribution( blocked );
  kmeans->add_kernel( find_nearest_center );

  IMP_kernel
    *group_coordinates = new IMP_kernel( coordinates,masked_coordinates );
  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  group_coordinates->add_sigma_oper( std::shared_ptr<ioperator>( new ioperator("no_op") ) );
  group_coordinates->add_in_object(grouping);
  group_coordinates->set_explicit_beta_distribution(grouping);
  kmeans->add_kernel( group_coordinates );

  kmeans->analyze_dependencies();
  kmeans->execute();

  delete env;

  return 0;
}

  // // masked coordinates is a Nx2k array with only 1 nonzero coordinate for each i<N
  // IMP_distribution
  //   *k2blocked = new IMP_distribution
  //       (decomp,"disjoint_block",dim*ncluster,-1,globalsize);
  // IMP_object
  //   *masked_coordinates = new IMP_object( k2blocked );
  // IMP_kernel
  //   *group_coordinates = new IMP_kernel( coordinates,masked_coordinates );
  // group_coordinates->set_name("group coordinates");
  // group_coordinates->localexecutefn = &coordinate_masking;
  // group_coordinates->add_in_object(grouping);
  // group_coordinates->last_dependency()->set_explicit_beta_distribution(grouping);
  //  // group_coordinates->localexecutectx = grouping;
  // group_coordinates->add_sigma_oper( new ioperator("no_op") );
  // kmeans->add_kernel( group_coordinates );

  // // locally sum the masked coordinates
  // IMP_distribution
  //   *klocal = new IMP_block_distribution(decomp,2*k,1,-1);
  // IMP_object
  //   *partial_sums = new IMP_object( klocal );
  // IMP_kernel
  //   *compute_new_centers1 = new IMP_kernel( masked_coordinates,partial_sums );
  // printf("alpha ortho %d\n",masked_coordinates->get_orthogonal_dimension());
  // compute_new_centers1->set_name("partial sum calculation");
  // compute_new_centers1->localexecutefn = &center_calculation_partial;
  // compute_new_centers1->set_explicit_beta_distribution( k2blocked );
  // printf("beta ortho %d\n",k2blocked->get_orthogonal_dimension());
  // kmeans->add_kernel( compute_new_centers1 );

