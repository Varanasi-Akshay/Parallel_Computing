/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** kmeans clustering
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_ops.h"
#include "mpi_static_vars.h"
#include "kmeans_functions.h"

// specific parameters for kmeans
int nsteps, k, globalsize;
two_object_struct two_objects;

void dummy_centers_1d( std::shared_ptr<object> centers,int icase ) {
  auto comm = static_cast<communicator*>(centers.get());
  int ntids = comm->nprocs(), mytid = comm->procid();
  int
    k = centers->global_volume(); //get_orthogonal_dimension();
  printf("ortho: %d\n",k);
  double *center_data;
  try { center_data = centers->get_data(mycoord);
  } catch (...) { throw(std::string("could not get data in initial centers")); }

  switch (icase) {
  case 0 : // centers 1D, equally spaced
  case 1 : // centers 1D, equally spaced
    for (int ik=0; ik<k; ik++) {
      center_data[ik] = ( (double)ik+1 )/(k+1);
      //printf("[%d] set %d as %e\n",mytid,ik,center_data[ik]);
    }
    break;
  case 2 : // centers 1D, random
    srand((int)(mytid*(double)RAND_MAX/ntids));
    for (int ik=0; ik<k; ik++) {
      float randomfraction = (rand() / (double)RAND_MAX);
      center_data[ik] = randomfraction;
    }
    break;
  }
}

//! Set all coordinates on this processor to the normalized processor number
void coordinates_on_center_1d( kernel_function_args )
{
  auto outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index( *outvector.get(),p ),
    len = outvector->volume(p),
    gsize = outvector->global_volume();

  int k = outvector->get_orthogonal_dimension();

  int tar = tar0;
  for (index_int i=0; i<len; i++) {
    index_int iset = outvector->first_index_r(p).coord(0)+1;
    double v = ( (double)iset )/gsize;
    //printf("[%d] set l:%d g:%d to %e\n",p->coord(0),i,tar0+i,v);
    for (int kk=0; kk<k; kk++)
      outdata[tar++] = v;
  }
};

void coordinates_linear( kernel_function_args )
{
  double
    *outdata = outvector->get_data(p);

  int k = outvector->get_orthogonal_dimension();

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // various structures
  auto out_nstruct = outvector->get_numa_structure(),
    out_gstruct = outvector->get_enclosing_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_gsize = out_gstruct->local_size_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (outvector->get_dimensionality()==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int
	I = INDEX1D(i,out_offsets,out_nsize),
	Ig = COORD1D(i,out_gsize);
      // fmt::print("[{}] index {} location {} is global {}\n",
      // 		 p->as_string(),i,I,Ig);
      for (int kk=0; kk<k; kk++)
	outdata[I*k+kk] = (double)Ig;
    }
  } else
    throw(std::string("kmeans only works on linear coordinates"));

};

TEST_CASE( "work with predictable centers","[1]" ) {
  return;
  INFO( "mytid=" << mytid );

  for (int icase=0; icase<=2; icase++) {
    int dim,ncluster,globalsize;

    INFO( "case=" << icase );

    // Cases
    // 0: one dimensional, put the initial points on the centers
    // 1: same as zero, but using collectives
    // 2: one dimensional, random points
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      ncluster = env->get_architecture()->nprocs(); dim = 1; globalsize = ncluster;
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }
    INFO( "number of clusters: " << ncluster );
    
    /* 
     * All the declarations; dependent on problem parameters
     */

    // centers are replicated, size = ncluster, ortho = dim
    mpi_distribution 
      *kreplicated = new mpi_replicated_distribution(decomp,dim,ncluster);
    auto centers = std::shared_ptr<object>( new mpi_object( kreplicated ) );
    CHECK( centers->get_orthogonal_dimension()==1 );
    CHECK( centers->global_volume()==dim*ncluster );
    
    // coordinates are N x dim with the N distributed
    mpi_distribution
      *twoblocked = new mpi_block_distribution
          (decomp,dim,-1,globalsize);
    auto coordinates = std::shared_ptr<object>( new mpi_object( twoblocked ) );

    // distances are Nxk, with the N distributed
    //snippet kmeansdistance
    mpi_distribution
      *kblocked = new mpi_block_distribution
          (decomp,ncluster,-1,globalsize);
    auto distances = std::shared_ptr<object>( new mpi_object( kblocked ) );
    //snippet end

    // grouping should be N integers, just use reals
    //snippet kmeansgroup
    mpi_distribution
      *blocked = new mpi_block_distribution
          (decomp,-1,globalsize);
    auto grouping = std::shared_ptr<object>( new mpi_object( blocked ) );
    //snippet end

    // VLE only for tracing?
    //snippet kmeansmindist
    auto min_distance = std::shared_ptr<object>( new mpi_object( blocked ) );
    //snippet end

    // masked coordinates is a Nx2k array with only 1 nonzero coordinate for each i<N
    //snippet kmeansmask
    mpi_distribution
      *k2blocked = new mpi_block_distribution
          (decomp,ncluster*dim,-1,globalsize);
    auto masked_coordinates = std::shared_ptr<object>( new mpi_object( k2blocked ) );
    //snippet end
    
    // sum masked coordinates
    //snippet kmeansnewcenters
    mpi_distribution
      *klocal = new mpi_block_distribution(decomp,ncluster*dim,1,-1);
    auto partial_sums = std::shared_ptr<object>( new mpi_object( klocal ) );
    mpi_kernel
      *compute_new_centers1 = new mpi_kernel( masked_coordinates,partial_sums );
    //snippet end
    
    /*
     * set initial centers
     */
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      dummy_centers_1d( centers,icase ); // not in the queue
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // test correct dummy centers
    CHECK( centers->volume(mycoord)==ncluster );
    double *centerdata = centers->get_data(mycoord);
    REQUIRE( centerdata!=nullptr );
    for (int icluster=0; icluster<ncluster; icluster++) {
      INFO( "checking cluster " << icluster );
      double c = centerdata[icluster];
      switch (icase) {
      case 0 :
      case 1 :
	CHECK( c == Approx( (double)(icluster+1)/(ncluster+1) ) );
	break;
      case 2 :
	CHECK(  ( (c>=0.) && (c<=1.) ) );
	break;
      }
    }

    /*
     * set initial coordinates
     */

    //snippet kmeansinitcoord
    mpi_kernel
      *set_random_coordinates = new mpi_kernel( coordinates );
    //snippet end

    set_random_coordinates->set_name("set random coordinates");
    switch (icase) {
    case 0 :
    case 1 :
    case 2 :
      set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // check that the local coordinates are in place
    REQUIRE_NOTHROW( set_random_coordinates->analyze_dependencies() );
    REQUIRE_NOTHROW( set_random_coordinates->execute() );
    {
      INFO( "test initial coordinates" );
      double *centerdata = coordinates->get_data(mycoord);
      REQUIRE( (long int)centerdata!=(long int)NULL );
      double c = centerdata[0]; 
      switch (icase) {
      case 0 :
      case 1 :
    	CHECK( c == Approx( (double)(mytid+1)/(globalsize+1) ) );
    	break;
      case 2 :
	CHECK(  ( (c>=0.) && (c<=1.) ) );
	break;
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }
    return;

    /* 
     * distance calculation: outer product of 
     * my coordinates x all centers
     */
    REQUIRE( coordinates!=nullptr );
    REQUIRE( distances!=nullptr );
    REQUIRE( centers!=nullptr );
    mpi_kernel *calculate_distances;
    switch (icase) {
    case 0 :
      //snippet kmeansdistcalc
      calculate_distances = new mpi_kernel( coordinates,distances );
      calculate_distances->last_dependency()->set_explicit_beta_distribution(coordinates.get());
      calculate_distances->add_in_object( centers);
      calculate_distances->last_dependency()->set_explicit_beta_distribution( centers.get() );
      calculate_distances->set_localexecutefn( &distance_calculation);
      //snippet end
      break ;
    case 1 :
    case 2 :
      //snippet kmeansdistkernel
      printf("the outer product kernel still uses the context. Wrong!\n");
      calculate_distances = new mpi_outerproduct_kernel
	( coordinates,distances,centers,&distance_calculation );
      //snippet end
      break;
    }
    calculate_distances->set_name("calculate distances");

    REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
    { // check that this is a local operation
      std::vector<std::shared_ptr<task>> tasks;
      REQUIRE_NOTHROW( tasks = calculate_distances->get_tasks() );
      for (auto t : tasks ) {
	CHECK( t->get_receive_messages().size()==1 );
	CHECK( t->get_send_messages().size()==1 );
      }
    }

    // that gives a zero distance to the mytid'th center
    calculate_distances->execute(); // REQUIRE_NOTHROW( calculate_distances->execute() );

    {
      double *dist = distances->get_data(mycoord);
      switch (icase) {
      case 0 :
      case 1 :
	for (int icluster=0; icluster<ncluster; icluster++) {
	  INFO( "icluster=" << icluster );
	  if (icluster==mytid) {
	    CHECK( dist[icluster] == Approx( 0.0 ) );
	  } else {
	    CHECK( dist[icluster] != Approx( 0.0 ) );
	  }
	}
	break;
      case 2 :
	break; // initial coordinates are not in any relation to the centers
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }

    /*
     * set initial grouping (i'th point in i'th group) and compute distance
     */
    mpi_kernel *find_nearest_center;
    switch (icase) {
    case 0:
      //snippet kmeansnearcalc
      find_nearest_center = new mpi_kernel( distances,grouping );
      find_nearest_center->set_localexecutefn( &group_calculation );
      find_nearest_center->set_explicit_beta_distribution( blocked );
      //find_nearest_center->last_dependency()->set_type_local();
      //snippet end
      break;
    case 1 :
    case 2 :
      //snippet kmeansnearkernel
      find_nearest_center = new mpi_outerproduct_kernel
	( distances,grouping,NULL,group_calculation);
      //snippet end
      break;
    }
    find_nearest_center->set_name("find nearest center");

    {
      double *group = grouping->get_data(mycoord);
      switch (icase) {
      case 0 : 
      case 1 :
      case 2 :
	group[0] = (double) mytid;
	break;
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    // compute new grouping
    REQUIRE_NOTHROW( find_nearest_center->execute() );
    {
      double *group = grouping->get_data(mycoord);
      switch (icase) {
      case 0 :
      case 1 :
	CHECK( group[0]== Approx( (double)mytid ) );
	break;
      case 2 :
	break; // no idea what the proper group is
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    /*
     * mask coordinates
     */
    mpi_kernel *group_coordinates;
    switch (icase) {
    case 0 :
      //snippet kmeansgroupcalc
      group_coordinates = new mpi_kernel( coordinates,masked_coordinates );
      group_coordinates->set_localexecutefn( &coordinate_masking );
      group_coordinates->add_in_object(grouping);
      group_coordinates->last_dependency()->set_explicit_beta_distribution(grouping.get());
      group_coordinates->last_dependency()->set_type_local();
      //snippet end
      break;
    case 1 :
    case 2 :
      //snippet kmeansgroupkernel
      group_coordinates = new mpi_outerproduct_kernel
	( coordinates,masked_coordinates,grouping,coordinate_masking );
      //snippet end
      break;
    }
    group_coordinates->set_name("group coordinates");
    group_coordinates->execute();

    {
      double *group = masked_coordinates->get_data(mycoord);
      CHECK( masked_coordinates->volume(mycoord)==1 ); // hm
      CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster );
      switch (icase) {
      case 0 :
      case 1 :
	for (int icluster=0; icluster<ncluster; icluster++) {
	  if (icluster!=mytid) {
	    CHECK( group[icluster] == Approx(0) );
	  } else {
	    CHECK( group[icluster] == Approx( (double)(mytid+1)/(globalsize+1) ) );
	  }
	}
	break;
      case 2 :
	break;
      default: printf("No case %d implemented\n",icase); throw(6);
      }
    }

    // locally sum the masked coordinates
    //snippet kmeansnewcenterscalc
    compute_new_centers1->set_name("partial sum calculation");
    compute_new_centers1->set_localexecutefn( &center_calculation_partial );
    compute_new_centers1->last_dependency()->set_type_local();
    //snippet end
    CHECK_NOTHROW( compute_new_centers1->execute() );

    { 
      double *centerdata = coordinates->get_data(mycoord);
      double *newcenter = partial_sums->get_data(mycoord);
      switch (icase) {
      case 0 : // the partial sums should be equal to the old 
      case 1 :
	// for (int k=0; k<ncluster*dim; k++)
	//   CHECK( centerdata[k]==Approx(newcenter[k]) );
	break;
      case 2 :
	break;
      default : throw("Unimplemented\n"); 
      }
    }

    //delete centers;
    //delete kreplicated;
  } // end of case loop
};

TEST_CASE( "distance to centers","[10]" ) {
  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( "dim=" << dim << ", points per process=" << localsize );

  mpi_distribution
    *dblocked = new mpi_block_distribution(decomp,dim,-1,globalsize);
  auto coordinates = std::shared_ptr<object>( new mpi_object( dblocked ) );

  mpi_kernel
    *set_random_coordinates = new mpi_kernel( coordinates );
  set_random_coordinates->set_name("set random coordinates");
  set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
  REQUIRE_NOTHROW( set_random_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_random_coordinates->execute() );
  {
    double *coordinate_data; REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );
    double c = coordinate_data[0]; 
    CHECK( c == Approx( (double)(mytid*localsize+1)/globalsize ) );
  }
  
  mpi_distribution 
    *kreplicated = new mpi_replicated_distribution(decomp,dim,ncluster);
  auto centers = std::shared_ptr<object>( new mpi_object( kreplicated ) );
  CHECK( centers->volume(mycoord)==ncluster );
  CHECK( centers->get_orthogonal_dimension()==dim );
  {
    REQUIRE_NOTHROW( centers->allocate() );
    double *centerdata ; REQUIRE_NOTHROW( centerdata = centers->get_data(mycoord) );
    int iloc = 0;
    for (int tid=0; tid<ntids; tid++) {
      for (int idim=0; idim<dim; idim++)
	centerdata[iloc++] = (double)(tid*localsize+1)/globalsize;
    }
  }

  mpi_distribution
    *kblocked = new mpi_block_distribution(decomp,ncluster,-1,globalsize);
  auto distances = std::shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  mpi_kernel
    *calculate_distances = new mpi_kernel( coordinates,distances );
  calculate_distances->set_explicit_beta_distribution(coordinates.get());
  calculate_distances->add_in_object( centers);
  calculate_distances->set_explicit_beta_distribution( centers.get() );
  calculate_distances->set_localexecutefn( &distance_calculation);

  REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_distances->execute() );
  {
    double *distance_data; REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (int icluster=0; icluster<ncluster; icluster++) {
      double dist = distance_data[icluster]; 
      INFO( "[" << mytid << "] cluster " << icluster << ", distance=" << dist );
      if (icluster==mytid)
	CHECK( dist==Approx(0.) );
      else
	CHECK( dist>=(.999/ntids) );
    }
  }
}

TEST_CASE( "find nearest center","[11]" ) {
  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( "dim=" << dim << ", points per process=" << localsize );

  mpi_distribution
    *kblocked = new mpi_block_distribution(decomp,ncluster,-1,globalsize);
  auto distances = std::shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  // set dummy distances
  {
    double *distance_data; REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (int icluster=0; icluster<ncluster; icluster++) {
      if (icluster==mytid)
	distance_data[icluster] = 0.;
      else
	distance_data[icluster] = 1./ntids;
    }
  }

  mpi_distribution
    *blocked = new mpi_block_distribution(decomp,-1,globalsize);
  auto grouping = std::shared_ptr<object>( new mpi_object( blocked ) );
  mpi_kernel
    *find_nearest_center = new mpi_kernel( distances,grouping );
  find_nearest_center->set_name("find nearest center");
  REQUIRE_NOTHROW( find_nearest_center->set_localexecutefn( &group_calculation ) );
  REQUIRE_NOTHROW( find_nearest_center->set_explicit_beta_distribution( blocked ) );
  REQUIRE_NOTHROW( find_nearest_center->analyze_dependencies() );
  REQUIRE_NOTHROW( find_nearest_center->execute() );

  {
    double *groups = grouping->get_data(mycoord);
    CHECK( grouping->volume(mycoord)==localsize );
    for (int i=0; i<localsize; i++) {
      CHECK( groups[i]==Approx(mytid) );
    }
  }
}

TEST_CASE( "group coordinates","[12]" ) {

  index_int localsize, globalsize;
  int dim, ncluster = ntids;

  SECTION( "1d" ) { dim = 1; localsize = 1; }
  SECTION( "2d" ) { dim = 2; localsize = 1; }
  globalsize = localsize*ntids;
  INFO( "dim=" << dim << ", points per process=" << localsize );

  algorithm *kmeans = new mpi_algorithm( decomp );
  kmeans->set_name(fmt::format("K-means clustering in {} dimensions",dim));

  mpi_distribution
    *dblocked = new mpi_block_distribution(decomp,dim,-1,globalsize);
  auto coordinates = std::shared_ptr<object>( new mpi_object( dblocked ) );

  mpi_kernel
    *set_random_coordinates = new mpi_origin_kernel( coordinates );
  set_random_coordinates->set_name("set random coordinates");
  set_random_coordinates->set_localexecutefn( &coordinates_on_center_1d );
  REQUIRE_NOTHROW( kmeans->add_kernel( set_random_coordinates ) );

  // make grouping array
  mpi_distribution
    *blocked = new mpi_block_distribution(decomp,-1,globalsize);
  auto grouping = std::shared_ptr<object>( new mpi_object( blocked ) );
  REQUIRE_NOTHROW( grouping->allocate() );
  {
    double *groups = grouping->get_data(mycoord);
    CHECK( grouping->volume(mycoord)==localsize );
    for (int i=0; i<localsize; i++) {
      REQUIRE_NOTHROW( groups[i]=(double)mytid );
    }
  }
  REQUIRE_NOTHROW( kmeans->add_kernel( new mpi_origin_kernel(grouping) ) );

  // masked coordinates is a N-by-kx(dim+1) array
  // where "dim+1" stands for "flag plus coordinate":
  // a negative flag is not reduced over
  mpi_distribution
    *kdblocked = new mpi_block_distribution
        (decomp,ncluster*(dim+1),-1,globalsize);
  std::shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates= std::shared_ptr<object>( new mpi_object( kdblocked ) ) );
  mpi_kernel *group_coordinates;
  REQUIRE_NOTHROW( group_coordinates = new mpi_kernel( coordinates,masked_coordinates ) );
  group_coordinates->add_sigma_operator( ioperator("no_op")  );
  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  REQUIRE_NOTHROW( group_coordinates->add_in_object(grouping) );
  REQUIRE_NOTHROW( group_coordinates->set_explicit_beta_distribution(grouping.get()) );
  kmeans->add_kernel( group_coordinates );

  REQUIRE_NOTHROW( kmeans->analyze_dependencies() );
  REQUIRE_NOTHROW( kmeans->execute() );

  {
    double *coord_data, *mask_data;
    REQUIRE_NOTHROW( coord_data = coordinates->get_data(mycoord) );
    REQUIRE_NOTHROW( mask_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );
    INFO( "inspecting mask on proc " << mytid );
    for (int ic=0; ic<ncluster; ic++) {
      INFO( "cluster " << ic );
      // first the mask parameter
      if (ic==mytid) {
	CHECK( mask_data[ic*(dim+1)]==Approx(1.) );
      } else {
	CHECK( mask_data[ic*(dim+1)]==Approx(-1.) );
      }
      // then the coordinates; duplicated for each cluster.
      for (int id=0; id<dim; id++)
	CHECK( coord_data[id]==Approx(mask_data[ic*(dim+1)+id+1]) );
    }
  }
}

TEST_CASE( "compute new centers","[13]" ) {

  int dim, ncluster = ntids, localsize = 1, globalsize = ntids*localsize;
  SECTION( "1D" ) { dim = 1; }
  INFO( "dim=" << dim );

  // masked coordinates is a N-by-kx(dim+1) array
  // where "dim+1" stands for "flag plus coordinate":
  // a negative flag is not reduced over
  mpi_distribution
    *kdblocked = new mpi_block_distribution
        (decomp,ncluster*(dim+1),localsize,-1);
  std::shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates= std::shared_ptr<object>( new mpi_object( kdblocked ) ) );
  masked_coordinates->allocate();
  {
    double *mask_data;
    REQUIRE_NOTHROW( mask_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    index_int k = ncluster*(dim+1);
    CHECK( masked_coordinates->get_orthogonal_dimension()==k );
    INFO( "inspecting mask on proc " << mytid );
    for (int ilocal=0; ilocal<localsize; ilocal++) {
      double *point = mask_data + ilocal*k;
      // for each point we have a coordinate+mask per cluster
      // the coordinates are all the same ?! check with [12]
      // that's fine for low numbers of clusters.
      for (int ic=0; ic<ncluster; ic++) {
	INFO( "cluster " << ic );
	for (int id=0; id<dim; id++) {
	  point[ ic*(dim+1)+id ] = 1./(mytid+1); // ???? 1./(ic+1);
	}
	// if the cluster corresponds to this processor: select
	if (ic==mytid) {
	  point[ ic*(dim+1)+dim ] = 1.;
	} else {
	  point[ ic*(dim+1)+dim ] = -1.;
	}
      }
    }
  }
  mpi_kernel *make_masked = new mpi_origin_kernel(masked_coordinates);
  make_masked->analyze_dependencies();
  make_masked->execute();

  /*
   * Reduct!
   */
  mpi_distribution 
    *kreplicated = new mpi_replicated_distribution(decomp,dim,ncluster);
  auto centers = std::shared_ptr<object>( new mpi_object( kreplicated ) );

  mpi_kernel *new_centers = new mpi_kernel(masked_coordinates,centers);
  if (dim==1) {
    new_centers->set_localexecutefn( &masked_reduction_1d );
  } else throw(std::string("masked reduction only in 1d"));
  new_centers->set_explicit_beta_distribution(kdblocked);
  REQUIRE_NOTHROW( new_centers->analyze_dependencies() );
  REQUIRE_NOTHROW( new_centers->execute() );

  {
    double *center_data;
    REQUIRE_NOTHROW( center_data = centers->get_data(mycoord) );
    CHECK( centers->volume(mycoord)==ncluster );
    CHECK( centers->get_orthogonal_dimension()==dim );
    for (int ic=0; ic<ncluster; ic++) {
      INFO( "cluster " << ic );
      for (int id=0; id<dim; id++) {
	center_data[ ic*dim + id ] = 1./(mytid+1);
      }
    }
  }
}

TEST_CASE( "distance to centers, general","[20]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;

  SECTION( "1d" ) { dim = 1; localsize = 5 * clusters_per_proc; }
  SECTION( "2d" ) { dim = 2; localsize = 5 * clusters_per_proc; }
  globalsize = localsize*ntids;
  INFO( fmt::format
	("[{}] dim={}, clusters per processor={} for a total of: {}; points per process={}",
	 mycoord.as_string(),dim,clusters_per_proc,ncluster,localsize) );
  mpi_distribution
    *dblocked = new mpi_block_distribution(decomp,dim,localsize,-1);
  auto coordinates = std::shared_ptr<object>( new mpi_object( dblocked ) );
  CHECK( coordinates->global_volume()==globalsize );

  /*
   * Set initial coordinates to the point indices, in every dimension:
   * (f,f,f) (f+1,f+1,f+1) ....
   */
  mpi_kernel
    *set_linear_coordinates = new mpi_kernel( coordinates );
  set_linear_coordinates->set_name("set random coordinates");
  set_linear_coordinates->set_localexecutefn( &coordinates_linear );
  REQUIRE_NOTHROW( set_linear_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_linear_coordinates->execute() );
  {
    double *coordinate_data;
    REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );
    index_int linear_first;
    REQUIRE_NOTHROW
      ( linear_first = coordinates->first_index_r(mycoord)
	.linear_location_in( coordinates->get_enclosing_structure() ) );
    int d = 0;
    for (int c=0; c<localsize; c++) {
      for (int id=0; id<dim; id++) {
	INFO( fmt::format("local coordinate: {}, dim: {}, linear: {}",c,id,d) );
	CHECK( coordinate_data[d++] == Approx( (double)linear_first +c  ) );
      }
    }
  }

  // nclusters is the #centers, each of size dim, and replicated
  mpi_distribution 
    *kreplicated = new mpi_replicated_distribution(decomp,dim,ncluster);
  auto centers = std::shared_ptr<object>( new mpi_object( kreplicated ) );
  CHECK( centers->volume(mycoord)==ncluster );
  CHECK( centers->get_orthogonal_dimension()==dim );
  // equally spaced cluster centers over the points
  fmt::MemoryWriter w; w.write("Cluster centers:");
  {
    REQUIRE_NOTHROW( centers->allocate() );
    double *centerdata ; REQUIRE_NOTHROW( centerdata = centers->get_data(mycoord) );
    int iloc = 0;
    for (int icluster=0; icluster<ncluster; icluster++) {
      double cluster_center = icluster * globalsize / (double)ncluster;
      w.write(" {},",cluster_center);
      for (int idim=0; idim<dim; idim++)
	centerdata[iloc++] = cluster_center;
    }
  }
  INFO( w.str() );

  // for each point the distance to the cluster centers
  mpi_distribution
    *kblocked = new mpi_block_distribution(decomp,ncluster,-1,globalsize);
  auto distances = std::shared_ptr<object>( new mpi_object( kblocked ) );
  CHECK( distances->volume(mycoord)==localsize );
  CHECK( distances->get_orthogonal_dimension()==ncluster );

  mpi_kernel
    *calculate_distances = new mpi_kernel( coordinates,distances );
  calculate_distances->last_dependency()->set_explicit_beta_distribution(coordinates.get());
  calculate_distances->add_in_object( centers);
  calculate_distances->last_dependency()->set_explicit_beta_distribution( centers.get() );
  calculate_distances->set_localexecutefn( &distance_calculation);

  REQUIRE_NOTHROW( calculate_distances->analyze_dependencies() );
  REQUIRE_NOTHROW( calculate_distances->execute() );

  {
    double *distance_data;
    REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (index_int ipoint=0; ipoint<localsize; ipoint++) {
      for (int icluster=0; icluster<ncluster; icluster++) {
	double
	  point_loc = ipoint+mytid*localsize,
	  cluster_loc = icluster*globalsize/(double)ncluster,
	  step_dist = std::abs( point_loc - cluster_loc );
	INFO( fmt::format("point {} at {} to cluster {} at {} has step dist {}",
			  ipoint,point_loc,icluster,cluster_loc,step_dist));
	double dist = 0;
	for (int id=0; id<dim; id++)
	  dist += step_dist*step_dist;
	dist = sqrt(dist);
	double d = distance_data[INDEXpointclusterdist(ipoint,icluster,localsize,ncluster)];
	CHECK( d==Approx(dist) );
      }
    }
  }
}

TEST_CASE( "Find nearest center, general","[30]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;

  SECTION( "1d" ) { dim = 1; localsize = 5 * clusters_per_proc; }
  SECTION( "2d" ) { dim = 2; localsize = 5 * clusters_per_proc; }
  globalsize = localsize*ntids;
  INFO( fmt::format
	("[{}] dim={}, clusters per processor={} for a total of: {}; points per process={}",
	 mycoord.as_string(),dim,clusters_per_proc,ncluster,localsize) );

  // for each point the distance to the cluster centers
  mpi_distribution
    *kblocked = new mpi_block_distribution(decomp,ncluster,-1,globalsize);
  auto distances = std::shared_ptr<object>( new mpi_object( kblocked ) );

  { // copied from above
    double *distance_data;
    REQUIRE_NOTHROW( distance_data = distances->get_data(mycoord) );
    for (index_int ipoint=0; ipoint<localsize; ipoint++) {
      fmt::MemoryWriter w;
      index_int gpoint = ipoint+mytid*localsize;
      double point_loc = gpoint;
      w.write("point g={}, cluster dists:",gpoint);
      for (int icluster=0; icluster<ncluster; icluster++) {
	double
	  cluster_loc = icluster*globalsize/(double)ncluster,
	  step_dist = std::abs( point_loc - cluster_loc );
	w.write(" {}:{}->{}",icluster,cluster_loc,step_dist);
	double dist = 0;
	for (int id=0; id<dim; id++)
	  dist += step_dist*step_dist;
	dist = sqrt(dist);
	REQUIRE_NOTHROW
	  ( distance_data[INDEXpointclusterdist(ipoint,icluster,localsize,ncluster)] = dist );
      }
      fmt::print("{}\n",w.str());
    }
  }

  // make grouping array
  mpi_distribution
    *blocked = new mpi_block_distribution(decomp,-1,globalsize);
  auto grouping = std::shared_ptr<object>( new mpi_object( blocked ) );
  REQUIRE_NOTHROW( grouping->allocate() );

  mpi_kernel
    *find_nearest_center = new mpi_kernel( distances,grouping );
  find_nearest_center->set_name("find nearest center");
  find_nearest_center->set_localexecutefn( &group_calculation );
  find_nearest_center->set_explicit_beta_distribution( blocked );

  find_nearest_center->analyze_dependencies();
  find_nearest_center->execute();

  {
    double *group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );
    CHECK( grouping->volume(mycoord)==localsize );
    for (index_int i=0; i<localsize; i++) {
      INFO( fmt::format("Proc {}, local point {}, global {}",
			mycoord.as_string(),i,i+mytid*localsize) );
      CHECK( group_data[i]>=0 );
      CHECK( group_data[i]<ncluster );
      CHECK( group_data[i]>=mytid*clusters_per_proc );
    }
  }
}

TEST_CASE( "group coordinates, general","[40]" ) {
  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;


  SECTION( "1d" ) { dim = 1; }
  SECTION( "2d" ) { dim = 2; }

  localsize = 5 * clusters_per_proc;
  globalsize = localsize*ntids;
  INFO( fmt::format("{} dim={} global size {}",mycoord.as_string(),dim,globalsize) );

  mpi_distribution *blocked;
  REQUIRE_NOTHROW( blocked = new mpi_block_distribution(decomp,-1,globalsize) );
  std::shared_ptr<object> grouping;
  REQUIRE_NOTHROW( grouping = std::shared_ptr<object>( new mpi_object( blocked ) ) );

  mpi_distribution *twoblocked;
  REQUIRE_NOTHROW( twoblocked = new mpi_block_distribution(decomp,dim,-1,globalsize) );
  std::shared_ptr<object> coordinates;
  REQUIRE_NOTHROW( coordinates = std::shared_ptr<object>( new mpi_object( twoblocked ) ) );
  CHECK( coordinates->get_orthogonal_dimension()==dim );

  domain_coordinate myfirst;
  REQUIRE_NOTHROW( myfirst = blocked->get_processor_structure(mycoord)->first_index_r() );
  CHECK( myfirst.get_dimensionality()==1 );

  /*
   * Set initial coordinates to the point indices, in every dimension:
   * (f,f,f) (f+1,f+1,f+1) ....
   * as in [20] above
   */
  mpi_kernel
    *set_linear_coordinates = new mpi_kernel( coordinates );
  set_linear_coordinates->set_name("set random coordinates");
  set_linear_coordinates->set_localexecutefn( &coordinates_linear );
  REQUIRE_NOTHROW( set_linear_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( set_linear_coordinates->execute() );
  { // check
    double *coordinate_data = coordinates->get_data(mycoord);
    for (index_int i=0; i<localsize; i++) {
      for (int id=0; id<dim; id++) {
	CHECK( coordinate_data[ i*dim+id ]==Approx(myfirst[0]+i) );
      }
    }
  }

  // masked coordinates are ncluster times orthogonal version of block
  // but we need an extra location to indicate inclusion
  // so that we can implement a masked MPI reduce.....
  mpi_distribution *kdblocked;
  REQUIRE_NOTHROW
    ( kdblocked = new mpi_block_distribution(decomp,ncluster*(dim+1),-1,globalsize) );
  std::shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates = std::shared_ptr<object>( new mpi_object( kdblocked ) ) );

  { // set the group of each data point to the first local cluster
    // sort of as in [30] above
    double *group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );
    CHECK( grouping->volume(mycoord)==localsize );
    for (index_int i=0; i<localsize; i++) {
      group_data[i] = mytid*clusters_per_proc;
    }
  }
  
  mpi_kernel
    *group_coordinates = new mpi_kernel( coordinates,masked_coordinates );

  group_coordinates->set_name("group coordinates");
  group_coordinates->set_localexecutefn( &coordinate_masking );
  group_coordinates->add_sigma_operator( ioperator("no_op")  );
  group_coordinates->add_in_object(grouping);
  group_coordinates->set_explicit_beta_distribution(grouping.get());

  REQUIRE_NOTHROW( group_coordinates->analyze_dependencies() );
  REQUIRE_NOTHROW( group_coordinates->execute() );

  {
    double *masked_data;
    REQUIRE_NOTHROW( masked_data = masked_coordinates->get_data(mycoord) );
    CHECK( masked_coordinates->volume(mycoord)==localsize );
    CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );

    double *group_data;
    REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );

    double *coordinate_data;
    REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );

    for (index_int i=0; i<localsize; i++) {
      double *masked_coordinate = masked_data+i*ncluster*(dim+1);

      // what group does this point belong to?
      int group = (int)( group_data[i] );
      INFO( fmt::format("local point {} is in group {}",i,group) );

      // compare the point to all clusters
      for (int ic=0; ic<ncluster; ic++) {
	INFO( fmt::format("cluster {} out of {}",ic,ncluster) );
	int maskv = masked_coordinate[ ic*(dim+1) ];
	// go through all the dimensions of a point
	if (ic==group)
	  CHECK( maskv==Approx(+1) );
	else
	  CHECK( maskv==Approx(-1) );
	for (int id=0; id<dim; id++) {
	  auto
	    mxi = masked_coordinate[ id+1 ], // first position is mask, so id+1
	    xi = coordinate_data[ i*dim+id ];
	  INFO( fmt::format
	  	("point belongs to cluster: @idim={} value={}, should be f+i={}+{}={}",
	  	 id,mxi,
		 myfirst[0],i,xi) );
	  // if the point belongs to this cluster: nonzero
	  CHECK( mxi==Approx(xi) );
	}
	// test the mask value?
      }
    }
  }
}

// defined in mpi_kmeans_kernel.cxx
void add_if_mask1( void *indata, void * outdata,int *len,MPI_Datatype *type );

TEST_CASE( "custom MPI reduction, one point per proc","[50]" ) {

  double coordinate_and_mask[2], result_and_mask[2], result;
  coordinate_and_mask[0] = -1;
  coordinate_and_mask[1] = mytid;

  SECTION( "pick first point" ) { result = 0;
    if (mytid==result)
      coordinate_and_mask[0] = +1;
  }

  SECTION( "pick last point" ) { result = ntids-1;
    if (mytid==result)
      coordinate_and_mask[0] = +1;
  }

  SECTION( "pick second and last point" ) { result = ntids;
    if (ntids<3) {
      printf("[50] needs 3 procs for one test\n"); return; }
    if (mytid==1 || mytid==ntids-1)
      coordinate_and_mask[0] = +1;
  }

  MPI_Datatype dim1_type;
  MPI_Type_contiguous(2,MPI_DOUBLE,&dim1_type);
  MPI_Type_commit(&dim1_type);

  MPI_Op masked_add;
  MPI_Op_create(add_if_mask1,1,&masked_add);

  MPI_Allreduce
    (coordinate_and_mask,result_and_mask,
     1,dim1_type,masked_add,MPI_COMM_WORLD);

  CHECK( result_and_mask[0]==Approx(1) );
  CHECK( result_and_mask[1]==Approx(result) );
  
  MPI_Type_free(&dim1_type);
  MPI_Op_free(&masked_add);
}

TEST_CASE( "reduce to new center","[51]" ) {

  index_int localsize, globalsize;
  int dim, clusters_per_proc = 2, ncluster = clusters_per_proc * ntids;


  SECTION( "1d" ) { dim = 1; }
  SECTION( "2d" ) { dim = 2; }

  localsize = 5 * clusters_per_proc;
  globalsize = localsize*ntids;
  INFO( fmt::format("{} dim={} global size {}",mycoord.as_string(),dim,globalsize) );

  mpi_distribution *blocked;
  REQUIRE_NOTHROW( blocked = new mpi_block_distribution(decomp,-1,globalsize) );
  std::shared_ptr<object> grouping;
  REQUIRE_NOTHROW( grouping = std::shared_ptr<object>( new mpi_object( blocked ) ) );

  mpi_distribution *twoblocked;
  REQUIRE_NOTHROW( twoblocked = new mpi_block_distribution(decomp,dim,-1,globalsize) );
  std::shared_ptr<object> coordinates;
  REQUIRE_NOTHROW( coordinates = std::shared_ptr<object>( new mpi_object( twoblocked ) ) );
  CHECK( coordinates->get_orthogonal_dimension()==dim );

  // masked coordinates are ncluster times orthogonal version of block
  // but we need an extra location to indicate inclusion
  // so that we can implement a masked MPI reduce.....
  mpi_distribution *kdblocked;
  REQUIRE_NOTHROW
    ( kdblocked = new mpi_block_distribution(decomp,ncluster*(dim+1),-1,globalsize) );
  std::shared_ptr<object> masked_coordinates;
  REQUIRE_NOTHROW( masked_coordinates = std::shared_ptr<object>( new mpi_object( kdblocked ) ) );

  domain_coordinate myfirst;
  REQUIRE_NOTHROW( myfirst = blocked->get_processor_structure(mycoord)->first_index_r() );
  CHECK( myfirst.get_dimensionality()==1 );

  /*
   * set masked_data as it was at the end of [40]
   */
  double *masked_data;
  REQUIRE_NOTHROW( masked_data = masked_coordinates->get_data(mycoord) );
  CHECK( masked_coordinates->volume(mycoord)==localsize );
  CHECK( masked_coordinates->get_orthogonal_dimension()==ncluster*(dim+1) );

  double *group_data;
  REQUIRE_NOTHROW( group_data = grouping->get_data(mycoord) );

  double *coordinate_data;
  REQUIRE_NOTHROW( coordinate_data = coordinates->get_data(mycoord) );

  for (index_int i=0; i<localsize; i++) {
    double *masked_coordinate = masked_data+i*ncluster*(dim+1);

    // what group does this point belong to?
    int group = (int)( group_data[i] );
    INFO( fmt::format("local point {} is in group {}",i,group) );

    // compare the point to all clusters
    for (int ic=0; ic<ncluster; ic++) {
      INFO( fmt::format("cluster {} out of {}",ic,ncluster) );
      // go through all the dimensions of a point
      if (ic==group)
	masked_coordinate[ ic*(dim+1) ] = +1;
      else
	masked_coordinate[ ic*(dim+1) ] = -1;
      for (int id=0; id<dim; id++) {
	masked_coordinate[ id+1 ] = coordinate_data[ i*dim+id ];
      }
    }
  }

  /*
   * Reduce based on mask
   */
  mpi_distribution 
    *kreplicated = new mpi_replicated_distribution(decomp,dim,ncluster);
  auto new_centers = std::shared_ptr<object>( new mpi_object( kreplicated ) );
  kernel *reduce_with_mask = new mpi_kernel(masked_coordinates,new_centers);
  REQUIRE_NOTHROW( reduce_with_mask->set_explicit_beta_distribution(masked_coordinates.get()) );
  REQUIRE_NOTHROW( reduce_with_mask->set_localexecutefn(&masked_reduction_1d) );
  REQUIRE_NOTHROW( reduce_with_mask->analyze_dependencies() );
  REQUIRE_NOTHROW( reduce_with_mask->execute() );

}
