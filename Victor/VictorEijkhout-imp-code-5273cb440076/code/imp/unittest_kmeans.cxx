#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "catch.hpp"

void dummy_centers_1d( object *centers ) {
  mpi_object *mpi_centers = (mpi_object*)centers;
  mpi_distribution *distro = mpi_centers->get_distribution();
  int
    k = distro->get_orthogonal_dimension();
  double *center_data;
  try { center_data = mpi_centers->get_data();
  } catch (int x) {printf("could not get data in initial centers\n"); throw(1); }

  for (int ik=0; ik<k; ik++) {
    center_data[ik] = ( (double)ik+1 )/(k+1);
  }
}

void coordinates_on_center_1d
(int step,int p,object *inv,object *outv,void *ctx) {
  mpi_object *outvector = (mpi_object*)outv;
  mpi_distribution *outdistro = outvector->get_distribution();
  index_int first=outdistro->my_first_index(),last=outdistro->my_last_index();

  int k = outdistro->get_orthogonal_dimension();
  if (k!=1) {
    printf("This routine is 1d only, not %d\n",k); throw(1);}

  double *outdata;
  if (inv!=NULL) {
    printf("why does this have an input object?\n"); throw(2);}
  try { outdata = outvector->get_data();
  } catch (int x) {printf("could not get data for random coordinates\n"); throw(1); }

  for (index_int i=0; i<last-first; i++) {
    outdata[i] = ( (double)i+1 )/(last-first+1);
  }
};

TEST_CASE( "work with predictable centers","" ) {

  INFO( "mytid=" << mytid );

  for (int icase=0; icase<1; icase++) {
    int dim,k,globalsize;

    INFO( "case=" << icase );

    // Cases
    // 0: one dimensional, put the initial points on the centers
    switch (icase) {
    case 0 :
      k = problem_environment->nprocs(); dim = 1; globalsize = k;
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // centers are dim x k replicated reals
    mpi_distribution 
      *kreplicated = new mpi_replicated_scalar(problem_environment,dim*k);
    mpi_object
      *centers = new mpi_object( kreplicated );

    switch (icase) {
    case 0 :
      dummy_centers_1d( centers ); // not in the queue
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // coordinates are Nx2 with the N distributed
    mpi_distribution
      *twoblocked = new mpi_disjoint_blockdistribution(problem_environment,dim,-1,globalsize);
    mpi_object
      *coordinates = new mpi_object( twoblocked );
    mpi_kernel
      *set_random_coordinates = new mpi_kernel( coordinates );
    set_random_coordinates->set_name("set random coordinates");
    switch (icase) {
    case 0 :
      set_random_coordinates->localexecutefn = &coordinates_on_center_1d;
      break;
    default: printf("No case %d implemented\n",icase); throw(5);
    }

    // check that the local coordinates are in place
    REQUIRE_NOTHROW( set_random_coordinates->execute() );
    {
      double *centerdata = coordinates->get_data();
      switch (icase) {
      case 0 :
	CHECK( centerdata[0] == Approx( (double)(mytid+1)/(globalsize+1) ) );
	break;
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }

    // calculate Nxk distances, with the N distributed
    mpi_distribution
      *kblocked = new mpi_disjoint_blockdistribution(problem_environment,k,-1,globalsize);
    mpi_object
      *distances = new mpi_object( kblocked );
    mpi_kernel
      *calculate_distances = new mpi_kernel( coordinates,distances );
    calculate_distances->set_name("calculate distances");
    calculate_distances->localexecutefn = &distance_calculation;
    calculate_distances->localexecutectx = centers;
    // k=2 (coordinates) vs k=5 (distances)
    calculate_distances->set_explicit_beta_distribution( coordinates->get_distribution() );

    REQUIRE_NOTHROW( calculate_distances->execute() );

    {
      double *dist = distances->get_data();
      switch (icase) {
      case 0 :
	for (int ik=0; ik<k; ik++) {
	  INFO( "ik=" << ik << ", mytid=" << mytid );
	  if (ik==mytid) {
	    CHECK( dist[ik] == Approx( 0.0 ) );
	  } else {
	    CHECK( dist[ik] != Approx( 0.0 ) );
	  }
	}
	break;
      default: printf("No case %d implemented\n",icase); throw(5);
      }
    }

  }
};
