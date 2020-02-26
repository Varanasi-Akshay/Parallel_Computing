/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** kmeans_functions.cxx : implementations of the kmeans support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "kmeans_functions.h"

#include "utils.h"

/****
 **** Replicated initial centers
 ****/
void set_initial_centers( object *mpi_centers,processor_coordinate &p ) {
  int k = mpi_centers->get_orthogonal_dimension();
  double *center_data;
  try { center_data = mpi_centers->get_data(p);
  } catch (int x) {printf("could not get data in initial centers\n"); throw(1); }
  for (int ik=0; ik<k; ik++) {
    center_data[2*ik] = ( (double)ik )/k;
    center_data[2*ik+1] = ( (double)ik )/k;
  }
}

/****
 **** Kernel for initial data generation
 ****/
void generate_random_coordinates( kernel_function_args ) {
  index_int first=outvector->first_index_r(p)[0],last=outvector->last_index_r(p)[0];
  double *outdata;
  try { outdata = outvector->get_data(p);
  } catch (int x) {printf("could not get data for random coordinates\n"); throw(1); }

  // initialize random
  srand(p.coord(0));

  for (index_int i=0; i<last-first; i++) {
    outdata[2*i] = rand() / (double)RAND_MAX;
    outdata[2*i+1] = rand() / (double)RAND_MAX;
  }
};

/****
 **** Kernels for update step
 ****/
void distance_to_center( kernel_function_args,void *ctx ) {
  auto coordinates = invectors.at(0);

  two_object_struct *two_objects = (two_object_struct*) ctx;
  auto
    grouping = invectors.at(1), centers = invectors.at(2);
  //two_objects->one, *centers = two_objects->two;
  index_int
    dim = coordinates->get_orthogonal_dimension(),
    nclusters = centers->get_orthogonal_dimension(),
    my_cluster;
  double *ownership = grouping->get_data(p);
  my_cluster = (int)ownership[0];
  if (my_cluster<0 || my_cluster>=nclusters) 
    throw("Invalid cluster number found %d",my_cluster);
}

/*!
  The invector[0] is coordinates, ortho=#space dimensions.
  The invector[1] is clusters, local size=#clusters, ortho=#space dimen.
  The outvector has distances to all cluster centers: 
  - same local size as invector[0],
  - ortho = ncluster
 */
void distance_calculation( kernel_function_args ) {
  if (invectors.size()<2)
    throw(std::string("Missing centers inobject"));
  auto invector = invectors.at(0);
  int dimension = invector->get_orthogonal_dimension();
  double *coordinates;
  coordinates = invector->get_data(p); 
  int ncoordinates = invector->volume(p);
  if (ncoordinates!=outvector->volume(p))
    throw(fmt::format("Distance calculatin in/out need to be compatible"));

  double *distances;
  distances = outvector->get_data(p);
  // {
  //   index_int first=outvector->first_index(p).coord(0),last=outvector->last_index(p).coord(0),
  //     localsize = last-first+1;
  //   if (localsize!=ncoordinates)
  //     throw(fmt::format("distance localsize {} != coord local size {}",localsize,ncoordinates));
  // }

  auto centers = invectors.at(1);
  double *center_coordinates;
  center_coordinates = centers->get_data(p);
  index_int nclusters = centers->volume(p);
  if (nclusters!=outvector->get_orthogonal_dimension())
    throw(fmt::format("ncluster={} is not distance ortho={}",
		      nclusters,outvector->get_orthogonal_dimension()));
  
  for (int ipoint=0; ipoint<ncoordinates; ipoint++) {
    for (int ikluster=0; ikluster<nclusters; ikluster++) {
      double dist = 0;
      for (int id=0; id<dimension; id++) {
	double
	  pp = coordinates[ ipoint*dimension+id ],
	  cc = center_coordinates[ ikluster*dimension+id ],
	  dd = pp-cc;
	dist += dd*dd;
	// printf("[%d] dimension %d: point %d @ %e, cluster %d %e\n",
	//        p.coord(0),id, ipoint,pp, ikluster,cc);
      }
      dist = sqrt(dist);
      index_int array_loc = ipoint*nclusters+ikluster;
      distances[ array_loc ] = dist;
      // fmt::print
      // 	("{} point {} at {}, cluster {} at {}, distance computed={} at array loc {}\n",
      // 	 p.as_string(),ipoint,pp,ikluster,cc,dist,array_loc);
    }
  }

  return;
};

void group_calculation( kernel_function_args ) {
  auto invector = invectors.at(0);
  double *distances;
  try { distances = invector->get_data(p); 
  } catch (int x) {printf("Could not get distance coordinate data\n"); throw(1); }
  index_int ncluster = invector->get_orthogonal_dimension();

  double *groups = outvector->get_data(p);
  index_int
    first=outvector->first_index_r(p).coord(0),last=outvector->last_index_r(p).coord(0),
    localsize = last-first+1;
  
  for (int i=0; i<localsize; i++) {
    int kmin = 0; double mindist = distances[INDEXpointclusterdist(i,kmin,localsize,ncluster)];
    for (int ik=1; ik<ncluster; ik++) {
      double otherdist = distances[INDEXpointclusterdist(i,ik,localsize,ncluster)];
      //printf("at ik=%d compare %f / %f\n",ik,otherdist,mindist);
      if (otherdist<mindist) {
	kmin = ik; mindist = otherdist;
      }
    }
    groups[i] = (double)kmin;
  }

  return;
}

/*!
  Replicate the coordinates for each cluster, with a +/- 1 mask to indicate membership
  \todo pass in the number of clusters as parameter
 */
void coordinate_masking( kernel_function_args ) {
  // invec: [ coord object, grouping ], outvector: masked object
  auto invector = invectors.at(0),
    grouping = invectors.at(1); // groups: ncluster integers, replicated
  int dim = invector->get_orthogonal_dimension();
  double *coordinates, *group, *selected;
  try {
    coordinates = invector->get_data(p);
    group = grouping->get_data(p);
    selected = outvector->get_data(p);
  } catch (...) {
    throw(fmt::format("Could not get data on {}\n",p.as_string()));
  }

  //  int ncluster = grouping->get_orthogonal_dimension();

  index_int first=outvector->first_index_r(p).coord(0),
    last=outvector->last_index_r(p).coord(0),
    localsize = last-first+1;
  index_int kd = outvector->get_orthogonal_dimension();
  int ncluster = kd/(dim+1);

  fmt::MemoryWriter w;
  for (index_int ipoint=0; ipoint<localsize; ipoint++) {
    double
      *coordinate = coordinates + ipoint*dim,
      *masked = selected + ipoint*ncluster*(dim+1);
    int igroup = (int)(group[ipoint]); // what is the group of this coordinate?
    int outp = 0; // positioning pointer in the output
    w.write("{} local {} belongs to {}; ",p.as_string(),ipoint,igroup);
    for (int icluster=0; icluster<ncluster; icluster++) {
      // set mask location to +1 or -1
      w.write("cluster {} : ",icluster);
      if (icluster==igroup) {
	w.write("yes, ");
	masked[ outp++ ] = +1.;
      } else {
	w.write("no, ");
	masked[ outp++ ] = -1;
      }
      // copy coordinate data
      for (int idim=0; idim<dim; idim++)
	masked[ outp++ ] = coordinate[ idim ];
    }
    if (ipoint==0)
      fmt::print( "{}\n",w.str() ); 
  }
}

// compute locally the partial sums
void center_calculation_partial( kernel_function_args ) {
  auto invector = invectors.at(0);
  //  invec: masked_coordinates, outvector: partial_sums
  double *select_coordinates; // distributed 2k x N
  try { select_coordinates = invector->get_data(p); 
  } catch (int x) {printf("Could not get selected coordinate data\n");}
  index_int
    localsize = invector->volume(p),
    k2 = invector->get_orthogonal_dimension(); // #groups * dimension

  if (outvector->volume(p)!=1)
    throw(fmt::format("partial sums should be 1-distributed, not {}",
		      outvector->local_size_r(p).as_string()));
	  
  {
    index_int k2check = outvector->get_orthogonal_dimension();
    if (k2check!=k2)
      throw(fmt::format("outvector k s/b {}, not {}",k2,k2check));
  }

  double *centers_partial;
  try { centers_partial = outvector->get_data(p);
  } catch (int x) {printf("Could not get centers partial data\n");}
  index_int kcheck = outvector->get_orthogonal_dimension();
  if (kcheck!=k2)
    throw(fmt::format("partial sum ortho is {}, not {}",kcheck,k2));
  
  for (index_int ik=0; ik<k2; ik++) {
    double s=0;
    for (index_int i=0; i<localsize; i++)
      s += select_coordinates[ ik+i*k2 ];
    centers_partial[ik] = s;
  }

  return;
}

/****
 **** Local norm calculation
 ****/
void local_norm_function( kernel_function_args,void *ctx )
{
  auto invector = invectors.at(0);
  index_int
    first=invector->first_index_r(p).coord(0),
    last=invector->last_index_r(p).coord(0);
  double *outdata = outvector->get_data(p);
  double *indata = invector->get_data(p);

  kmeans_vec_sum(outdata,indata,first,last);

};

void kmeans_gen_local(double *outdata,index_int first,index_int last) {

  for (index_int i=first; i<last; i++) {
    outdata[i-first] = (double)i;
  }
}

void avg_local(double *indata,double *outdata,index_int first,index_int last,double *nops) {
  int leftshift=1;
  // initialization
  for (index_int i=0; i<last-first; i++)
    outdata[i] = 0.;
  // shift 0
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift;
    outdata[i_out] += indata[i_in];
  }
  // shift to the right
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift-1;
    outdata[i_out] += indata[i_in];
  }
  // shift to the left
  for (index_int i=first; i<last; i++) {
    index_int i_out=i-first,i_in=i_out+leftshift+1;
    outdata[i_out] += indata[i_in];
  }
  *nops = 3.*(last-first);
  return;
}

void kmeans_vec_sum(double *outdata,double *indata,index_int first,index_int last) {
  *outdata = 0;
  for (index_int i=first; i<last; i++) {
    index_int i_in = i-first;
    *outdata += indata[i_in];
  }
}

#if 0
struct coord_and_mask{ double coord[3]; int mask; };
MPI_Datatype masked_coordinate;

void add_if_mask_mixed( void *indata, void * outdata,int *dim,MPI_Datatype *type ) {
  struct coord_and_mask *incoord = (struct coord_and_mask*)indata;
  struct coord_and_mask *outcoord = (struct coord_and_mask*)outdata;
  if (incoord->mask) {
    if (outcoord->mask) {
      for (int id=0; id<3; id++) {
	outcoord->coord[id] += incoord->coord[id];
	outcoord->mask += 1;
      }
    } else {
      for (int id=0; id<3; id++) {
	outcoord->coord[id] = incoord->coord[id];
	outcoord->mask = 1;
      }
    }
  } // if the input is masked, we leave the inout alone.
}

void masked_reduct_mixed(void *data) {

  struct coord_and_mask point;
  int lengths[2]; lengths[0] = 3; lengths[1] = 1;
  MPI_Aint displs[2];
  displs[0] = (size_t)&(point.coord) - (size_t)&(point);
  displs[1] = (size_t)&(point.mask) - (size_t)&(point);
  MPI_Datatype types[2]; types[0] = MPI_DOUBLE; types[1] = MPI_INT;
  MPI_Type_create_struct(2,lengths,displs,types,&masked_coordinate);
  
  MPI_Op masked_add;
  MPI_Op_create(add_if_mask,1,&masked_add);
}

#endif
