/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** balance_functions.cxx : implementations of the load balancing support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "balance_functions.h"

//snippet transform_average
distribution *transform_by_average(distribution *unbalance,double *stats_data) {
  if (unbalance->get_dimensionality()!=1)
    throw(std::string("Can only average in 1D"));
  if (!unbalance->is_known_globally())
    throw(fmt::format
	  ("Can not transform-average <<{}>>: needs globally known",unbalance->get_name()));

  decomposition *decomp = dynamic_cast<decomposition*>(unbalance);
  if (decomp==nullptr)
    throw(std::string("Could not cast to decomposition"));

  auto astruct = new parallel_structure(decomp);
  int nprocs = decomp->domains_volume();
  for (int p=0; p<nprocs; p++) {
    auto me = unbalance->coordinate_from_linear(p);
    double
      cleft = 1./3, cmid = 1./3, cright = 1./3,
      work_left,work_right,work_mid = stats_data[p];
    index_int size_left=0,size_right=0,
      size_me = unbalance->volume(me);

    if (p==0) {
      size_left = 0; work_left = 0;
      cleft = 0; cmid = 1./2; cright = 1./2;
    } else {
      size_left = unbalance->volume( me-1 );
      work_left = stats_data[p-1];
    }
    
    

    if (p==nprocs-1) {
      size_right = 0; work_right = 0;
      cright = 0; cmid = 1./2; cleft = 1./2;    
    } else {
      size_right = unbalance->volume( me+1 );
      work_right = stats_data[p+1];
    }

    index_int new_size = ( cleft * work_left * size_left + cmid * work_mid * size_me
			   + cright * work_right *size_right ) / 3.;
    // fmt::print("{} New size: {},{},{} -> {}\n",
    // 	       me.as_string(),size_left,size_me,size_right,new_size);

    auto idx = std::shared_ptr<indexstruct>( new contiguous_indexstruct(1,new_size) );
    auto old_pstruct = unbalance->get_processor_structure(me);
    auto new_pstruct = std::shared_ptr<multi_indexstruct>
      ( new multi_indexstruct( std::vector<std::shared_ptr<indexstruct>>{ idx } ) );
    astruct->set_processor_structure(me,new_pstruct);
  }
  astruct->set_is_known_globally();
  return unbalance->new_distribution_from_structure(astruct);
}
//snippet end

void setmovingweight( kernel_function_args , int laststep ) {
  int
    dim = p.get_same_dimensionality(outvector->get_dimensionality()),
    k = outvector->get_orthogonal_dimension();
  if (k>1)
    throw(fmt::format("Moving weight not implemented for k>1: got {}",k));
  double
    *outdata = outvector->get_data(p);
  
  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  
  // placement in the global data structures
  auto out_nstruct = outvector->get_numa_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_offsets = outvector->offset_vector();
  auto
    out_gsize = outvector->global_volume();
  
  if (dim==1) {
    double center = step*out_gsize/(1.*laststep);
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      //fmt::print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata[I]);
      double
	imst = i-center,
	w = 1 + sqrt(out_gsize) * exp( -imst*imst );
      outdata[I] = w;
    }
  } else
    throw(fmt::format("Moving weight not implemented for d={}",dim));

}
