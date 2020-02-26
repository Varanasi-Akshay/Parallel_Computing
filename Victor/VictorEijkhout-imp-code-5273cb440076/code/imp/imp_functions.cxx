/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_functions.cxx : implementations of the support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "imp_functions.h"

/*!
  A no-op function. This can be used as the function of an origin kernel: whatever data is
  in the output vector of the origin kernel will be used as-is.
*/
void vecnoset( kernel_function_args )
{
  return;
}

/*!
 * A plain copy. This is often used to copy the beta distribution into the gamma.
 *
 * Subtlety: we use the output index only, so we can actually cover the case
 * where the distribution does not quite allow for copying, as in transposition.
 */
void veccopy( kernel_function_args )
{
  int
    dim = p.get_same_dimensionality(outvector->get_dimensionality()),
    k = outvector->get_orthogonal_dimension();
  auto invector = invectors.at(0);
  double
    *outdata = outvector->get_data(p), *indata = invector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  //snippet numaoffsets
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    //in_nstruct->first_index_r() - in_gstruct->first_index_r(),
  //out_nstruct->first_index_r() - out_gstruct->first_index_r();
  //snippet end

  // fmt::print
  //   ("Copy on {}: struct={} from {} to {}\nnuma={} global={} into\nnuma={} global={}\n\n",
  //    p.as_string(),pstruct->as_string(),
  //    invector->get_name(),outvector->get_name(),
  //    in_nstruct->as_string(),in_gstruct->as_string(),
  //    out_nstruct->as_string(),out_gstruct->as_string());
  if (dim==1) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	for (int ik=0; ik<k; ik++)
	  outdata[k*I+ik] = indata[k*I+ik];
      }
    } else {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	index_int I = INDEX1D(i,out_offsets,out_nsize);
	//fmt::print("[{}] copy global index {}@{}:{}\n",p->as_string(),i,I,indata[I]);
	outdata[I] = indata[I];
      }
    }
  } else if (dim==2) {
    if (k>1) {
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  for (int ik=0; ik<k; ik++)
	    outdata[k*IJ+ik] = indata[k*IJ+ik];
	}
      }
    } else {
      int done=0;
      //snippet copyloop2d
      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	  index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	  if (!done) {
	    fmt::print("{} copy: {}\n",p.as_string(),indata[IJ]); done = 1; }
	  outdata[IJ] = indata[IJ];
	}
      }
      //snippet end
    }
  } else if (dim==3) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  index_int IJK = INDEX3D(i,j,k,out_offsets,out_nsize);
	  outdata[IJK] = indata[IJK];
	}
      }
    }
  } else
    throw(fmt::format("veccopy not implemented for d={}",dim));

  *flopcount += outvector->volume(p);
}

/*!
  Delta function
 */
void vecdelta( kernel_function_args, domain_coordinate& delta)
{
  int dim = outvector->get_dimensionality();

  double
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  std::shared_ptr<multi_indexstruct> out_nstruct,out_gstruct;
  try {
    out_nstruct = outvector->get_numa_structure();
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> getting numa structure in vecdelta",c)); }
  try {
    out_gstruct = outvector->get_global_structure();
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> getting global structure of <<{}>> in vecdelta",
		      c,outvector->get_name())); }
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_offsets = outvector->offset_vector();

  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      outdata[I] = 0.;
    }
    if (outvector->get_processor_structure(p)->contains_element(delta))
      outdata[ INDEX1D(delta[0],out_offsets,out_nsize) ] = 1.;
  } else
    throw(fmt::format("Can not set delta for dim={}",dim));
}

/*!
  Set vector elements linearly. A useful test case.
 */
void vecsetlinear( kernel_function_args )
{
  int dim = outvector->get_dimensionality();
  double
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // various structures
  auto out_nstruct = outvector->get_numa_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    out_nsize = out_nstruct->local_size_r(),
    out_gsize = out_gstruct->local_size_r(),
    out_offsets = outvector->offset_vector();
  //out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,out_offsets,out_nsize);
      outdata[I] = (double)COORD1D(i,out_gsize);
    }
  } else if (dim==2) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	outdata[IJ] = COORD2D(i,j,out_gsize);
      }
    }
  } else if (dim==3) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  index_int IJK = INDEX3D(i,j,k,out_offsets,out_nsize);
	  outdata[IJK] = COORD3D(i,j,k,out_gsize);
	}
      }
    }
  } else
    throw(fmt::format("vecsetlinear not implemented for d={}",dim));

  *flopcount += 0.;
}

/*!
  Set vector elements linearly.
 */
void vecsetlinear2d( kernel_function_args )
{
  int dim = p.get_same_dimensionality( outvector->get_dimensionality() );
  double
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    len = outvector->volume(p);
  domain_coordinate
    gfirst = outvector->first_index_r(p),
    lens = outvector->local_size_r(p),
    gsizes = outvector->global_size();

  auto bigstruct = outvector->get_global_structure();
  const domain_coordinate first2 = outvector->first_index_r(p); // weird: the ampersand is essential
  index_int first_index = first2.linear_location_in( bigstruct );

  if (dim==2) {
    for (int i=0; i<lens[0]; i++) {
      index_int gi = gfirst[0]+i;
      for (int j=0; j<lens[1]; j++) {
	index_int gj = gfirst[1]+j;
	outdata[i*lens[0]+j] = gi*gsizes[0] + gj;
      }
    }
  } else
    throw(std::string("dimensionality should be 2 for setlinear2d"));

  *flopcount += 0.;
}

/*!
  Set vector elements linearly. A useful test case.
 */
void vecsetconstant( kernel_function_args, double value )
{
  auto outdist = outvector->get_distribution();
  double
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outdist,p),
    len = outvector->volume(p);

  index_int first_index = outvector->first_index_r(p).coord(0);

  for (index_int i=0; i<len; i++) {
    outdata[tar0+i] = value;
  }  

  *flopcount += 0.;
}

void vecsetconstantzero( kernel_function_args ) {
  vecsetconstant(kernel_function_call,0.); 
};

void vecsetconstantone( kernel_function_args ) {
  vecsetconstant(kernel_function_call,1.); 
};

/*!
  Set vector elements linearly. A useful test case.
*/
void vecsetconstantp( kernel_function_args ) {
  int dim;
  try {
    int dim0 = outvector->get_dimensionality();
    dim = p.get_same_dimensionality(dim0);
  } catch (std::string c) {
    throw(fmt::format("Error <<c>> checking dim coordinate <<{}>> against <<{}>>",
		      c,p.as_string(),outvector->as_string()));
  }
  double value;
  try {
    int ivalue = outvector->get_decomposition()->linearize(p);
    value = (double)ivalue;
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> in converting coordinate <<{}>>",c,p.as_string()));
  }
  fmt::print("Set constantp on {} to {}\n",p.as_string(),value);

#include "impfunc_struct_index.cxx"

  if (dim==1) {
    fmt::print("[{}] writes pstruct={}: {}--{}\n",
	       p.as_string(),pstruct->as_string(),pfirst.as_string(),plast.as_string());
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int loc = INDEX1D(i,offsets,nsize);
      fmt::print("[{}] write {} @{}->{} in {}\n",p.as_string(),value,i,loc,(long int)outdata);
      outdata[ loc ] = value;
    }
  } else if (dim==2) {
    index_int
      ioffset = offsets[0], joffset = offsets[1]; 
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	outdata[ INDEX2D(i,j,offsets,nsize) ] = value;
      }
    }
  } else if (dim==3) {
    index_int
      ioffset = offsets[0], joffset = offsets[1], koffset = offsets[2];
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
	for (index_int k=pfirst[2]; k<=plast[2]; k++) {
	  outdata[ INDEX3D(i,j,k,offsets,nsize) ] = value;
	}
      }
    }
  } else
    throw(fmt::format("vecsetconstantp not implemented for d={}",dim));

  *flopcount += 0.;
}

/*! 
  Deceptively descriptive name for a very ad-hoc function:
  set \$! y_i = p+.5\$!
*/
void vector_gen(kernel_function_args ) {
  int dim;
  try {
    int dim0 = outvector->get_dimensionality();
    dim = p.get_same_dimensionality(dim0);
  } catch (std::string c) {
    throw(fmt::format("Error <<c>> checking dim coordinate <<{}>> against <<{}>>",
		      c,p.as_string(),outvector->as_string()));
  }
  double value;
  try {
    int ivalue = outvector->get_decomposition()->linearize(p);
    value = (double)ivalue;
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> in converting coordinate <<{}>>",c,p.as_string()));
  }
  fmt::print("Set constantp on {} to {}\n",p.as_string(),value);

  int k = outvector->get_orthogonal_dimension();
  if (k>1) throw(std::string("No ortho supported"));

#include "impfunc_struct_index.cxx"

  int plinear = p.linearize(outvector.get());

  if (dim==1) {
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int I = INDEX1D(i,offsets,nsize);
      outdata[I] = plinear+.5;
    }
  } else
    throw(fmt::format("vector_gen not implemented for d={}",dim));
  *flopcount += plast[0]-pfirst[0]+1;
}

/*
 * Shift functions
 */

//snippet omprightshiftbump
/*!
  Shift an array to the right without wrap connections.
  We leave the global first position undefined.
*/
void vecshiftrightbump( kernel_function_args ) {
  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    //in_nstruct->first_index_r() - in_gstruct->first_index_r(),
  //out_nstruct->first_index_r() - out_gstruct->first_index_r();

  index_int pfirst0 = pfirst[0];
  if (pfirst0==0) pfirst0++;
  for (index_int i=pfirst0; i<=plast[0]; i++) {
    index_int Iout = INDEX1D(i,out_offsets,out_nsize), Iin = INDEX1D(i,in_offsets,in_nsize);
    outdata[Iout] = indata[Iin-1];
  }
  index_int len = plast[0]-pfirst0;
  *flopcount += len;
}
//snippet end

//snippet leftshiftbump
/*!
  Shift an array to the left without wrap connections.
  We leave the global last position undefined.
*/
void vecshiftleftbump( kernel_function_args ) {
  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();

  index_int pfirst0 = pfirst[0], plast0 = plast[0];
  if (plast0==in_gstruct->last_index_r()[0]) plast0--;
  fmt::print("p={} copies {}-{}\n",p.as_string(),pfirst0,plast0);
  for (index_int i=pfirst0; i<=plast0; i++) {
    index_int Iout = INDEX1D(i,out_offsets,out_nsize), Iin = INDEX1D(i,in_offsets,in_nsize);
    outdata[Iout] = indata[Iin+1];
  }
  index_int len = plast0-pfirst0+1;
  *flopcount += len;
}
//snippet end

/*
 * Nbody stuff
 */

/*!
  Compute the center of mass of an array of particles by comparing two-and-two

  - k=1: add charges
  - k=2: 0=charges added, 1=new center
 */
void scansumk( kernel_function_args,int k ) {
  int
    dim = p.get_same_dimensionality(outvector->get_dimensionality());
  auto invector = invectors.at(0);
  double *indata = invector->get_data(p);
  int insize = invector->volume(p);

  double *outdata = outvector->get_data(p);
  int outsize = outvector->volume(p);

  if (k<0 || k>2)
    throw(fmt::format("scansumk k={} meaningless",k));

  if (2*outsize!=insize)
    throw(fmt::format("scansum: in/out not compatible: {} {}\n",insize,outsize));

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p),
    qstruct = invector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();

  if (dim==1) {
    // fmt::print("[{}] summing {}-{} into {}-{}\n",
    // 	       p.as_string(),qfirst[0],qlast[0],pfirst[0],plast[0]);
    index_int Iin = INDEX1Dk(qfirst[0],in_offsets,in_nsize,k);
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      index_int Iout = INDEX1Dk(i,out_offsets,out_nsize,k);
      double v1 = indata[Iin], v2 = indata[Iin+k], v3=v1+v2;
      outdata[Iout] = v3;
      if (k==2) {
	double x1 = indata[Iin+1], x2 = indata[Iin+3];
      	outdata[Iout+1] = sqrt( v3*x1*x1*x2*x2 / (v1*x2*x2+v2*x1*x1) );
      }
      Iin += 2*k;
    }
  } else
    throw(std::string("scansumk only for d=1"));

  *flopcount += k*outsize;
}

//! Short-cut of \ref scansumk for k=1
void scansum( kernel_function_args ) {
  scansumk(step,p,invectors,outvector,flopcount,1);
}

/*!
  Sum an array into a scalar

  \todo do some sanity check on the size of the output
*/
void summing( kernel_function_args ) {
  auto invector = invectors.at(0);
  int
    dim = p.get_same_dimensionality(invector->get_dimensionality()),
    k = invector->get_orthogonal_dimension();

  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto qstruct = invector->get_processor_structure(p);
  domain_coordinate
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    // in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    // out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  double s = 0.; index_int len=0;
  if (dim==1) {
    if (k>1) {
      for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
	index_int I = INDEX1D(i,in_offsets,in_nsize);
	for (int ik=0; ik<k; ik++)
	  s += indata[k*I+ik];
      }
    } else {
      for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
	index_int I = INDEX1D(i,in_offsets,in_nsize);
	s += indata[I];
      }
    }
    len = qlast[0]-qfirst[0]+1;
  } else
    throw(fmt::format("veccopy not implemented for d={}",dim));

  *outdata = s;

  *flopcount += len*k;

#if 0
  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = invector->volume(p);

  double s = 0.;
  int ortho = 1;
  //printf("[%d] summing %d element, starting %e",p.coord(0),len,indata[src0]);

  for (index_int i=0; i<len; i++) {
    //printf("[%d] i=%d idx=%d data=%e\n",p,i,src0+i,indata[src0+i]);
    //fmt::print("[{}] i={}, idx={}, data={}\n",p.as_string(),i,src0+i,indata[src0+i]);
    s += indata[src0+i];
  }
  //printf(", giving %e\n",s);
#endif
}

/*!
  Sum an array into a scalar and take the root. This is for norms

  \todo do some sanity check on the size of the output
*/
void rootofsumming( kernel_function_args ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = invector->volume(p);

  double s = 0.;
  int ortho = 1;

  for (index_int i=0; i<len; i++)
    s += indata[src0+i];

  *outdata = sqrt(s);

  *flopcount += len;
}

//snippet ompnormsquared
//! Compute the local part of the norm of a vector squared.
void local_normsquared( kernel_function_args ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = invector->volume(p);

  double s = 0;
  //fmt::MemoryWriter w;
  for (index_int i=0; i<len; i++) {
    //w.write("{} ",indata[src0+i]);
    s += indata[src0+i]*indata[src0+i];
  }
  // fmt::print("norm squared of {}: {} comes to {}\n",
  // 	     invector->get_name(),w.str(),s);
  outdata[tar0] = s;

  *flopcount += len;
}
//snippet end

//! Compute the norm of the local part of a vector.
void local_norm( kernel_function_args ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = invector->volume(p);

  double s = 0;
  for (index_int i=0; i<len; i++) {
    s += indata[src0+i]*indata[src0+i];
  }
  outdata[tar0] = sqrt(s);

  *flopcount += len;
}

//! The local part of the inner product of two vectors.
void local_inner_product( kernel_function_args ) {
  if (invectors.size()<2)
    throw(fmt::format("local inner product: #vectors={}, s/b 2",invectors.size()));
  auto invector = invectors.at(0), othervector = invectors.at(1);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *otherdistro = dynamic_cast<distribution*>(othervector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p), *otherdata = othervector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    oth0 = othervector->location_of_first_index(*othervector.get(),p),
    len = invector->volume(p);

  if (len!=otherdistro->volume(p))
    throw(fmt::format("Incompatible sizes {}:{} {}:{}\n",
	  invector->get_name(),len,othervector->get_name(),othervector->volume(p)));

  double s = 0;
  for (index_int i=0; i<len; i++) {
    s += indata[src0+i]*otherdata[oth0+i];
  }
  //printf("[%d] local normquared %e written to %d\n",p,s,outloc);
  outdata[tar0] = s;

  *flopcount += len;
}

/*!
  Pointwise square root
 */
void vectorroot( kernel_function_args ) {
  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);//outdistro->local_size(p);

  int ortho = 1;

  for (index_int i=0; i<len; i++) {
    // if (i==0)
    //   fmt::print("Vector root : sqrt of {}\n",indata[src0+i] );
    outdata[tar0+i] = sqrt( indata[src0+i] );
  }  

  *flopcount += len*ortho;
}

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in as an extra input object.
*/
void vecscaleby( kernel_function_args ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);

  double a;
  {
    auto inscalar = invectors.at(1);
    try {
      inscalar->require_type_replicated();
      if (!inscalar->local_size_r(p)[0]==1)
	throw(std::string("Inscalar object not single component"));
      a = inscalar->get_data(p)[0];
    } catch (std::string c) { fmt::print("Error <<{}>> getting inscalar value\n",c);
      throw(fmt::format("vecscaleby of <<{}>> by <<{}>> failed",
			invector->get_name(),inscalar->as_string()));
    }
    //printf("scale by object data: %e\n",a);
  }
  
  for (index_int i=0; i<len; i++) {
    outdata[tar0+i] = a*indata[src0+i];
  }
  *flopcount += len;
}

//! \todo replace this by vecscalebyc with lambda
void vecscalebytwo( kernel_function_args ) {
  vecscalebyc(step,p,invectors,outvector,flopcount,2.);
}

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in through the context as a (void*)(double*)&scalar
  \todo replace void ctx by actual scale
*/
//snippet ompscalevec
void vecscalebyc( kernel_function_args,double a ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    // in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    // out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  for (index_int i=pfirst[0]; i<=plast[0]; i++) {
    index_int Iout = INDEX1D(i,out_offsets,out_nsize), Iin = INDEX1D(i,in_offsets,in_nsize);
    // if (i==pfirst[0])
    //   fmt::print("{}: scale {}, value {} by {}, starting with {}, which is local index {}->{}\n",
    // 		 p.as_string(),invector->get_name(),indata[Iin],a,pfirst[0],Iin,Iout);
    outdata[Iout] = a*indata[Iin];
  }

  *flopcount += plast[0]-pfirst[0]+1;
}
//snippet end

/*!
  Multiply a vector by a scalar; this is a limited case of an AXPY.
  The scalar comes in through the context as a (void*)(double*)&scalar,
  or as an extra input object.
*/
void vecscaledownby( kernel_function_args ) {
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);//outdistro->local_size(p);

  double a,ainv;
  {
    auto inscalar = invectors.at(1);
    try {
      inscalar->require_type_replicated();
    } catch (std::string c) { fmt::print("Error <<{}>>\n",c); throw("vecscaledownby failed\n"); }
    if (!inscalar->local_size_r(p)[0]==1)
      throw("Inscalar object not single component\n");
    a = inscalar->get_data(p)[0];
  }
  //  else throw("could not find scalar for scaleby\n");
  ainv = 1./a;
  
  for (index_int i=0; i<len; i++) {
    outdata[tar0+i] = ainv*indata[src0+i];
  }
  *flopcount += len;
}

//! \todo change the void* to double* or even double
void vecscaledownbyc( kernel_function_args,double a ) {
  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    len = outvector->volume(p);//outdistro->local_size(p);

  double ainv = 1./a;
  
  for (index_int i=0; i<len; i++) {
    outdata[tar0+i] = ainv*indata[src0+i];
  }
  *flopcount += len;
}

/*!
  Scalar combination of two vectors into a third.
  The second vector and both scalars come in through the context
  as an doubledouble_object_struct structure.
\todo 
*/
void vecaxbyz( kernel_function_args,void *ctx ) {
  auto invector1 = invectors.at(0), invector2 = invectors.at(2);
  distribution
    *indistro1 = dynamic_cast<distribution*>(invector1.get()),
    *indistro2 = dynamic_cast<distribution*>(invector2.get());
  double *x1data = invector1->get_data(p), *x2data = invector2->get_data(p);
  charcharxyz_object_struct *ssx = (charcharxyz_object_struct*)ctx;
  double *outdata = outvector->get_data(p);
  // scalars
  double
    s1 = *( invectors.at(1)->get_data(p) ),
    s2 = *( invectors.at(3)->get_data(p) );
  if (ssx->c1=='-') s1 = -s1;
  if (ssx->c2=='-') s2 = -s2;
  
  //fmt::print("axbyz computing {} uses scalars {},{}\n",outvector->get_name(),s1,s2);
  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src1 = invector1->location_of_first_index(*invector1.get(),p),
    src2 = invector2->location_of_first_index(*invector2.get(),p),
    len = outvector->volume(p);//outvector->get_processor_structure(p)->volume();

  //fmt::MemoryWriter w;
  for (index_int i=0; i<len; i++) {
    //w.write("{}+{} ",x1data[src1+i],x2data[src2+i]);
    outdata[tar0+i] = s1*x1data[src1+i] + s2*x2data[src2+i];
  }
  // fmt::print("axbyz computing {}x{} + {}x{}->{} : {}\n",
  // 	     s1,invector1->get_name(),s2,invector2->get_name(),outvector->get_name(),w.str());
  *flopcount += 3*len;
}

/*
 * Central difference computation
 */
void central_difference( kernel_function_args ) {
  central_difference_damp(step,p,invectors,outvector,flopcount,1.);
}

/*
 * Central difference computation with a damping parameter
 */
//snippet centraldiff
void central_difference_damp( kernel_function_args,double damp)
{
  //snippet end
  int
    dim = p.get_same_dimensionality(outvector->get_dimensionality()),
    k = outvector->get_orthogonal_dimension();
  if (dim>1) throw(std::string("Central differences only 1d"));
  if (k>1) throw(std::string("Central differences no ortho"));

  auto invector = invectors.at(0);
  //snippet centraldiff
  double
    *outdata = outvector->get_data(p), *indata = invector->get_data(p);
  //snippet end
  
  // description of the indices on which we work
  //snippet centraldiff
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  //snippet end
  
  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    // in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    // out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  //snippet centraldiff
  index_int lo=pfirst[0],hi=plast[0];
  //snippet end
  if (lo==out_gstruct->first_index_r()[0]) { // dirichlet left boundary condition
    index_int i = lo;    
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata[Iout] = ( 2*indata[Iin] - indata[Iin+1] )*damp;
    *flopcount += 3;
    lo++;
  }
  if (hi==out_gstruct->last_index_r()[0]) {
    index_int i = hi;    
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata[Iout] = ( 2*indata[Iin] - indata[Iin-1] )*damp;
    *flopcount += 3;
    hi--;
  }

  // ... but then we have a regular three-point stencil
  //snippet centraldiff
  for (index_int i=lo; i<=hi; i++) {
    index_int
      Iin  = INDEX1D(i,in_offsets,in_nsize),
      Iout = INDEX1D(i,out_offsets,out_nsize);
    outdata[Iout] = ( 2*indata[Iin] - indata[Iin-1] - indata[Iin+1] )
                    *damp;
  }
  *flopcount += 4*(hi-lo+1);
  //snippet end
  
}

//! Recursive function for index calculation in any number of dimensions
index_int INDEXanyD(domain_coordinate &i,domain_coordinate &off,domain_coordinate &siz,int d) {
  if (d==1) {
    index_int
      id = i[0], od = off[0];
    //fmt::print("for d={}, i={}, off={}\n",d,id,od);
    return id-od;
  } else {
    index_int
      p = INDEXanyD(i,off,siz,d-1),
      sd = siz[d-1], id = i[d-1], od = off[d-1];
    //fmt::print("for d={}, prev={}, size={}, i={}, off={}\n",d,p,sd,id,od);
    return p*sd + id-od;
  }
};

//snippet centraldiffd
void central_difference_anyd( kernel_function_args ) {
  //snippet end
  int
    dim = p.get_same_dimensionality(outvector->get_dimensionality()),
    k = outvector->get_orthogonal_dimension();
  if (k>1) throw(std::string("Central differences no ortho"));

  auto invector = invectors.at(0);
  //snippet centraldiffd
  double
    *outdata = outvector->get_data(p), *indata = invector->get_data(p);
  //snippet end
  
  // description of the indices on which we work
  //snippet centraldiffd
  auto pstruct = outvector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();
  //snippet end
  
  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_global_structure(),
    out_gstruct = outvector->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = invector->offset_vector(),
    out_offsets = outvector->offset_vector();
    // in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    // out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  //snippet centraldiffd
  auto begin = pstruct->begin();
  auto end = pstruct->end();
  for ( auto ii=begin; !( ii==end ); ++ii) {
    domain_coordinate i = *ii;
    index_int
      Iin  = INDEXanyD(i,in_offsets,in_nsize,dim),
      Iout = INDEXanyD(i,out_offsets,out_nsize,dim);
    outdata[Iout] = 2*indata[Iin]; // - indata[Iin-1] - indata[Iin+1] );
  }
  *flopcount += 4*pstruct->volume();
  //snippet end
  
}

//! Recursive derivation of multigrid coarse levels
std::shared_ptr<indexstruct> halfinterval(index_int i) {
  return std::shared_ptr<indexstruct>{ new contiguous_indexstruct(i/2) };
}
//! Signature function for 1D multigrid
std::shared_ptr<indexstruct> doubleinterval(index_int i) {
  return std::shared_ptr<indexstruct>{ new contiguous_indexstruct(2*i,2*i+1) };
}

//! \todo change void* to string
void print_trace_message( kernel_function_args,void *ctx ) {
  auto inobj = invectors.at(0);
  std::string *c = (std::string*)(ctx);
  if (p.is_zero())
    fmt::print("{}: {}\n",*c,inobj->get_data(p)[0]);
};

