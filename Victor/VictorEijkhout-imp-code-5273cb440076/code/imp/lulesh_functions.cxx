/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** lulesh_functions.cxx : implementations of the lulesh support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "lulesh_functions.h"

/****
 **** Elements to local node
 ****/

//snippet lulesheverydivby2
//! Divide each component by two
domain_coordinate *signature_coordinate_element_to_local( domain_coordinate *i ) {
  return i->operate_p( ioperator("/2") );
};
std::shared_ptr<multi_indexstruct> signature_struct_element_to_local( std::shared_ptr<multi_indexstruct> i ) {
  if (!i->is_contiguous())
    throw(std::string("signature_struct_element_to_local only for contiguous"));
  auto divop = ioperator("/2");
  return std::shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      (i->first_index_r().operate(divop),i->last_index_r().operate(divop)) );
};
//snippet end

/*!
  Element to local node mapping: Take every input element, and replicate it a number of times.
 */
void element_to_local_function( kernel_function_args,int dim )
{
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p),
    qstruct = invector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_enclosing_structure(),
    out_gstruct = outvector->get_enclosing_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();
  if (dim==1) {
    for (index_int i=qfirst[0]; i<=qlast[0]; i++)
      for (index_int ii=2*i; ii<2*i+2; ii++) {
        index_int
          I = INDEX1D(i,in_offsets,in_nsize), II = INDEX1D(ii,out_offsets,out_nsize);
        outdata[II] = indata[I];
      }
  } else if (dim==2) {
  //snippet bcaste2ln
    for (index_int i=qfirst[0]; i<=qlast[0]; i++)
      for (index_int ii=2*i; ii<2*i+2; ii++)
        for (index_int j=qfirst[1]; j<=qlast[1]; j++)
          for (index_int jj=2*j; jj<2*j+2; jj++) {
            index_int IJ = INDEX2D(i,j,in_offsets,in_nsize),
              IIJJ = INDEX2D(ii,jj,out_offsets,out_nsize);
            outdata[IIJJ] = indata[IJ];
          }
  //snippet end
  }
  // fmt::print("Copy struct={} from {} to {}\nnuma={} global={} into\nnuma={} global={}\n\n",
  //              pstruct->as_string(),
  //              invector->get_name(),outvector->get_name(),
  //              in_nstruct->as_string(),in_gstruct->as_string(),
  //              out_nstruct->as_string(),out_gstruct->as_string());
  
  *flopcount += pstruct->volume();
}

/****
 **** Local nodes to global
 ****/

//! Compute local node numbers from global, multi-d case.
//snippet luleshng2nl
std::shared_ptr<multi_indexstruct> signature_local_from_global
    ( std::shared_ptr<multi_indexstruct> g,std::shared_ptr<multi_indexstruct> enc ) {
  int dim = g->get_same_dimensionality(enc->get_dimensionality());
  domain_coordinate_allones allones(dim);
  auto range = std::shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      ( g->first_index_r()*2-allones,g->last_index_r()*2 ) );
  return range->intersect(enc);
};
//snippet end

void local_to_global_function
    ( kernel_function_args,std::shared_ptr<multi_indexstruct> local_nodes_domain)
{
  auto invector = invectors.at(0);
  double
    *indata = invector->get_data(p), *outdata = outvector->get_data(p);
  int dim = outvector->get_same_dimensionality( invector->get_dimensionality() );

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->location_of_first_index(*invector.get(),p),
    outlen = outvector->volume(p),
    inlen = invector->volume(p);

  int ortho = 1;
  if (0) {
  } else if (dim==2) {
    auto pstruct = outvector->get_processor_structure(p);
    domain_coordinate
      pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r();

    auto in_nstruct = invector->get_numa_structure(),
      out_nstruct = outvector->get_numa_structure(),
      in_gstruct = invector->get_enclosing_structure(),
      out_gstruct = outvector->get_enclosing_structure();
    domain_coordinate
      out_gfirst = out_gstruct->first_index_r(), out_glast = out_gstruct->last_index_r(),
      in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
      in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
      out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

    //snippet l2gfunction
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      bool skip_first_i = i==out_gfirst[0], skip_last_i = i==out_glast[0];
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
        bool skip_first_j = j==out_gfirst[1], skip_last_j = j==out_glast[1];
        outdata[ INDEX2D(i,j,out_offsets,out_nsize) ] = 
          ( !skip_first_i && !skip_first_j
            ? indata[ INDEX2D(2*i-1,2*j-1,in_offsets,in_nsize) ] : 0 )
          +
          ( !skip_first_i && !skip_last_j
            ? indata[ INDEX2D(2*i-1,2*j,in_offsets,in_nsize) ] : 0 )
          +
          ( !skip_last_i && !skip_first_j
            ? indata[ INDEX2D(2*i,2*j-1,in_offsets,in_nsize) ] : 0 )
          +
          ( !skip_last_i && !skip_last_j
            ? indata[ INDEX2D(2*i,2*j,in_offsets,in_nsize) ] : 0 )
          ;
      }
    }
    //snippet end
  } else if (dim==1) {
    index_int itar = tar0, isrc = src0, global_last = outvector->global_volume();
    for (index_int g=outvector->first_index_r(p).coord(0);
         g<=outvector->last_index_r(p).coord(0); g++) {
      index_int e = g/2; int m = g%2;
      if (g>=2 && g<global_last-1) {
        if (m==0) {
          outdata[itar++] = indata[isrc] + indata[isrc+2]; isrc++;
        } else {
          outdata[itar++] = indata[isrc] + indata[isrc+2]; isrc += 3;
           }
      } else
        outdata[itar++] = indata[isrc++];
    }
  } else
    throw(fmt::format("Can not sum_mod2 for dim={}",dim));

  *flopcount += outlen;
}

/****
 **** Global node back to local
 ****/

//snippet lulesh_global_node_to_local
std::shared_ptr<multi_indexstruct> signature_global_node_to_local
    ( std::shared_ptr<multi_indexstruct> l ) {
  return l->operate( ioperator(">>1") )->operate( ioperator("/2") );
};
//snippet end

/*!
  Duplicate every global node over two local nodes.
  \todo the left/right tests should be against first/last coordinate, not gsizes
*/
void function_global_node_to_local
    ( kernel_function_args,std::shared_ptr<multi_indexstruct> local_nodes_domain)
{
  int dim = p.get_same_dimensionality(outvector->get_dimensionality());
  auto global_nodes = invectors.at(0), local_nodes = outvector;
  double
    *local_nodes_data = local_nodes->get_data(p),
    *global_nodes_data = global_nodes->get_data(p);

  // description of the indices on which we work
  auto local_nodes_struct = local_nodes->get_processor_structure(p),
    global_nodes_struct = global_nodes->get_processor_structure(p);
  domain_coordinate
    pfirst = local_nodes_struct->first_index_r(), plast = local_nodes_struct->last_index_r(),
    qfirst = global_nodes_struct->first_index_r(), qlast = global_nodes_struct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = global_nodes->get_numa_structure(),
    out_nstruct = local_nodes->get_numa_structure(),
    in_gstruct = global_nodes->get_enclosing_structure(),
    out_gstruct = local_nodes->get_enclosing_structure();
  domain_coordinate
    global_nodes_sizes = global_nodes->get_enclosing_structure()->local_size_r(),
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (dim==2) {
    //snippet function_global_to_local
    for (index_int i=qfirst[0]; i<=qlast[0]; i++) {
      for (index_int j=qfirst[1]; j<=qlast[1]; j++) {
        bool
          left_i = i==0, right_i = i==global_nodes_sizes[0],
          left_j = j==0, right_j = j==global_nodes_sizes[1];
        index_int Iin = INDEX2D(i,j,in_offsets,in_nsize);
        double g = global_nodes_data[Iin];// [ i*global_nodes_sizes[0]+j ];
        if (!left_i && !left_j) {
          index_int Iout = INDEX2D( 2*i-1,2*j-1, out_offsets,out_nsize );
          local_nodes_data[Iout] = g; }
        if (!left_i && !left_j) {
          index_int Iout = INDEX2D( 2*i,  2*j-1, out_offsets,out_nsize );
          local_nodes_data[Iout] = g; }
        if (!left_i && !left_j) {
          index_int Iout = INDEX2D( 2*i-1,2*j,   out_offsets,out_nsize );
          local_nodes_data[Iout] = g; }
        if (!left_i && !left_j) {
          index_int Iout = INDEX2D( 2*i,  2*j,   out_offsets,out_nsize );
          local_nodes_data[Iout] = g; }
      }
    }
    //snippet end
  } else
    throw(std::string("Function function_global_node_to_local only for d=2"));
}

/****
 **** And finally local nodes back to elements
 ****/

//snippet luleshsigl2e
std::shared_ptr<multi_indexstruct> signature_local_to_element
    ( int dim,std::shared_ptr<multi_indexstruct> i ) {
  domain_coordinate_allones allones(dim);
  auto times2 = ioperator("*2");
  return std::shared_ptr<multi_indexstruct>
    ( new contiguous_multi_indexstruct
      ( i->first_index_r()*2,i->last_index_r()*2+allones ) );
}
//snippet end

/*!
  Element to local node mapping: Take every input element, and replicate it a number of times.
 */
void local_node_to_element_function( kernel_function_args )
{
  auto invector = invectors.at(0);
  int dim = outvector->get_same_dimensionality(invector->get_dimensionality());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  // description of the indices on which we work
  auto pstruct = outvector->get_processor_structure(p),
    qstruct = invector->get_processor_structure(p);
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    qfirst = qstruct->first_index_r(), qlast = qstruct->last_index_r();

  // placement in the global data structures
  auto in_nstruct = invector->get_numa_structure(),
    out_nstruct = outvector->get_numa_structure(),
    in_gstruct = invector->get_enclosing_structure(),
    out_gstruct = outvector->get_enclosing_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();
  // fmt::print("{} node to element offsets {} in numa {}\n",
  //              p.as_string(),in_offsets.as_string(),in_nstruct->as_string());

  if (dim==2) {
  //snippet lugatherl2e
    for (index_int i=pfirst[0]; i<=plast[0]; i++)
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
        index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
        outdata[IJ] =
          ( indata[INDEX2D(2*i,  2*j,   in_offsets,in_nsize)] +
            indata[INDEX2D(2*i,  2*j+1, in_offsets,in_nsize)] +
            indata[INDEX2D(2*i+1,2*j,   in_offsets,in_nsize)] +
            indata[INDEX2D(2*i+1,2*j+1, in_offsets,in_nsize)]
            ) / 4.;                      
      }
  //snippet end
  } else {
    throw(std::string("no lulesh functions except 2d"));
  }

  *flopcount += 4*pstruct->volume();
}

