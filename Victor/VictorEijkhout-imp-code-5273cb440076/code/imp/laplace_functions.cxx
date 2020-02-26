/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** laplace_functions.cxx : implementations of the laplace support functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "laplace_functions.h"

void laplace_bilinear_fn( kernel_function_args ) {
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
    out_gfirst = out_gstruct->first_index_r(), out_glast = out_gstruct->last_index_r(),
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (dim==2) {
  //snippet laplacebil
    for (index_int i=pfirst[0]; i<=plast[0]; i++) {
      bool skip_first_i = i==out_gfirst[0], skip_last_i = i==out_glast[0];
      for (index_int j=pfirst[1]; j<=plast[1]; j++) {
        bool skip_first_j = j==out_gfirst[1], skip_last_j = j==out_glast[1];
        index_int IJ = INDEX2D(i,j,out_offsets,out_nsize);
	if (!skip_first_i && !skip_first_j && !skip_last_i && !skip_last_j)
	  // fmt::print("{} index [{},{}] @{} from {}--{}\nusing {},{},{} {},{},{} {},{},{}\n",
	  // 	     p->as_string(),
	  // 	     i,j,IJ,
	  // 	     INDEX2D(i-1,j-1,in_offsets,in_nsize),
	  // 	     INDEX2D(i+1,j+1,in_offsets,in_nsize),

	  // 	     indata[ INDEX2D(i-1,j-1,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i-1,j,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i-1,j+1,in_offsets,in_nsize) ],

	  // 	     indata[ INDEX2D(i,j-1,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i,j,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i,j+1,in_offsets,in_nsize) ],

	  // 	     indata[ INDEX2D(i+1,j-1,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i+1,j,in_offsets,in_nsize) ],
	  // 	     indata[ INDEX2D(i+1,j+1,in_offsets,in_nsize) ]
	  // 	     );
        outdata[IJ] =
        outdata[ INDEX2D(i,j,out_offsets,out_nsize) ] = 
	  // center
	  8 *
	  indata[ INDEX2D(i,j,in_offsets,in_nsize) ]
	  +
	  // up
	  -1 *
          ( !skip_first_i 
            ? indata[ INDEX2D(i-1,j,in_offsets,in_nsize) ] : 0 )
	  +
	  // dn
	  -1 *
          ( !skip_last_i 
            ? indata[ INDEX2D(i+1,j,in_offsets,in_nsize) ] : 0 )
	  +
	  // lt
	  -1 *
          ( !skip_first_j
            ? indata[ INDEX2D(i,j-1,in_offsets,in_nsize) ] : 0 )
	  +
	  // rt
	  -1 *
          ( !skip_last_j
            ? indata[ INDEX2D(i,j+1,in_offsets,in_nsize) ] : 0 )
	  +
	  // lu
	  -1 *
          ( !skip_first_i && !skip_first_j
            ? indata[ INDEX2D(i-1,j-1,in_offsets,in_nsize) ] : 0 )
          +
	  // ru
	  -1 *
          ( !skip_first_i && !skip_last_j
            ? indata[ INDEX2D(i-1,j+1,in_offsets,in_nsize) ] : 0 )
          +
	  // ld
	  -1 *
          ( !skip_last_i && !skip_first_j
            ? indata[ INDEX2D(i+1,j-1,in_offsets,in_nsize) ] : 0 )
          +
	  // rd
	  -1 *
          ( !skip_last_i && !skip_last_j
            ? indata[ INDEX2D(i+1,j+1,in_offsets,in_nsize) ] : 0 )
          ;
      }
    }
  //snippet end
  } else {
    throw(std::string("no lulesh functions except 2d"));
  }

  *flopcount += 4*pstruct->volume();
}
