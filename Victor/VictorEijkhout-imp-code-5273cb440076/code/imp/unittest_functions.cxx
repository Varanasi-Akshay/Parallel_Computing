/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** unittest_functions.cxx : independent implementations of unittest functions
 ****
 ****************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "imp_base.h"
#include "unittest_functions.h"

/*!
  Three point averaging with bump connections:
  the halo has the same size as the alpha domain
  \todo this ignores non-zero first index and such.
*/ 
void threepointsumbump( kernel_function_args ) {
  if (outvector->get_dimensionality()>1)
    throw(std::string("threepointsumbump: only for 1-d"));

  auto invector = invectors.at(0);
  distribution
    *indistro = dynamic_cast<distribution*>(invector.get()),
    *outdistro = dynamic_cast<distribution*>(outvector.get());
  double
    *indata = invector->get_data(p),
    *outdata = outvector->get_data(p);

  auto pstruct = outvector->get_processor_structure(p),
    nstruct = outvector->get_numa_structure(),
    gstruct = outvector->get_global_structure();
  domain_coordinate
    pfirst = pstruct->first_index_r(), plast = pstruct->last_index_r(),
    nfirst = nstruct->first_index_r(),
    gfirst = gstruct->first_index_r(), glast = gstruct->last_index_r(),
    nsize = nstruct->local_size_r(),
    offsets = nfirst-gfirst;

  // index_int
  //   mfirst = outvector->first_index_r(p)[0],
  //   gfirst = outvector->global_first_index(0),
  //   mlast = outvector->last_index(p)[0],
  //   glast = outvector->global_last_index()[0];

  index_int
    tar0 = outvector->location_of_first_index(*outvector.get(),p),
    src0 = invector->get_numa_structure()
            ->linear_location_of(outvector->get_processor_structure(p)),
    len = outdistro->volume(p);

  index_int
    gsize = outdistro->global_volume(),
    ingsize = indistro->global_size().at(0);


  //snippet threepointsumbump
  int ilo = 0, ilen = len;
  if (pfirst==gfirst) { //(mfirst==gfirst) {
    // first element is globally first: the invector does not stick out to the left
    int i = 0; 
    outdata[tar0+i] = indata[src0+i] + indata[src0+i+1];
    ilo++;
  }
  if (plast==glast) { //(mlast==glast) {
     // local last is globally last; just compute and lower the length
    index_int i = ilen-1;
    outdata[tar0+i] = indata[src0+i-1] + indata[src0+i];
    // fmt::print("global last {} from {},{} giving {}\n",
    // 	       glast,indata[src0+i-1],indata[src0+i],outdata[tar0+i]);
    ilen--;
  }
  for (index_int i=ilo; i<ilen; i++) {
    // regular case: the invector sticks out one to the left
    outdata[tar0+i] = indata[src0+i-1]+indata[src0+i]+indata[src0+i+1];
  }
  //snippet end
  *flopcount += 2*len;
}

