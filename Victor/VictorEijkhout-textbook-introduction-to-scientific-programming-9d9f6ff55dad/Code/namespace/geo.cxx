/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** geo.cxx : geometry code in name space, using lib and header file
 ****
 ****************************************************************/

#include <iostream>
#include <vector>

//codesnippet nameinclude
#include "geolib.h"
using namespace geometry;
//codesnippet end

int main() {

  std::vector< vector > vectors;
  vectors.push_back( vector( point(1,1),point(4,5) ) );
  std::cout << "We have " << vectors.size() << " vectors" << std::endl;
  std::cout << "and the first has length " << vectors[0].size() << std::endl;

  return 0;
}
