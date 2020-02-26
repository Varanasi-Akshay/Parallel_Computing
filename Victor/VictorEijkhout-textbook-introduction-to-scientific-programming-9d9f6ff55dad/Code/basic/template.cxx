/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** template.cxx : example of template use
 ****
 ****************************************************************/


template< typename scalar >
scalar square(scalar n) {
  return n*n;
};

#include <iostream>
using std::cout;
using std::endl;

int main() {

  cout << "Real: " << square(2.0) << endl;
  cout << "Integer: " << square(2) << endl;

  return 0;
}

