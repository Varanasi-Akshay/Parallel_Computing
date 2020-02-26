/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** cindo.cxx : test cin terminating condition
 ****
 ****************************************************************/

#include <cmath>
using std::sqrt;

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {
  float var;
  // VLE  I was hoping that null would abort, but only illegal does
  while ( (bool)(cin >> var) ) {
    cout << "Root of " << var << " is " << sqrt(var) << endl;
  }
  return 0;
}

