/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** ar3.cxx : a templated program for machine epsilon
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//codesnippet itemplate
template<int s> 
std::vector<int> svector(s);
//codesnippet end

int main() {


//codesnippet itemplate
  svector(3) threevector;
  cout << threevector.size();
//codesnippet end

  return 0;
}

