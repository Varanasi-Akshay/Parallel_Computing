/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vectorcopy.cxx : example of vector copying
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

int main() {

  //codesnippet vectorcopy
  vector<float> v(5,0), vcopy;
  v[2] = 3.5;
  vcopy = v;
  cout << vcopy[2] << endl;
  //codesnippet end

  return 0;
}
