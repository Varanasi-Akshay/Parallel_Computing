/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** arrayprint.cxx : printable array class
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  float values[] = {1.1, 2.2, 3.3};
  cout << values.size() << endl;

  return 0;
}
