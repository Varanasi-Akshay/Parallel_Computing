/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** ref.cxx : using references, not as parameter
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  //codesnippet refint
  int i;
  int &ri = i;
  i = 5;
  cout << i << "," << ri << endl;
  i *= 2;
  cout << i << "," << ri << endl;
  ri -= 3;
  cout << i << "," << ri << endl;
  //codesnippet end

  return 0;
}
