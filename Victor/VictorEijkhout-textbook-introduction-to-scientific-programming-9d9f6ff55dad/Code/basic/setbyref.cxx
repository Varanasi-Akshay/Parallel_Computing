/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** setbyref.cxx : illustration of passing by ref
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

//codesnippet setbyref
void f( int &i ) {
  i = 5;
}
int main() {

  int var = 0;
  f(var);
  cout << var << endl;
  //codesnippet end

  return 0;
}
