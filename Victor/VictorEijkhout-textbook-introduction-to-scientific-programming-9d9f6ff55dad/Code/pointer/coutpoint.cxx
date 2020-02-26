/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** coutpoint.cxx : print a pointer
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  //codesnippet coutpoint
  int i;
  cout << "address of i, decimal: "
       << (long)&i << endl;
  cout << "address if i, hex    : "
       << std::hex << &i << endl;
  //codesnippet end

  return 0;
}
