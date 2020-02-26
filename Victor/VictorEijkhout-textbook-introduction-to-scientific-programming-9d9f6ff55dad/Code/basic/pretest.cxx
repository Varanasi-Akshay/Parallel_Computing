/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pretest.cxx : illustrating if test
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  //codesnippet pretest
  cout << "before the loop" << endl;
  for (int i=5; i<4; i++)
    cout << "in iteration " << i << endl;
  cout << "after the loop" << endl;
  //codesnippet end
  return 0;
}
