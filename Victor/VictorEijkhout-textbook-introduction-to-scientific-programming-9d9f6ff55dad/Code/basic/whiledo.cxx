/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** whiledo.cxx : while loop with pre-test
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {
  int invar;

  //codesnippet whiledo
  cout << "Enter a positive number: " ;
  cin >> invar;
  while (invar>0) {
    cout << "Enter a positive number: " ;
    cin >> invar;
  }
  cout << "Sorry, " << invar << " is negative" << endl;
  //codesnippet end
  
  return 0;
}

