/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** string.cxx : basic string stuff
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

using std::string;

int main() {
  string first{"this"}, second{"is"}, third,sum;
  third = "text";
  sum = first+" "+second+" "+third;

  cout << "Sum string is: <<" << sum << ">>" << endl;
  cout << "Sum string has " << sum.size() << " characters" << endl;
  cout << "The second character is <<" << sum[1] << ">>" << endl;

  return 0;
}

