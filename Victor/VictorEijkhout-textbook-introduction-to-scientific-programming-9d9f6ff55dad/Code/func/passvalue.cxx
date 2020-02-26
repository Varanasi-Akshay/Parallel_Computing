/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** passvalue.cxx : illustration pass-by-value
 ****
 ****************************************************************/

#include <iostream>
#include <cmath>
using std::cout;
using std::endl;

//examplesnippet passvalue
double f( double x ) {
  x = x*x;
  return x;
}
//examplesnippet end

int main() {

  double number,other;
  //examplesnippet passvalue
  number = 5.1;
  cout << "Input starts as: "
       << number << endl;
  other = f(number);
  cout << "Input var is now: "
       << number << endl;
  cout << "Output var is: "
       << other << endl;
  //examplesnippet end

  return 0;
}
