/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** f2c.cxx : fahrenheit to centigrade conversion
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {
  float c,f;

  cout << "Enter a temperature in Fahrenheit: " << endl;
  cin >> f;
  c = (f-32)*5/9.;
  cout << "Equivalent Celsius: " << c << endl;
  
  return 0;
}
