/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** iof.cxx : formatted float io
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
using std::fixed;
using std::scientific;
using std::setprecision;
using std::right;
using std::setbase;
using std::setfill;
using std::setw;

int main() {

  double x;
  cout << "Float precision applies to non-exponent:" << endl;
  x = 1.234567;
  for (int i=0; i<10; i++) {
    cout << setprecision(4) << x << endl;
    x *= 10;
  }
  cout << endl;
  
  cout << "Fixed precision applies to fractional part:" << endl;
  x = 1.234567;
  cout << fixed;
  for (int i=0; i<10; i++) {
    cout << setprecision(4) << x << endl;
    x *= 10;
  }
  cout << endl;
  
  cout << "Combine width and precision:" << endl;
  x = 1.234567;
  cout << fixed;
  for (int i=0; i<10; i++) {
    cout << setw(10) << setprecision(4) << x << endl;
    x *= 10;
  }
  cout << endl;
  
  cout << "Combine width and precision:" << endl;
  x = 1.234567;
  cout << scientific;
  for (int i=0; i<10; i++) {
    cout << setw(10) << setprecision(4) << x << endl;
    x *= 10;
  }
  cout << endl;
  
  return 0;
}

