/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** rotate.cxx : rotate coordinates
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <cmath>
using std::cos;
using std::sin;

void rotate(float &x,float &y,const float theta) {
  float
    xx = -cos(theta)*x + sin(theta)*y,
    yy = sin(theta)*x + cos(theta)*y;
  x = xx; y = yy;
}

int main() {

  float x = sqrt(2)/2, y=sqrt(2)/2;
  const float pi = 2*acos(0.0);
  rotate(x,y,pi/4);
  cout << "Rotated to the y-axis: (" << x << "," << y << ")" << endl;

  return 0;
}
