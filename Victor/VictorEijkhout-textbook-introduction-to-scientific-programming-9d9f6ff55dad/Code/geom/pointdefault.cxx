/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pointscale.cxx : Vector class with private data
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <cmath>
using std::sqrt;

class Vector {
private:
  double vx,vy;
public:
//codesnippet pointdef1
  Vector() {};
  Vector( double x,double y ) {
    vx = x; vy = y;
  };
//codesnippet end
  Vector scale( double a ) {
    return Vector( vx*a, vy*a ); };
  double length() { return sqrt(vx*vx + vy*vy); };
};

int main() {
//codesnippet pointdef2
  Vector p1(1.,2.), p2;
  cout << "p1 has length " << p1.length() << endl;
  p2 = p1.scale(2.);
  cout << "p2 has length " << p2.length() << endl;
//codesnippet end

  return 0;
}
