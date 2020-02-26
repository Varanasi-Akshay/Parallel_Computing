/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pointprivate.cxx : Vector class with private data
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

//codesnippet pointprivate
class Vector {
private: // recommended!
  double vx,vy;
public:
  Vector( double x,double y ) {
    vx = x; vy = y;
  };
  //codesnippet end
  //codesnippet pointprivateset
  double x() { return vx; };
  double y() { return vy; };
  void setx( double newx ) {
    vx = newx; };
  void sety( double newy ) {
    vy = newy; };
  //codesnippet end
  //codesnippet pointprivatedefine
}; // end of class definition

int main() {
  Vector p1(1.,2.);
  //codesnippet end
  cout << "p1 = " << p1.x() << "," << p1.y() << endl;

  //codesnippet pointprivatesetuse
  p1.setx(3.12);
  /* ILLEGAL: p1.x() = 5; */
  cout << "P1's x=" << p1.x() << endl;
  //codesnippet end

  return 0;
}
