/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pointfun.cxx : function taking struct arguments
 ****
 ****************************************************************/

#include <cmath>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

struct vector { double x; double y; } ;

//codesnippet structpass
double distance
    ( struct vector p1,struct vector p2 ) {
  double d1 = p1.x-p2.x, d2 = p1.y-p2.y;
  return sqrt( d1*d1 + d2*d2 );
}
//codesnippet end

int main() {

  //codesnippet structpass
  struct vector p1 = { 1.,1. };
  cout << "Displacement x,y?" << endl;
  double dx,dy; cin >> dx >> dy;
  cout << "dx=" << dx << ", dy=" << dy << endl;
  struct vector p2 = { p1.x+dx,p1.y+dy };
  cout << "Distance: " << distance(p1,p2) << endl;
  //codesnippet end
  
  return 0;
}

