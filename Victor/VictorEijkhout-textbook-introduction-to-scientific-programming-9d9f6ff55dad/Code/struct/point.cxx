/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** point.cxx : struct for cartesian vector
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

//codesnippet structdef
struct vector { double x; double y; } ;
//codesnippet end

//codesnippet structuse
int main() {

  struct vector p1,p2;

  p1.x = 1.; p1.y = 2.;
  p2 = {3.,4.};

  p2 = p1;
  cout << "p2: " << p2.x << "," << p2.y << endl;
  //codesnippet end
  
  return 0;
}

