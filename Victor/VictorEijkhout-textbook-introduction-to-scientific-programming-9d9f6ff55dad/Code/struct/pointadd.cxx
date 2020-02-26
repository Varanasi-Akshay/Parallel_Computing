/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pointadd.cxx : function returning struct
 ****
 ****************************************************************/

#include <cmath>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

struct vector { double x; double y; } ;

//codesnippet structreturn
struct vector vector_add
      ( struct vector p1,
	struct vector p2 ) {
   struct vector p_add =
     {p1.x+p2.x,p1.y+p2.y};
   return p_add;
};
//codesnippet end

int main() {

  struct vector p1,p2,p3;

  p1.x = 1.; p1.y = 1.;
  p2 = {4.,5.};

//codesnippet structreturn
  p3 = vector_add(p1,p2);
  cout << "Added: " <<
    p3.x << "," << p3.y << endl;
//codesnippet end
  
  return 0;
}

