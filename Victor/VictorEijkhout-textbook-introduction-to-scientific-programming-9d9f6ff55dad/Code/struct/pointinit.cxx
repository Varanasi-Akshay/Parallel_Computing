/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vectorinit.cxx : 
 ****
 ****************************************************************/

//codesnippet pointinit
struct vector_a { double x; double y; } ;
// needs compiler option: -std=c++11
struct vector_b { double x=0; double y=0; } ;

int main() {

  // initialization when you create the variable:
  struct vector_a x_a = {1.5,2.6};
  // initialization done in the structure definition:
  struct vector_b x_b;
  // ILLEGAL:
  // x_b = {3.7, 4.8};
  x_b.x = 3.7; x_b.y = 4.8;
  //end snippet
  
  return 0;
}

