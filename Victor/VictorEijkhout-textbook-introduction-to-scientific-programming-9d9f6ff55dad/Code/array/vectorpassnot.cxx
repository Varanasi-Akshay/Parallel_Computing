/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vectorpassnot.cxx : example of vector passed by value
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//codesnippet vectorpassval
void set0
  ( vector<float> v,float x )
{
  v[0] = x;
}
//codesnippet end

int main() {

  //codesnippet vectorpassval
  vector<float> v(1);
  v[0] = 3.5;
  set0(v,4.6);
  cout << v[0] << endl;
  //codesnippet end

  return 0;
}
