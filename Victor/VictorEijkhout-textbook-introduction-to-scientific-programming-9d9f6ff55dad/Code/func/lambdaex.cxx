/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** lambdaex.cxx : lambda example
 ****
 ****************************************************************/

#include <functional>
using std::function;

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

int main() {

  cout <<
    //codesnippet lambdaexp
  [] (float x,float y) -> float {
    return x+y; } ( 1.5, 2.3 )
    //codesnippet end
  << endl;

  //codesnippet lambdavar
  auto summing = 
    [] (float x,float y) -> float {
    return x+y; };
  cout << summing ( 1.5, 2.3 ) << endl;
  //codesnippet end
  
  return 0;
}
