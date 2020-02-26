/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** lambdait.cxx : lambda example
 ****
 ****************************************************************/

#include <iostream>
#include <cmath>
using std::cout;
using std::endl;

float f(float x) { return 2*x; };

float nf(float x,int n) {
  if (n==0)
    return x;
  else
    return f( nf(x,n-1) );
};

int main() {

  for (int exponent=1; exponent<=5; exponent++) {
    //codesnippet lambdacapt
    auto powerfunction = [exponent] (float x) -> float {
      return pow(x,exponent); };
    //codesnippet end
    cout << "To the power " << exponent << endl;
    for (float x=1.; x<=9.; x+=1.)
      cout << x << ":" << powerfunction(x) << endl;
  }
  
  return 0;
}
