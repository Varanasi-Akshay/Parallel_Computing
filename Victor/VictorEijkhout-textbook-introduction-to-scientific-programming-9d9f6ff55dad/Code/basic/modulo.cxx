/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** modulo.cxx : playing with modulo
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {
  int number1,number2;

  cout << "Number 1: " ;
  cin >> number1;
  cout << "Number 2: " ;
  cin >> number2;

  //codesnippet modulo
  int quotient,modulo;
  quotient = number1/number2;
  modulo = number1 - number2*quotient;

  cout << "Compute modulus: " << modulo << endl;
  cout << "built-in modulus: " << number1%number2 << endl;
  //codesnippet end
  cout << "truncate: " << number1-modulo << endl;

  return 0;
}
