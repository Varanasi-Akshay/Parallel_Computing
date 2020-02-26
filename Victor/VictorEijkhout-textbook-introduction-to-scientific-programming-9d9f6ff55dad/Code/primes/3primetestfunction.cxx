/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** primetestfunction.cxx : use a function to test primality
 ****     of a user input number
 ****
 **** prerequisites : arithmetic, for & while loops
 ****
 **** teaches : function with return result
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

bool isprime(int number) {
  for (int divisor=2; divisor<number; divisor++) {
    if (number%divisor==0) {
      return false;
    }
  }
  return true;
}

int main() {
  int number;
  cout << "Enter a number: " << endl;
  cin >> number;

  if (isprime(number)) {
    cout << ".. is prime" << endl;
  } else {
    cout << ".. is not prime" << endl;
  }

  return 0;
}
