/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** arraypass.cxx : arrays are passed by reference
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//examplesnippet localparm
void change_scalar(int i) { i += 1; }
//examplesnippet end

//examplesnippet readonlyparm
/* This does not compile:
   void change_const_scalar(const int i) { i += 1; }
*/
//examplesnippet end

//examplesnippet refparm
void change_scalar_by_reference(int &i) { i += 1; }
//examplesnippet end

//examplesnippet starparm
void change_scalar_old_style(int *i) { *i += 1; }
//examplesnippet end

//examplesnippet arrayparm
void change_array_location( int ar[], int i ) { ar[i] += 1; }
//examplesnippet end

int main() {

  int number;

  number = 3;
  change_scalar(number);
  cout << "is this still 3? " << number << endl;
  
  number = 3;
  change_scalar_by_reference(number);
  cout << "is this still 3? " << number << endl;
  
//examplesnippet starparm
  number = 3;
  change_scalar_old_style(&number);
//examplesnippet end
  cout << "is this still 3? " << number << endl;
  
//examplesnippet arrayparm
  int numbers[5];
  numbers[2] = 3.;
  change_array_location(numbers,2);
//examplesnippet end
  cout << "is this still 3? " << numbers[2] << endl;
  
  return 0;
}
