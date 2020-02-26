/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** arraytime.cxx : time the overhead for flexible vectors
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

int main() {

  vector<int> array(50);
  cout << "Array length: " << array.size() << endl;
  
  vector< vector<int> > matrix(100,array);
  cout << "Matrix length: " << matrix.size() << endl;
  cout << ".. first row length: " << matrix[0].size() << endl;

  matrix[0][1] = 3.14;
  cout << ".. this had better not be 3.14: " << matrix[1][0] << endl;

  return 0;
}
