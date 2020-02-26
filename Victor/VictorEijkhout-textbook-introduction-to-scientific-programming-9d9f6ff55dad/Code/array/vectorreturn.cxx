/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vectorreturn.cxx : return vector from function
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//codesnippet vectorreturn
vector<int> make_vector(int n) {
  vector<int> x(n);
  x[0] = n;
  return x;
}
//codesnippet end
  
int main() {

  //codesnippet vectorreturn
  vector<int> x1 = make_vector(10); // "auto" also possible!
  cout << "x1 size: " << x1.size() << endl;
  cout << "zero element check: " << x1[0] << endl;
  //codesnippet end
  
  return 0;
}
