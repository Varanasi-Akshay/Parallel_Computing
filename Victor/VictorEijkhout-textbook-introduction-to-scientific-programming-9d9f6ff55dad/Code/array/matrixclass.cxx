/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** matrixclass.cxx : wrap an array in a class
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//examplesnippet matrixclass
class matrix {
private:
  std::vector<double> the_matrix;
  int m,n;
public:
  matrix(int m,int n) {
    this->m = m; this->n = n;
    the_matrix.reserve(m*n);
  };
  void set(int i,int j,double v) {
    the_matrix[ i*n +j ] = v;
  };
  double get(int i,int j) {
    return the_matrix[ i*n +j ];
  };
  //examplesnippet end
  double &element(int i,int j) {
    return the_matrix[ i*n +j ];
  };
  //examplesnippet matrixclass
};
//examplesnippet end
  
int main() {

  matrix A(20,30);

  A.set(0,1,5.14);
  cout << "Just set (0,1) to " << A.get(0,1) << endl;

  //examplesnippet matrixelement
  A.element(2,3) = 7.24;
  cout << "Just set (2,3) to " << A.get(2,3) << endl;
  //examplesnippet end

  return 0;
}
