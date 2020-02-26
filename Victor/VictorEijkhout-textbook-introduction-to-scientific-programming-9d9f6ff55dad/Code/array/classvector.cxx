/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** classvector.cxx : use of vector in class
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

class vectorclass {
private:
  vector<int> internal;
public:
  vectorclass(int l) {
    for (int i=0; i<l; i++)
      internal.push_back(0);
  };
  int size() { return internal.size(); };
};

int main() {
  int array_length;
  cout << "How many elements? ";
  cin >> array_length;

  vector<double> my_array(array_length);
  cout << "my array has length " << my_array.size() << endl;

  vector<double> my_reserve; my_reserve.reserve(array_length);
  cout << "my reserve has length " << my_reserve.size() << endl;
  
  vectorclass vc(array_length);
  cout << "my class has length " << vc.size() << endl;
  return 0;
}
