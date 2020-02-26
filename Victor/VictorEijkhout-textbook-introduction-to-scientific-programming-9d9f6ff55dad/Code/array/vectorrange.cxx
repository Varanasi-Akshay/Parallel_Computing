/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vectorrange.cxx : range-based indexing over a vector
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

int main() {

  {
    cout << "Rangecopy" << endl;
    //codesnippet vectorrange
    vector<float> myvector
      = {1.1, 2.2, 3.3};
    for ( auto e : myvector )
      e *= 2;
    cout << myvector[2] << endl;
    //codesnippet end
    cout << "--rangecopy" << endl;
  }
  {
    cout << "Rangeref" << endl;
    //codesnippet vectorrangeref
    vector<float> myvector
      = {1.1, 2.2, 3.3};
    for ( auto &e : myvector )
      e *= 2;
    cout << myvector[2] << endl;
    //codesnippet end
    cout << "--rangeref" << endl;
  }

  return 0;
}
