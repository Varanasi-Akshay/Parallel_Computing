/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** exceptdestruct.cxx : an exception calls the destructor
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

//examplesnippet exceptdestruct
class SomeObject {
public:
  SomeObject() { cout <<
    "calling the constructor" 
    << endl; };
  ~SomeObject() { cout <<
    "calling the destructor" 
    << endl; };
};
//examplesnippet end

int main() {

//examplesnippet exceptdestruct
  try {
    SomeObject obj;
    cout << "Inside the nested scope" << endl;
    throw(1);
  } catch (...) {
    cout << "Exception caught" << endl;
  }
//examplesnippet end

  return 0;
}
