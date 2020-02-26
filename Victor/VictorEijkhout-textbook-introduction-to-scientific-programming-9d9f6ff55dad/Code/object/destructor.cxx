/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016-8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** destructor.cxx : illustration of objects going out of scope
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

//examplesnippet destructor
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

//examplesnippet destructor
  cout << "Before the nested scope" << endl;
  {
    SomeObject obj;
    cout << "Inside the nested scope" << endl;  
  }
  cout << "After the nested scope" << endl;
//examplesnippet end

  return 0;
}
