/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** destructor.cxx : illustration of objects going out of scope
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

//codesnippet objaccessref
class SomeObject {
private:
  float x=0.;
public:
  SomeObject( float v ) : x(v) {};
  float &xvalue() { return x; };
};

int main() {
  SomeObject myobject(1.);
  cout << "Object member initially :"
       << myobject.xvalue() << endl;
  myobject.xvalue() = 3.;
  cout << "Object member updated   :"
       << myobject.xvalue() << endl;
  //codesnippet end

  return 0;
}
