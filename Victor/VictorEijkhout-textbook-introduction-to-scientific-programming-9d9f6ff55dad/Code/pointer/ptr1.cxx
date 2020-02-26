/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** ptr1.cxx : shared pointers
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

#include <memory>
using std::shared_ptr;

//codesnippet thingcall
class thing {
public:
  thing() { cout << "calling constructor\n"; };
  ~thing() { cout << "calling destructor\n"; };
};
//codesnippet end

int main() {

  //codesnippet shareptr1
  cout << "set pointer1"
       << endl;
  auto thing_ptr1 =
    shared_ptr<thing>
      ( new thing );
  cout << "overwrite pointer"
       << endl;
  thing_ptr1 = nullptr;
  //codesnippet end

  return 0;
}
