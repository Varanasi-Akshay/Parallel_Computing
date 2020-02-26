/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** ptr2.cxx : shared pointers
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

#include <memory>
using std::shared_ptr;

class thing {
public:
  thing() { cout << "calling constructor\n"; };
  ~thing() { cout << "calling destructor\n"; };
};

int main() {

  //codesnippet shareptr2
  cout << "set pointer2" << endl;
  auto thing_ptr2 =
    shared_ptr<thing>
      ( new thing );
  cout << "set pointer3 by copy"
       << endl;
  auto thing_ptr3 = thing_ptr2;
  cout << "overwrite pointer2"
       << endl;
  thing_ptr2 = nullptr;
  cout << "overwrite pointer3"
       << endl;
  thing_ptr3 = nullptr;
  //codesnippet end

  return 0;
}
