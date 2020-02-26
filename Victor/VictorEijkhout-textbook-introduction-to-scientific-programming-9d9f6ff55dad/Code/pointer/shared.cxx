/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** shared.cxx : shared pointers
 ****
 ****************************************************************/

#include <iostream>
#include <memory>
using namespace std;

//codesnippet thingcall
class thing {
public:
  thing() { cout << "calling constructor\n"; };
  ~thing() { cout << "calling destructor\n"; };
};
//codesnippet end

int main() {

  cout << "Ptr1" << endl;
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
  cout << "-- ptr1" << endl;

  cout << "Ptr2" << endl;
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
  cout << "-- ptr2" << endl;

  return 0;
}
