/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** rhsref.cxx : result of an expression can not be reference
 ****
 ****************************************************************/

#include <iostream>
using std::cout;
using std::endl;

//codesnippet rhsrefclass
class myclass {
private:
  int stored{0};
public:
  myclass(int i) : stored(i) {};
  int &data() { return stored; };
};
//codesnippet end

int main() {

  //codesnippet rhsref
  myclass obj(5);
  cout << "object data: " << obj.data() << endl;
  int dcopy = obj.data();
  dcopy++;
  cout << "object data: " << obj.data() << endl;
  int &dref = obj.data();
  dref++;
  cout << "object data: " << obj.data() << endl;
  auto dauto = obj.data();
  dauto++;
  cout << "object data: " << obj.data() << endl;
  auto &aref = obj.data();
  aref++;
  cout << "object data: " << obj.data() << endl;
  //codesnippet end

  return 0;
}
