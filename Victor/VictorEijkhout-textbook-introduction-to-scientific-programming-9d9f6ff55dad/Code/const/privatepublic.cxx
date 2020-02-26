/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** privatepublic.cxx : illustrating the use of const
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

class transparent {
public:
  int remember{0};
  transparent( int i ) { remember = i; };
  int mynumber() { return remember; };
};

class solid {
private:
  int remember{0};
public:
  solid( int i ) { remember = i; };
  int mynumber() { return remember; };
};

int main() {
  transparent Istore(5);
  cout << " I have " << Istore.mynumber() << endl;
  Istore.remember = 6;
  cout << " I have " << Istore.mynumber() << endl;

  solid Iremember(5);
  cout << " I remember " << Iremember.mynumber() << endl;
  // The following line leads to a compiler error:
  //Iremember.remember = 6;

  return 0;
}
