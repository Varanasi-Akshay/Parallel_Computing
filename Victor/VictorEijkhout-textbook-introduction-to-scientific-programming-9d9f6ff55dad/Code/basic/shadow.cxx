/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** shadow.cxx : illustrate scope
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  {
    cout << "True" << endl;
    //codesnippet shadowtrue
    bool something{true};
    int i = 3;
    if ( something ) {
      int i = 5;
      cout << i << endl;
    }
    cout << i << endl;
    if ( something ) {
      float i = 1.2;
      cout << i << endl;
    }
    cout << i << endl;
    //codesnippet end
    cout << "true" << endl;
  }

  {
    cout << "False" << endl;
    //codesnippet shadowfalse
    bool something{false};
    int i = 3;
    if ( something ) {
      int i = 5;
      cout << i << endl;
    }
    cout << i << endl;
    if ( something ) {
      float i = 1.2;
      cout << i << endl;
    }
    cout << i << endl;
    //codesnippet end
    cout << "false" << endl;
  }

  return 0;
}
