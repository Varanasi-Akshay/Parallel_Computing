/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** printpoint.cxx : print a pointer
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

int main() {

  {
    cout << "PrintfPoint" << endl;
    //codesnippet printfpoint
    int i;
    printf("address of i: %ld\n",
	   (long)(&i));
    printf(" same in hex: %lx\n",
	   (long)(&i));
    //codesnippet end
    cout << "printfpoint" << endl;
  }
  {
    cout << "CoutPoint" << endl;
    //codesnippet coutpoint
    int i;
    cout << "address of i, decimal: "
	 << (long)&i << endl;
    cout << "address if i, hex    : "
	 << std::hex << &i << endl;
    //codesnippet end
    cout << "coutpoint" << endl;
  }

  return 0;
}
