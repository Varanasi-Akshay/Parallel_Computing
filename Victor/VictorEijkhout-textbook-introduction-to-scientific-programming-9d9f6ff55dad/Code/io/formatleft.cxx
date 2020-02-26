/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** formatleft.cxx : left aligned io
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#include <iomanip>
using std::left;
using std::right;
using std::setbase;
using std::setfill;
using std::setw;

int main() {

  //codesnippet formatleft
  for (int i=1; i<200000000; i*=10)
    cout << "Number: "
	 << left << setfill('.') << setw(6) << i << endl;
  //codesnippet end
  cout << endl;
  
  return 0;
}

