/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** io.cxx : formatted io
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;
using std::setfill;
using std::left;
using std::right;
using std::setbase;

int main() {

  cout << "Unformatted:" << endl;
  //codesnippet cunformat
  for (int i=1; i<200000000; i*=10)
    cout << "Number: " << i << endl;
  cout << endl;
  //codesnippet end
  cout << "--unformatted" << endl;
  
  cout << "Width is 6:" << endl;
  cout << setw(6) << 1 << 2 << 3 << endl;
  cout << endl;
  
  cout << "Set width:" << endl;
  //codesnippet formatwidth6
  cout << "Width is 6:" << endl;
  for (int i=1; i<200000000; i*=10)
    cout << "Number: "
	 << setw(6) << i << endl;
  cout << endl;
  //codesnippet end
  cout << "--set width:" << endl;
  
  cout << "Padding:" << endl;
  //codesnippet formatpad
  for (int i=1; i<200000000; i*=10)
    cout << "Number: "
	 << setfill('.') << setw(6) << i << endl;
  cout << endl;
  //codesnippet end
  cout << "--padding:" << endl;
  
  cout << "Left align:" << endl;
  //codesnippet formatleft
  for (int i=1; i<200000000; i*=10)
    cout << "Number: "
	 << left << setfill('.') << setw(6) << i << endl;
  //codesnippet end
  cout << endl;
  cout << "--left align:" << endl;

  cout << "Base 16:" << endl;
  //codesnippet format16
  cout << setbase(16) << setfill(' ');
  for (int i=0; i<16; i++) {
    for (int j=0; j<16; j++)
      cout << i*16+j << " " ;
    cout << endl;
  }
  //codesnippet end
  cout << endl;
  cout << "--base 16:" << endl;
  
  cout << "Format16:" << endl;
  cout << setbase(16) << setfill('0') << right ;
  for (int i=0; i<16; i++) {
    for (int j=0; j<16; j++)
      cout << setw(2) << i*16+j << " " ;
    cout << endl;
  }
  cout << endl;
  cout << "--format16:" << endl;
  return 0;
}

