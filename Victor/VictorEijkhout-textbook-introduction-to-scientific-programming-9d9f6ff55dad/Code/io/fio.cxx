/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** fio.cxx : file io
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#include <iomanip>
using std::right;
using std::setbase;
using std::setfill;
using std::setw;

#include <fstream>
using std::ofstream;

int main() {

  //codesnippet fio
  ofstream file_out;
  file_out.open("fio_example.out");
  //codesnippet end

  int number;
  cout << "A number please: ";
  cin >> number;
  //codesnippet fio
  file_out << number << endl;
  file_out.close();
  //codesnippet end
  cout << "Written." << endl;

  return 0;
}

