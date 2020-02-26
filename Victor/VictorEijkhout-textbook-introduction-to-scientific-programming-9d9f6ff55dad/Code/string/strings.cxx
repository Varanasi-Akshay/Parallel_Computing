/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** strings.cxx : exploration of the string class
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <string>
using std::string;

int main() {

  string five_text{"fiver"};
  cout << five_text.size() << endl;

  string five_chars;
  cout << five_chars.size() << endl;
  for (int i=0; i<5; i++)
    five_chars.push_back(' ');
  cout << five_chars.size() << endl;

  string my_string;
  my_string = "foo";
  my_string += "bar";
  cout << my_string << ": " << my_string.size() << endl;
  cout << my_string[ my_string.size()-1 ] << endl;
  
  return 0;
}
