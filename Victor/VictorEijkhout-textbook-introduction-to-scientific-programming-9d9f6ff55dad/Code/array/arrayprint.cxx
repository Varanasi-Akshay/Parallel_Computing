/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** arrayprint.cxx : printable array class
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

//codesnippet printablevector
class printable {
private:
  vector<int> values;
public:
  printable(int n) {
    values = vector<int>(n);
  };
  string stringed() {
    string p("");
    for (int i=0; i<values.size(); i++)
      p += to_string(values[i])+" ";
    return p;
  };
  //codesnippet end
  //codesnippet vectorinheritat
  int &at(int i) {
    return values.at(i);
  };
  //codesnippet end
  //codesnippet printablevector
};
//codesnippet end

int main() {

  int length = 5;
  printable pv(length);
  for (int i=0; i<length; i++)
    pv.at(i) = length-i;
  cout << pv.stringed() << endl;

  return 0;
}
