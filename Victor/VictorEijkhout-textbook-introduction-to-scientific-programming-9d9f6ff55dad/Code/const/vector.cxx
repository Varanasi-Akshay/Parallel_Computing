/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016/7 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** vector.cxx : illustrating the use of const
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

class transparent {
private:
  std::vector<int> bag;
public:
  transparent() {};
  int add(int i) { bag.push_back(i); };
  int get_value(int i) { return bag.at(i); };
  int &get_reference(int i) { return bag.at(i); };
  const int &get_const_reference(int i) { return bag.at(i); };
};

int main() {
  transparent Istore;
  Istore.add(5);
  Istore.add(7);
  Istore.add(8);
  cout << " I have " << Istore.get_value(1) << endl;
  int get1 = Istore.get_value(1);
  get1++;
  cout << " I have " << Istore.get_reference(1) << endl;
  Istore.get_reference(1)++;
  cout << " I have " << Istore.get_reference(1) << endl;

  cout << " I have " << Istore.get_const_reference(1) << endl;
  // The next line gives a compiler error
  //Istore.get_const_reference(1)++;

  return 0;
}
