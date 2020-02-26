/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2018 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** lambdafun.cxx : storing a lambda
 ****
 ****************************************************************/

//codesnippet lambdaclass
#include <functional>
using std::function;
//codesnippet end

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//codesnippet lambdaclass
class SelectedInts {
private:
  vector<int> bag;
  function< bool(int) > selector;
public:
  SelectedInts( function< bool(int) > f ) {
    selector = f; };
  void add(int i) {
    if (selector(i))
      bag.push_back(i);
  };
  int size() { return bag.size(); };
};
//codesnippet end

int main() {

  SelectedInts greaterthan5
    ( [] (int i) -> bool { return i>5; } );
  int upperbound = 20;
  for (int i=0; i<upperbound; i++)
    greaterthan5.add(i);
  cout << "Ints under " << upperbound <<
    " greater than 5: " << greaterthan5.size() << endl;
  
  int threshold;
  cout << "Give a threshold: "; cin >> threshold; cout << endl;
  //codesnippet lambdaclassed
  SelectedInts greaterthan
    ( [threshold] (int i) -> bool { return i>threshold; } );
  for (int i=0; i<upperbound; i++)
    greaterthan.add(i);
  cout << "Ints under " << upperbound <<
    " greater than " << threshold << ": " << greaterthan.size() << endl;
  //codesnippet end

  return 0;
}
