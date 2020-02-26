/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** arraymaxidx.cxx : static array length examples
 ****
 ****************************************************************/

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//examplesnippet arraypass
void print_first_index( int ar[] ) {
  cout << "First index: " << ar[0] << endl;
}
//examplesnippet end

int main() {

  //examplesnippet arrayinit
  {
    int numbers[] = {5,4,3,2,1};
    cout << numbers[3] << endl;
  }
  {
    int numbers[5]{5,4,3,2,1};
    cout << numbers[3] << endl;
  }
  {
    int numbers[5] = {2};
    cout << numbers[3] << endl;
  }
  //examplesnippet end

  cout << "Rangemax" << endl;
  {
    //examplesnippet rangemax
    int numbers[] = {1,4,2,6,5};
    int tmp_max = numbers[0];
    for (auto v : numbers)
      if (v>tmp_max)
        tmp_max = v;
    cout << "Max: " << tmp_max << " (should be 6)" << endl;
    //examplesnippet end
  }
  cout << "--rangemax" << endl;
    
  cout << "Idxmax" << endl;
  {
    //examplesnippet idxmax
    int numbers[] = {1,4,2,6,5};
    int tmp_idx = 0;
    int tmp_max = numbers[tmp_idx];
    for (int i=0; i<5; i++) {
      int v = numbers[i];
      if (v>tmp_max) {
        tmp_max = v; tmp_idx = i;
      }
    }
    cout << "Max: " << tmp_max
         << " at index: " << tmp_idx << endl;
    //examplesnippet end
  }
  cout << "--idxmax" << endl;
    
//examplesnippet arraypass
  {
    int numbers[] = {1,4,2,5,6};
    print_first_index(numbers);
  }
//examplesnippet end
  
  return 0;
}
