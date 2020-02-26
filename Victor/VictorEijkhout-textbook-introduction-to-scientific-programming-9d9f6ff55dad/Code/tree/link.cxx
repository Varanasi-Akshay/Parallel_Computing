/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** link.cxx : linked list using old-style pointers
 ****
 ****************************************************************/

#include <cmath>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <vector>
using std::vector;

//codesnippet linklist
class Node {
private:
  int data{0},count{0};
  Node *next{nullptr};
public:
  Node() {}
  Node(int value) { data = value; count++; };
  bool hasnext() {
    return next!=nullptr; };
  //codesnippet end
  //codesnippet linkinsert
  Node *insert(int value) {
    if (value==this->data) {
      // we have already seen this value: just count
      count ++;
      return this;
    } else if (value>this->data) {
      // value belong in the tail
      if (!hasnext())
	next = new Node(value);
      else
	next = next->insert(value);
      return this;
    } else {
      // insert at the head of the list
      Node *newhead = new Node(value);
      newhead->next = this;
      return newhead;
    }
  };
  //codesnippet end
  void print() {
    cout << data << ":" << count;
    if (hasnext()) {
      cout << ", "; next->print();
    }
  };
};

int main() {

  Node *head = new Node(1);
  head->print();
  cout << endl;
  
  head = head->insert(5);
  head = head->insert(5);
  head->print();
  cout << endl;

  head = head->insert(2);
  head->print();
  cout << endl;

  return 0;
}
