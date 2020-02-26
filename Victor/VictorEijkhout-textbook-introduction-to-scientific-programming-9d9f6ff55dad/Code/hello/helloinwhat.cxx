/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** helloinwhat.cxx : read line and count, loop output
 ****
 ****************************************************************/

//codesnippet readin
#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#include <sstream>
using std::stringstream;
//codesnippet end

int main() {

//codesnippet readin
  std::string saymany;
  int howmany;

  cout << "How many times? ";
  getline( cin,saymany );
  stringstream saidmany(saymany);
  saidmany >> howmany;
//codesnippet end
  // howmany = stoi(saymany);

  std::string saywhat;
  cout << "say what? ";
  getline( cin, saywhat );

  cout << "Here it comes:" << endl;
  for ( int ihow=0; ihow<howmany; ihow++)
    cout << saywhat << endl;
  return 0;
}
