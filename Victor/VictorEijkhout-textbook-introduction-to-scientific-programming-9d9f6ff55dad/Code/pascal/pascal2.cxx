/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** pascal2.cxx : pascal triangle program with triangular storage
 ****
 ****************************************************************/

#include <cmath>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#include <iomanip>
using std::setw;

#include <vector>
using std::vector;

class pascal {
private:
  int nrows,number_length;
  vector<int> array;
public:
  pascal(int n) {
    nrows = n;
    int triangle_size = total(nrows);
    array = vector<int>(triangle_size);
    fill();
  };
  int total(int i) { return i*(i+1)/2; };
  void fill() {
    set(1,1,1);
    for (int row=2; row<=nrows; row++)
      fillrow(row);
    int imax = get(nrows,(nrows+1)/2);
    number_length = imax > 0 ? (int) log10 ((double) imax) + 1 : 1;
  };
  void fillrow(int row) {
    for (int col=1; col<=row; col++) {
      int value;
      if (col==1 || col==row)
	value = 1;
      else
	value = get(row-1,col-1)+get(row-1,col);
      set(row,col,value);
    }
  }
  int get(int row,int col) {
    return array.at( total(row-1) + (col-1) );
  };
  int set(int row,int col,int value) {
    array.at( total(row-1) + (col-1) ) = value;
  };
  void print() {
    for (int row=1; row<=nrows; row++) {
      cout << "Row " << setw(number_length+1) << row << ":";
      cout << std::string((number_length-1)*(nrows-row),' ');
      for (int col=1; col<=row; col++)
	cout << setw(number_length+1) << get(row,col);
      cout << endl;
    }
  };
  void print(int m) {
    for (int row=1; row<=nrows; row++) {
      cout << std::string(nrows-row,' ');
      for (int col=1; col<=row; col++) {
	int v = get(row,col);
	char c = v%m ? '*' : ' '; 
	cout << ' ' << c;
      }
      cout << endl;
    }
  };
};

int main() {
  int size;
  cout << "What size?";
  cin >> size; cout << endl;
  pascal triangle(size);
  triangle.print();
  triangle.print(2);
  triangle.print(5);
  triangle.print(7);
  
  return 0;
}

