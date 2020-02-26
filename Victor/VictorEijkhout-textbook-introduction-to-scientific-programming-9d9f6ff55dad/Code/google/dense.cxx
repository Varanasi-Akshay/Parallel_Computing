/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** dense.cxx : adjacency matrix implementation, dense
 ****
 ****************************************************************/

#include <cmath>
#include <iostream>
using std::cout;
using std::endl;
#include <iomanip>
#include <random>
#include <ctime>

#include <vector>
using std::vector;

class Link {
private:
  int target{-1}; float probability{-1.};
public:
  Link(int t,float p) : target(t),probability(p) {};
  Link(int t) : Link(t,1.) {};
  void set_probability( float p ) { probability = p; };
};

class Page {
private:
  vector<Link> links;
public:
  Page() {};
  //Page( int nlinks ) { links = vector<Link>(nlinks,0.); };
  int n_out_links() const { return links.size(); };
  void normalize() { int nlinks = n_out_links();
    for ( auto &link : links )
      link.set_probability( 1./nlinks );
  };
  void add_link(int to) {
    links.push_back( Link(to) );
    normalize();
  };
};

class State {
private:
  vector<float> probabilities;
public:
  State(int size) {
    probabilities = vector<float>(size,0);
  };
  float &at(int i) const {
    return probabilities.at(i);
  };
  int size() const {
    return probabilities.size();
  };
};

class Web {
private:
  vector<Page> pages;
public:
  //! Create a square matrix of a specified size.
  Web(int size) {
    pages = vector<Page>(size,Page(size));
  };
  //! Add a page. This models a link from page i to page j.
  void addpage(int i,int j) {
    // test whether i,j already exists
    pages.at(i).add_link(j);
  };
  //! Count the number of links from a page.
  int number_of_outlinks(int page) {
    return pages.at(page).n_out_links();
  };
  //! Given a probability vector, compute a new vector
  State transition(State oldstate) {};
  void print() {
  };
};

int main() {

  Web internet(10);

  return 0;
}
