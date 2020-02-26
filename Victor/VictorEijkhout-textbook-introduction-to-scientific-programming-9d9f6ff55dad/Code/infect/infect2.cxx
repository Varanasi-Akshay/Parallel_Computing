/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017/8 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** infect2.cxx : multiple persons random infection
 ****               using classes
 ****
 ****************************************************************/

#include <cmath>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;
#include <iomanip>
using std::setw;

#include <random>
#include <ctime>
#include <vector>

#include "infect_lib.h"

int main() {

  // initialize random generator
  srand(100*time(NULL)%100);

  int npeople;
  cout << "Size of population? ";
  cin >> npeople;
  cout  << endl;
  Population population(npeople);

  float probability;
  cout << "Probability of transfer? ";
  cin >> probability;
  cout << endl;
  
  // set patient zero
  population.random_infection();
  population.set_probability_of_transfer(probability);

  std::vector<int> history;
  int max_infected = 1;
  int max_step = 1;
  for ( ; ; max_step++) {

    int count_infected{0};
    population.update();
    count_infected = population.count_infected();
    cout << "In step " << setw(3) << max_step
         << " #sick: " << setw(4) << count_infected
         << " : " << population.display() << endl;
    //cout << "In step " << max_step << " #sick: " << count_infected << endl;

    history.push_back(count_infected);
    if (count_infected>max_infected)
      max_infected = count_infected;
    population.display();
    if (count_infected==0)
      break;

  }

  cout << "Disease ran its course by step " << max_step << endl;
  cout << "maximum number infected at any time: " << max_infected << endl;

  return 0;
}
