/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** infect2.cxx : multiple persons random infection
 ****
 ****************************************************************/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
//#include <algorithm>
#include <ctime>
#include <vector>

using namespace std;

#include "infect_lib.h"

int main() {

  // initialize random generator
  srand(100*time(NULL)%100);

  int npeople;
  cout << "Size of population? ";
  cin >> npeople;
  cout  << endl;
  int population[npeople],tmp_population[npeople];

  float probability;
  cout << "Probability of transfer? ";
  cin >> probability;
  cout << endl;
  
  // set initial population
  for (int person=0; person<npeople; person++)
    population[person] = 0;
  // set patient zero
  population[npeople/2] = random_duration();

  std::vector<int> history;
  int max_infected = 1;
  int max_step = 1;
  for ( ; ; max_step++) {

    time_step_without_spread(tmp_population,population,npeople);
    spread_infection(tmp_population,population,npeople,probability);

    // copy into the official state
    int count_infected=0;
    for (int i=0; i<npeople; i++) {
      count_infected += tmp_population[i]>0;
      population[i] = tmp_population[i];
    }
    history.push_back(count_infected);
    if (count_infected>max_infected)
      max_infected = count_infected;
    display_population(population,npeople);
    if (count_infected==0)
      break;

  }

  cout << "Disease ran its course by step " << max_step << endl;

  return 0;
}
