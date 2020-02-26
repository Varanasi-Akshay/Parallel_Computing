/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** infect3.cxx : testing reproduction number in 1d
 ****
 ****************************************************************/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
using namespace std;

#include "infect_lib.h"

int main() {

  int npeople;
  cout << "Size of population?";
  cin >> npeople;
  cout  << endl;
  int population[npeople],tmp_population[npeople];

  // int nsteps;
  // cout << "Number of time steps?";
  // cin >> nsteps;
  // cout  << endl;
  vector<int> history;

  float chance_of_infection;
  cout << "Chance of infection? ";
  cin >> chance_of_infection;
  cout << endl;

  // int repr_number;
  // cout << "Reproduction number?";
  // cin >> repr_number;
  // cout  << endl;

  float inoculation; int iinoc;
  cout << "Innocalation percentage?";
  cin >> iinoc;
  cout << endl;
  if (iinoc>=0 && iinoc<100)
    inoculation = .01 * iinoc;

  // set initial population
  for (int ip=0; ip<npeople; ip++) {
    float chance = (float) rand()/(float)RAND_MAX;
    if (chance<inoculation)
      population[ip] = -1;
    else
      population[ip] = 0;
  }
  int n_susceptible = 0;
  for (int i=0; i<npeople; i++)
    n_susceptible += population[i]==0;

  int contacts_per_day=6, duration  = 5;

  // set patient zero
  population[npeople/2] = duration;

  // count how many people can get infected
  int max_infected = 1, max_step=0;
  history.push_back(1);

  for ( ; ; max_step++) {

    // next generation is the same as this by default
    time_step_without_spread(tmp_population,population,npeople);

    // compute infections of susceptible people
    spread_randomly(tmp_population,population,npeople,
                    contacts_per_day,chance_of_infection,duration);

    // copy into the official state
    int count_infected=0;
    for (int i=0; i<npeople; i++) {
      count_infected += tmp_population[i]>0;
      population[i] = tmp_population[i];
    }
    history.push_back(count_infected);
    if (count_infected>max_infected)
      max_infected = count_infected;
    //display_population(population,npeople);
    if (count_infected==0)
      break;
  }
  cout << "An inoculation percentage of " << iinoc
       << " leads to a maximum number of infected of "
       << setw(3) << max_infected-1 << " or "
       << 100.*max_infected/npeople << " percent" << endl;
  cout << "The disease ran its course in " << history.size() << " days" << endl;
  // cout << "history:";
  // for (int istep=0; istep<max_step; istep++)
  //   cout << setw(4) << history[istep];
  // cout << endl;
  int n_unaffected = 0;
  for (int i=0; i<npeople; i++)
    n_unaffected += population[i]==0;
  cout << "susceptible people unaffected: " << n_unaffected
       << " out of " << n_susceptible << endl;

  return 0;
}
