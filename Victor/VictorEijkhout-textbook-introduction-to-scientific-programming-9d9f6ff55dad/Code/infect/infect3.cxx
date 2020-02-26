/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2017 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** infect3.cxx : testing reproduction number in 1d
 ****
 ****************************************************************/

#include <cstring>

#include <iostream>
using std::cin;
using std::cout;
using std::endl;

#include <iomanip>
using std::setw;

#include <ctime>
#include <vector>

#include "infect_lib.h"

int main( int argc,char **argv) {

  // initialize random generator
  srand(100*time(NULL)%100);

  bool display{false};
  for (int iarg=1; iarg<argc; iarg++) {
    if (!strcmp(argv[iarg],"-h"))
      cout << "Usage: " << argv[0] << " [ -d ] " << endl;
    else if (!strcmp(argv[iarg],"-d"))
      display = true;
  }

  int npeople;
  cout << "Size of population?";
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

  float inoculation; 
  cout << "Innocalation percentage?";
  cin >> inoculation;
  cout << endl;
  population.inoculate(inoculation);
  int n_susceptible = population.count_susceptible();

  int contacts_per_day=6, duration  = 5;

  // count how many people can get infected
  std::vector<int> history;
  population.random_infection();
  int max_infected = 1, max_step=0;
  history.push_back(1);

  for ( ; ; max_step++) {

    int infected{0};
    population.update(contacts_per_day,duration);
    infected = population.count_infected();
    if (display)
      cout << "In step " << setw(3) << max_step
	   << " #sick: " << setw(4) << infected
	   << " : " << population.display() << endl;

    history.push_back(infected);
    if (infected>max_infected)
      max_infected = infected;
    
    if (infected==0)
      break;

  }

  cout << "\nAn inoculation percentage of " << inoculation
       << " and probability of transfer " << probability
       << " leads to a maximum number of infected of "
       << setw(3) << max_infected-1 << " or "
       << 100.*max_infected/npeople << " percent" << endl;
  cout << "The disease ran its course in " << history.size() << " days" << endl;
  // cout << "history:";
  // for (int istep=0; istep<max_step; istep++)
  //   cout << setw(4) << history[istep];
  // cout << endl;
  int n_unaffected = population.count_susceptible();
  // for (int i=0; i<npeople; i++)
  //   n_unaffected += population[i]==0;
  cout << "susceptible people unaffected: " << n_unaffected
       << " out of " << n_susceptible << endl;
  
  return 0;
}
