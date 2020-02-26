/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** wardevolve.cxx : time evolution of a ward of sick/healthy people
 ****
 **** teaches: use of classes, class containing array of other class
 ****
 ****************************************************************/

#include <iostream>
#include <random>
#include <algorithm>
using namespace std;

class patient {
public:
  int patient_health;
public:
  patient(int initial_state) {
    patient_health = initial_state;
  };
};

class ward {
public:
  std::vector<patient> patients;
  std::vector<patient> tmp_patients;
  default_random_engine make_random;
public:
  ward(int npatients) {
    uniform_int_distribution<int> health_state(0,1);
    for (int ip=0; ip<npatients; ip++) {
      patients.push_back( patient(health_state(make_random)) );
      tmp_patients.push_back( patient(0) );
    }
  };
  void nexttime() {
    uniform_int_distribution<int> propagate(0,5);
    for (int ip=0; ip<patients.size(); ip++) {
      int surrounding_state = 0;
      if (ip>0)
	surrounding_state += patients[ip-1].patient_health;
      if (ip<patients.size()-1)
	surrounding_state += patients[ip+1].patient_health;
      tmp_patients[ip].patient_health =
	surrounding_state * propagate(make_random);
    }
    for (int ip=0; ip<patients.size(); ip++) {
      patients[ip].patient_health = tmp_patients[ip].patient_health;
    }
  };
  void print() {
    for (int ip=0; ip<patients.size(); ip++) {
      if (patients[ip].patient_health==0)
	cout << "- ";
      else 
	cout << "X ";
    }
    cout << endl;
  };
};
    
int main() {
  cout << "How many patients? ";
  int npatients;
  cin >> npatients;

  ward sickward(npatients);
  sickward.print();

  for (int step=0; step<40; step++) {
    sickward.nexttime();
    sickward.print();
  }

  return 0;
}
