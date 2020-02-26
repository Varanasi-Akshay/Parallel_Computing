/****************************************************************
 ****
 **** This file belongs with the course
 **** Introduction to Scientific Programming in C++/Fortran2003
 **** copyright 2016 Victor Eijkhout eijkhout@tacc.utexas.edu
 ****
 **** ward.cxx : ward is a list of patients
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
  default_random_engine init_health;
public:
  ward(int npatients) {
    uniform_int_distribution<int> health_state(0,1);
    for (int ip=0; ip<npatients; ip++) {
      patients.push_back( patient(health_state(init_health)) );
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

  for (int ip=0; ip<npatients; ip++) {
    cout << "Patient " << ip << " is ";
    if (sickward.patients[ip].patient_health==0)
      cout << "healthy";
    else 
      cout << "sick";
    cout << endl;
  }

  sickward.print();

  return 0;
}
