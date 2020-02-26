#include <iostream>
using namespace std;

#include "lib.h"

int main() {
  PrimeGenerator sequence;
  cout << "5 is prime: " << isprime(5) << endl;
  cout << sequence.nextprime() << endl;

  return 0;
}
