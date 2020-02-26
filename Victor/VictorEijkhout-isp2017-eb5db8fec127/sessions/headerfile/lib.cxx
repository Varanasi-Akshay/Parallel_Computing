#include "lib.h"

PrimeGenerator::nextprime() {
  lastnumber += 2;
  return lastnumber;
};

bool isprime(int i) {
  return i%2!=0;
}
