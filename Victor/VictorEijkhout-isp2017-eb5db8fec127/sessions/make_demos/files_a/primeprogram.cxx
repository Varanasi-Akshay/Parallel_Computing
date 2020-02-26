#include <iostream>
using namespace std;

bool isprime(int n) {
  return n%2==0 || n%3==0 || n%5==0 || n%7==0;
}

int main() {
  int n;
  cin >> n;
  cout << "The number " << n << " is prime: " << isprime(n) << endl;
  return 0;
}
