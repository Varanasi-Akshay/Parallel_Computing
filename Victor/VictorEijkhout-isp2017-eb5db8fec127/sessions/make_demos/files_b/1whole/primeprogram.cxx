#include <iostream>
using namespace std;

bool isprime(int n) {
  return n%3==0 || n%5==0;
}

int main() {
  int n;
  cin >> n;
  cout << n << " is prime : " << isprime(n) << endl;
  return 0;
}
