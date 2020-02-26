#include <iostream>
using namespace std;

int main() {
  // Find longest collatz sequence under 1000
  int longest_run=0;
  for (int start=1; start<1000; start++) {
    // find the length of the sequence that starts with `start'
    int current_length=0;
    int current_iteration=start;
    while (current_iteration!=1) {
      current_length++;
      // compute next iteration
      if (current_iteration%2==0)
	current_iteration /= 2;
      else
	current_iteration = 3*current_iteration+1;
    }
    // if this sequence is longer, remember
    if (current_length>longest_run) {
      cout << "Found a new max length @" << start << " of " << current_length << endl;
      longest_run = current_length;
    }
  }
  return 0;
}
