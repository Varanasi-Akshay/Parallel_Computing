#include <iostream.h>
using namespace std;

// you save with ^X ^S, then
// get out by ^X ^C

// navigation : ^a ^e start/end of line
// ^p ^n : prev/next line
int main() {
  // ^k is kill, you can repeat that a number of times
  // then go elsewhere in your program, then:
  // ^y is yank-back : put what you killed
  if (something) {
    x=1;
    // moving about: ^f forward, ^b back
    // ESC f : word forward, ESC b : word back
    // ESC ^f : block forward
    // (note: control is a modifier key, ESC is not! touch and let go!)
  } else {
    y=2;
  }
  // mark one point with ^space
  // go somewhere else, then ^w : scoop up whole block
  // and then again ^y
  return 0;
}
