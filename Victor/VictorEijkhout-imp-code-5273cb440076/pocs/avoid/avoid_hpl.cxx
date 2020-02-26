#include "fmt/format.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tasklib.hpp"

int main( int argc,char **argv) {
  int steps{1}, blocking{5}, nlocal{0}, latency{100};
  int nodes{3}, over{10}, cores{-1};
  bool dot{false}; int verbose{0};

  fmt::print("================\n");
  int ret = set_options(argc,argv,nlocal,latency,blocking,cores,nodes,over,steps,dot,verbose);
  if (ret!=0) return 1;
  if (blocking>=over) {
    fmt::print("Sorry, `over' has to be at least `blocking'\n"); return 1; }

  fmt::print("Running {} for {} steps, with blocking={}",argv[0],steps,blocking);
  if (cores>0)
    fmt::print(", on {} cores",cores);
  fmt::print(", local domain {} pts, latency {} ops",nlocal,latency);
  fmt::print("\n");

  /*
   * Initial disjoint distribution
   */
  distribution blocked(nodes,over,3);
  fmt::print("================\n\n");

  /*
   * Build a task graph over `blocking' steps;
   * both global graphs and one per node.
   */
  parallel square(blocked.nnodes(),blocked);

  try {
    square.make_3d(blocked,blocking);
  } catch (std::string c) {
    fmt::print("Error building graph: {}",c); return -1;
  } catch (std::out_of_range o) {
    fmt::print("Error building graph\n"); throw(o);
  }

  try {
    square.graph_building(verbose);
    square.graph_leveling(cores,verbose);
    square.graph_dotting(blocked,dot);
    square.graph_execution(steps,nlocal,latency,verbose);
  } catch ( std::string e ) {
    fmt::print("Abort: {}\n",e);
  } catch ( int e ) {
    fmt::print("Abort with code: {}\n",e);
  }

  return 0;
}
