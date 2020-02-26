  try {
    for ( auto &g : processor_graphs ) {
      g.build_k0();
      if (verbose)
	fmt::print("k0: {} nodes inherited on {}\n",g.get_k0().size(),g.get_node());
    }

    // k4 & k5
    for ( auto &g : processor_graphs ) {
      g.build_k4();
      if (verbose)
	fmt::print("k4: {} nodes locally executable on {}\n",g.get_k4().size(),g.get_node());
      if (verbose>1)
	fmt::print("Node {}, k4={}\n",g.get_node(),g.get_k4().listing());
    }

    for ( auto &g : processor_graphs ) {
      g.build_k5();
      if (verbose)
	fmt::print("k5: {} nodes contribute to {}\n",g.get_k5().size(),g.get_node());
      if (verbose>1)
	fmt::print("Node {}, k5={}\n",g.get_node(),g.get_k5().listing());
    }

  } catch (std::string c) {
    fmt::print("Error analyzing graph k4, k5: {}\n",c); return -1;
  } catch (std::out_of_range o) {
    fmt::print("Error analyzing k4, k5\n"); throw(o);
  }
    
  try {
    // k1: one to benefit each neighbour
    for ( auto &g : processor_graphs ) {
      for ( auto &h : processor_graphs ) {
	if (g.get_node()==h.get_node()) continue;
	g.build_k1(h.get_node(),h.get_k5());
      }
      if (verbose>1) { fmt::print("Node {}, ",g.get_node());
	for ( auto &h : processor_graphs ) {
	  int iother = h.get_node();
	  fmt::print("k1 for {}: {}; ",iother,g.get_k1s().at(iother).listing());
	}
	fmt::print("\n");
      }
      if (verbose>1)
	fmt::print("Node {}, k1={}\n",g.get_node(),g.get_k1().listing());
    }

    // k1, as constructed by all neighbours
    for ( auto &g : processor_graphs ) {
      taskbucket k1s;
      for (int iother=0; iother<blocked.nnodes(); iother++) {
	if (g.get_node()==iother) continue;
	for ( auto b : processor_graphs.at(iother).get_k1s() ) {
	  k1s.merge_in(b);
	}
      }
      g.set_k1fromother(k1s);
    }

    // k2
    for ( auto &g : processor_graphs ) {
      g.build_k2();
      if (verbose>1)
	fmt::print("Node {}, k2={}\n",g.get_node(),g.get_k2().listing());
    }

    // k3 & buffer
    for ( auto &g : processor_graphs ) {
      g.build_k3();
      g.build_k3buffer();
      if (verbose>1)
	fmt::print("Node {}, k3={}\n",g.get_node(),g.get_k3().listing());
    }    
  } catch (std::string c) {
    fmt::print("Error analyzing graph k1: {}\n",c); return -1;
  } catch (std::out_of_range o) {
    fmt::print("Error analyzing graph\n"); throw(o);
  } catch (...) {
    fmt::print("Error analyzing graph k1\n"); return -1;
  }

