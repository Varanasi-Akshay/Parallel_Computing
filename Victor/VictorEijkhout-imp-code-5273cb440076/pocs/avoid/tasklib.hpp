#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int set_options(int argc,char **argv,
		int &nlocal,int &latency,int &blocking,int &cores,int &nodes,int &over,
		int &steps,bool &dot,int &verbose) {
  nlocal = 1; cores = 1; latency = 100;
  for (int iarg=1; iarg<argc; iarg++) {
    if (std::string(argv[iarg])==std::string("-h")) {
      fmt::print("Usage: {} \n",argv[0]);
      fmt::print("      [ -steps s ] [ -blocking n ]\n");
      fmt::print("      [ -nodes n ] [ -over n ] [ -nlocal n ] [ -latency n ]\n");
      fmt::print("      [ -cores n ]\n");
      fmt::print("      [ -vV ] [ -d ]\n");
      fmt::print("================\n");
      return 1;
    }
    if (std::string(argv[iarg])==std::string("-blocking")) {
      iarg++; blocking  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-cores")) {
      iarg++; cores  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-nodes")) {
      iarg++; nodes  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-over")) {
      iarg++; over  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-steps")) {
      iarg++; steps  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-nlocal")) {
      iarg++; nlocal  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-latency")) {
      iarg++; latency  = std::stoi(argv[iarg]);
      continue;
    }
    if (std::string(argv[iarg])==std::string("-d")) {
      dot = true;
      continue;
    }
    if (std::string(argv[iarg])==std::string("-v")) {
      verbose = 1;
      continue;
    }
    if (std::string(argv[iarg])==std::string("-V")) {
      verbose = 2;
      continue;
    }
    fmt::print("Unrecognized argument {}: <<{}>>\n",iarg,argv[iarg]);
  }
  return 0;
};

class nodegraph; // forward declaration so that we can friend

#define PP(ip,jp,ndomains) (ip)*ndomains+jp
#define PPP(ip,jp,kp,ndomains) ((ip)*ndomains+jp)*ndomains+kp
#define NN(in,jn,nnodes) (in)*nnodes+jn
#define NNN(in,jn,kn,nnodes) ( (in)*nnodes+jn )*nnodes + kn

/*!
 * Distribution are stored inverted
 */
class distribution {
private:
  std::vector< int > assignment;
  int domain_side{0},node_side{0}, dimension{1};
public:
  distribution() {};
  //! For an overdecomposed block distribution set block ownership
  distribution(int nodes,int over,int dim=1) {
    node_side = nodes; domain_side = nodes*over; dimension = dim;
    if (dim==1) {
      assignment = std::vector<int>(domain_side);
      for (int in=0; in<nodes; in++) {
	for (int io=0; io<over; io++) {
	  int ip = in*over+io;
	  assignment.at(ip) = in;
	}
      }
    } else if (dim==2) {
      assignment = std::vector<int>(domain_side*domain_side);
      for (int in=0; in<nodes; in++) {
	for (int jn=0; jn<nodes; jn++) {
	  for (int io=0; io<over; io++) {
	    for (int jo=0; jo<over; jo++) {
	      int
		ip = in*over+io,
		jp = jn*over+jo;
	      assignment.at(PP(ip,jp,domain_side)) = NN(in,jn,node_side);
	    }
	  }
	}
      }
    } else if (dim==3) {
      assignment = std::vector<int>(domain_side*domain_side*domain_side);
      for (int in=0; in<nodes; in++) {
	for (int jn=0; jn<nodes; jn++) {
	  for (int kn=0; kn<nodes; kn++) {
	    for (int io=0; io<over; io++) {
	      for (int jo=0; jo<over; jo++) {
		for (int ko=0; ko<over; ko++) {
		  int
		    ip = in*over+io,
		    jp = jn*over+jo,
		    kp = kn*over+ko;
		  assignment.at(PPP(ip,jp,kp,domain_side)) = NNN(in,jn,kn,node_side);
		}
	      }
	    }
	  }
	}
      }
    } else
      throw(fmt::format("Can not define distribution in {}D",dim));
    fmt::print
      ("Created distribution of dimension {}:\n{} nodes, overdecomposed {}, total tasks {}\n",
       dim,nnodes(),over,nprocs());
  };
  int space_dimension() const { return dimension; };
  int get_assignment(int p) const {
    try {
      return assignment.at(p);
    } catch (...) { fmt::print("Error getting assignment for {} in {}\n",p,assignment.size());
      throw(-1);
    }
  };
  int nnodes() const { return assignment.at( nprocs()-1 )+1; };
  int nprocs() const { return assignment.size(); };
  int get_domain_side() const { return domain_side; };
  int get_node_side() const { return node_side; };
  void print() const {
    for (int p=0; p<assignment.size(); p++)
      std::cout << p << ":" << assignment.at(p) << " ";
    std::cout << std::endl;
  };
};

/*!
 * A task is a (step,proc) coordinate
 */
class task {
  friend class nodegraph;
  friend class taskbucket;
private:
  int proc{-1},step{-1},level{-1}; bool has_successor{false};
  std::vector< std::shared_ptr<task> > predecessors;
  bool is_duplicate{false};
public:
  std::string as_string() const { return fmt::format("({},{})@l{}",step,proc,level); };
  std::string label() const {
    if (is_duplicate)
      return fmt::format("S{}D{}d",step,proc);
    else
      return fmt::format("S{}D{}",step,proc);
  };
  std::string description() const { fmt::MemoryWriter w; w.write("{}<-[",label());
    for ( auto p : predecessors ) w.write("{},",p->label());
    w.write("] ");
    return w.str();
  };
  task(){};
  task(int s,int p) : step(s),proc(p) {};
  std::shared_ptr<task> duplicate() {
    auto d = std::shared_ptr<task>( new task( *this ) );
    d->set_duplicate();
    return d;
  };
  void add_predecessor( std::shared_ptr<task> t ) {
    predecessors.push_back(t);
    t->has_successor = true;
  };
  void set_duplicate() { is_duplicate = true; };
  //! Is a task local to a node, given a distribution?
  bool is_local_to_node(int n,const distribution &d) const {
    try {
      return n==d.get_assignment(proc);
    } catch (...) { fmt::print("Error testing on node {} of task on {}\n",n,proc); throw(1); }
  };
  bool can_be_executed_locally(int n,const distribution &d,bool trace=false) const {
    if (!is_local_to_node(n,d))
      return false;
    else if (predecessors.size()==0)
      return true;
    else {
      if (trace) fmt::print("Can {} be executed on {}? pred:",label(),n);
      for ( auto p : predecessors ) {
	bool can = p->can_be_executed_locally(n,d,false);
	if (!can) {
	  if (trace) fmt::print(" {}:NO\n",p->label());
	  return false;
	} else {
	  if (trace) fmt::print(" {}:yes",p->label());
	}
      } if (trace) fmt::print("\n");
      return true;
    }
  };
};

//! Disjoint vector of tasks
class taskbucket {
  friend class nodegraph;
  friend class parallel;
private:
  int node{-1};
  int target{-1}; //!< Only meaningful if you are a k1
  std::string name;
  std::vector< std::shared_ptr<task> > bucket;
public:
  bool contains( std::shared_ptr<task> t ) const {
    for ( auto t_already : bucket )
      if (t->proc==t_already->proc && t->step==t_already->step)
	return true;
    return false;
  };
  int contains_where( const std::shared_ptr<task> t ) const {
    int icount{0};
    for ( auto t_already : bucket )
      if (t->proc==t_already->proc && t->step==t_already->step)
	return icount;
      else icount++;
    return -1;
  };
  std::tuple<bool,std::shared_ptr<task>> find( const std::shared_ptr<task> t ) const {
    for ( auto t_already : bucket )
      if (t->proc==t_already->proc && t->step==t_already->step) {
	//fmt::print("Match k5 task <<{}>> to k3 <<{}>>\n",t->label(),t_already->label());
	return {true,t_already};
      }
    return {false,nullptr};
  };
  std::shared_ptr<task> &at(int i) {
    return bucket.at(i);
  };
  bool adding( std::shared_ptr<task> t ) {
    if (contains(t))
      return false;
    else {
      bucket.push_back(t); return true;
    }
  };
  void add_in_front( std::shared_ptr<task> t ) { bucket.insert(bucket.begin(),t); };
  int size() const { return bucket.size(); };
  void merge_in( taskbucket &other ) {
    for ( auto t : other.bucket )
      adding(t);
  };

  /***
   *** Level stuff
   ***/
private:
  int levels{-1};
public:
  void set_nlevels(int l) {
    if (levels!=-1) {
      fmt::print("Levels of {} was already set to {}, can not set to {}\n",name,levels,l);
      throw(1); };
    levels = l;
  }
  int nlevels() const {
    if (levels==-1) {
      fmt::print("Levels were not set for taskbucket\n"); throw(1); }
    if (levels==0 && bucket.size()>0) {
      fmt::print("Suspiciously zero levels for nonzero graph\n"); throw(1); }
    return levels;
  };
  /*! Set levels, with 1 as lowest possible, using `procs' at most per level
   */
  void set_levels(int procs=1) {
    // levels & how many tasks on each
    std::vector<int> level_count;
    // maximum assigned level
    int max_level{1};
    for ( int ipass=0 ; ; ipass++ ) {
      // if all tasks have a level at least 1, exit
      bool done{true};
      for ( auto t : bucket )
	done = done && t->level>=1; // or t->predecessors.size()==0 ?
      if (done) break;
      // go through the bucket once, and set levels
      bool progress{false};
      for ( auto t : bucket ) {
	if (t->level>=1) // already assigned
	  continue;
	// if all predecessors are either origin or assigned, assign this task a free level
	bool can_be{true};
	int free_level{1};
	for ( auto p : t->predecessors ) {
	  //fmt::print(" over {},", p->as_string());
	  can_be = can_be && (p->predecessors.size()==0 || p->level>=0);
	  if (p->level>=free_level)
	    free_level = p->level+1;
	}
	//fmt::print("; can be {} with {}\n",can_be,free_level);
	if (can_be) {
	  // if procs-per-level limited, maybe increase free level
	  if (procs>0) {
	    for ( ; ; ) {
	      while (free_level>=level_count.size()) // making a new level
		level_count.push_back(0);
	      if (level_count.at(free_level)<procs) { // still space on this level
		level_count.at(free_level)++; break; }
	      free_level++; // look at next level
	    }
	  }
	  // assign!
	  t->level = free_level; progress = true;
	  if (t->level>max_level) max_level = t->level;
	}
      }
      if (!progress) { fmt::print("Not making any progress\n"); throw(1); }
    }
    set_nlevels(max_level);
  };
  void set_parallel() {
    for ( auto t : bucket )
      t->level = 0;
    set_nlevels(1);
  };
  std::string as_string() const { fmt::MemoryWriter w;
    for (auto t : bucket) w.write("{}\n",t->description());
    return w.str();
  };
  std::string listing() const { fmt::MemoryWriter w;
    for (auto t : bucket) w.write("{}, ",t->label());
    return w.str();
  };
  void print() const { 
    fmt::print("Bucket \"{}\" on node {}, #={}: {}\n",
	       name,node,bucket.size(),as_string());
  };
};

/*!
 * A task graph is a bucket of tasks
 */
class nodegraph {
  friend class parallel;
private:
  int node{-1}; //!< neg for global graph, otherwise 
  distribution dist;
  taskbucket k0,k0fromothers,k1,k1fromothers,k2,k2base,k3,k3buffer,k4,k5;
  std::vector<taskbucket> k0s,k1s;
  taskbucket tasks;
public:
  nodegraph() {};
  nodegraph(distribution &d) {
    int nodes = d.nnodes();
    dist = d;
  };
  void set_node(int n) { node = n; tasks.node = n;
    k0.node = n; k0.name = std::string{"k0"}; k0fromothers.node = n;
    k1.node = n; k1.name = std::string{"k1"}; k1fromothers.node = n;
    k2.node = n; k2.name = std::string{"k2"};
    k3.node = n; k3.name = std::string{"k3"};
    k4.node = n; k5.node = n;
  };
  int get_node() const { return node; };
  std::function< const std::vector<nodegraph>&() > get_all_nodes;

  void add_task( std::shared_ptr<task> t ) { tasks.adding(t); };
  std::shared_ptr<task> find_task(int s,int p) {
    for ( auto t : tasks.bucket )
      if (t->step==s && t->proc==p)
	return t;
    throw(fmt::format("Can not find ({},{}) in graph on node {}",s,p,node));
  };
  const taskbucket get_tasks() { return tasks; };

  /*
   * k0 is inherited from the previous level: not actually computed,
   * and therefore 1. one level 2. zero time, 3. completely parallel
   */
  void build_k0() {
    if (node<0) throw(fmt::format("can only build k0 on specific node"));
    for ( auto t : tasks.bucket )
      if (t->predecessors.size()==0 && t->is_local_to_node(node,dist))
	k0.adding(t);
  };
  const std::vector<taskbucket> &get_k0s() const { return k0s; };
  const taskbucket &get_k0() const { return k0; };
  void set_k0fromothers( taskbucket k0 ) { k0fromothers = k0; };
  const taskbucket &get_k0fromothers() const { return k0fromothers; };
  void build_k0fromothers() {
    taskbucket k0other;
    for ( const auto &other : get_all_nodes() )
      for ( const auto &t : get_k5().bucket )
	if (other.get_k0().contains(t))
	  k0other.adding(t);
    set_k0fromothers( k0other );
  };

  /*
   * k1
   */
  const std::vector<taskbucket> &get_k1s() const { return k1s; };
  const taskbucket &get_k1() const { return k1; };
  /*! Elements that are in the halo of `other'
   * For now, this includes both elements that are in k0 and computed elements
   */
  std::tuple<taskbucket,taskbucket> build_k1( const nodegraph &other ) const {
    const auto other_k5 = other.get_k5();
    taskbucket k0other,k1other;
    k0other.target = other.get_node();
    k1other.target = other.get_node();
    for ( auto t : k4.bucket ) {
      if (other_k5.contains(t))
	if (get_k0().contains(t))
	  k0other.adding(t);
	else
	  k1other.adding(t);
    }
    return std::make_tuple(k0other,k1other);
  };
  //! Union of other's k1s.
  void set_k1fromothers( taskbucket k1 ) { k1fromothers = k1; };
  const taskbucket &get_k1fromothers() const { return k1fromothers; };
  //! Runtime where each task takes unit time.
  int k1_runtime(int nlocal) const {
    int dim = dist.space_dimension();
    // fmt::print("k1 levels={}, local time={}^{}={}\n",
    // 	       k1.nlevels(),nlocal,dim,std::pow(nlocal,dim));
    return k1.nlevels() * std::pow(nlocal,dim);
  };

  /*
   * k2
   */
  void build_k2() {
    for ( auto t : k4.bucket ) {
      if (k0.contains(t) || k1.contains(t))
	goto next_task;
      k2.adding(t);
    next_task: continue;
    }
  };
  const taskbucket &get_k2() const { return k2; };
  //! Runtime where each task takes unit time.
  int k2_runtime(int nlocal) const { int tim = 0;
    int dim = dist.space_dimension();
    return k2.nlevels() * std::pow(nlocal,dim);
  };

  /*
   * k3 : Tasks that will be executed after halo transfer, so that is
   * k5 (all mine) - k4 (my local) - k1s (halo done elsewhere)
   * Because of duplication we make a new task, and do that recursively
   */
  const taskbucket &get_k3() const { return k3; };
  void build_k3() {
    taskbucket replaced,replacements;
    for ( auto t : k5.bucket ) {
      if (get_k0().contains(t)
	  || k0fromothers.contains(t)
	  || k1fromothers.contains(t)
	  || get_k4().contains(t))
	continue;
      auto replaced_t = t->duplicate();
      for (int ip=0; ip<t->predecessors.size(); ip++) {
	auto pred = t->predecessors.at(ip);
	if (replaced.contains(pred)) {
	  int iloc = replaced.contains_where(pred);
	  replaced_t->predecessors.at(ip) = replacements.at(iloc);
	}
      }
      k3.adding(replaced_t);
      if (replaced.adding(t))
	replacements.adding(replaced_t);
    }
  };
  //! What are the tasks that send into our k3 area?
  void build_k3buffer() {
    for ( auto t : k3.bucket )
      for ( auto t_pred : t->predecessors )
	if (k1fromothers.contains(t_pred))
	  k3buffer.adding(t_pred);
  };
  int k3_transfertime(int nlocal,int latency) const {
    return k3buffer.size()*nlocal+latency;
  };
  int k3_runtime(int nlocal) const {
    int dim = dist.space_dimension();
    return k3.nlevels() * pow(nlocal,dim);
  };

  /*
   * k4
   */
  const taskbucket &get_k4() const { return k4; };
  //! Find the tasks that can be executed locally
  void build_k4() {
    if (node<0) throw(fmt::format("can only build k4 on specific node"));
    for ( auto t : tasks.bucket )
      if (t->can_be_executed_locally(node,dist))
	k4.adding(t);
  };

  /*!
   * k5: all my tasks with recursive predecessors.
   * The structure is to make sure they get added in layers.
   */
  void build_k5() { //( std::vector<taskbucket> other) {
    if (node<0) throw(fmt::format("can only build k5 on specific node"));
    taskbucket next_bucket;
    for ( auto t : tasks.bucket )
      if (!t->has_successor)
	next_bucket.adding(t);
    while (next_bucket.size()>0) {
      auto t = next_bucket.at(0);
      k5.add_in_front(t);
      next_bucket.bucket.erase( next_bucket.bucket.begin() );
      for ( auto p : t->predecessors )
	next_bucket.adding(p);
    }
  };
  const taskbucket &get_k5() const { return k5; };

  void print() {
    std::cout << "Graph on " << node << " has " << tasks.size() << " tasks" << std::endl;
    std::cout << "k4 can be executed locally: " << k4.size() << std::endl;
    std::cout << "k5 in extended domain: " << k5.size() << std::endl;
    std::cout << "Ordinary halo surplus: " << k5.size()-tasks.size() << std::endl;
    for (int in=0; in<k1s.size(); in++) {
      if (k1s.at(in).size()>0)
	std::cout << "k1 doing " << k1s.at(in).size() << " tasks before " << in << std::endl;
    }
    std::cout << "k2 strictly local: " << k2.size() << std::endl;
    std::cout << "k3 after halo transfer: " << k3.size() << std::endl;
    std::cout << "Extra tasks: " << k4.size()+k3.size()-tasks.size() << std::endl;
    std::cout << "tasks to receive from: " << k3buffer.size() << std::endl;
    std::cout << std::endl;
  };
  void dot() const {
    std::ofstream dotfile;
    if (node<0) {
      // global graph
      dotfile.open(fmt::format("nodegraph-g.dot"));
      dotfile << fmt::format("graph nodegraph ") << "{" << std::endl;
      for ( auto n : tasks.bucket )
	for ( auto p : n->predecessors )
	  dotfile << fmt::format("{} -- {}",p->label(),n->label()) << std::endl;
    } else {
      // node graph
      dotfile.open(fmt::format("nodegraph-{}.dot",node));
      dotfile << fmt::format("graph nodegraph{} ",node) << "{" << std::endl;
      // indicate k1 clusters
      for ( const auto &k : get_k1s() ) { int t = k.target;
	dotfile << "  subgraph cluster_" << t << " { label = \"k1_" << t << "\"" <<std::endl;
	for ( auto n : k.bucket )
	  dotfile << fmt::format("    {}",n->label()) << std::endl;
	dotfile << "}" << std::endl;
      }
      
      { // indicate k2 cluster
	dotfile << "  subgraph cluster_k2 { label = \"k2\"" <<std::endl;
	for ( auto n : k2.bucket )
	  dotfile << fmt::format("    {}",n->label()) << std::endl;
	dotfile << "}" << std::endl;
      }

      { // indicate k5 bits for me cluster
	for ( auto other : get_all_nodes() ) {
	  int other_number = other.get_node();
	  if (get_node()==other_number) continue;
	  dotfile << "  subgraph cluster_k5_" << other_number
		  << " { label = \"k5_" << other_number << "\"" <<std::endl;
	  for ( auto t : k5.bucket )
	    if (other.get_k0().contains(t) || other.get_k1().contains(t))
	      dotfile << fmt::format("    {}",t->label()) << std::endl;
	  dotfile << "}" << std::endl;
	}
      }
      //fmt::print("Node {}, k3={}\n",node,k3.as_string());
      //fmt::print("Node {}, k5={}\n",node,k5.as_string());
      for ( auto n : k5.bucket ) {
	auto find_n_local = tasks.find(n);
	auto find_n_in3 = k3.find(n);
	std::string nlabel;
	// if the k5 node is in k3 and not local
	if (std::get<0>(find_n_in3) && !std::get<0>(find_n_local))
	  nlabel = std::get<1>(find_n_in3)->label();
	else
	  nlabel = n->label();
	for ( auto p : n->predecessors ) {
	  auto find_p_local = tasks.find(p);
	  auto find_p_in3 = k3.find(p);
	  std::string plabel;
	  if (std::get<0>(find_p_in3) && !std::get<0>(find_p_local))
	    plabel = std::get<1>(find_p_in3)->label();
	  else
	    plabel = p->label();
	  dotfile << fmt::format("{} -- {}",plabel,nlabel) << std::endl;
	}
      }
    }
    dotfile << "}" << std::endl;
    dotfile.close();
  };

public:
  int execute(int steps=1,int nlocal=1,int latency=1000,int verbose=0) {
    if (verbose>0)
      fmt::print(".... executing node {} with nlocal={}\n",get_node(),nlocal);
    int t1 = k1_runtime(nlocal);
    int t2 = k2_runtime(nlocal);
    int s3 = k3_transfertime(nlocal,latency);
    int t3 = k3_runtime(nlocal);
    int t = t1 + std::max(t2,s3) + t3;
    if (verbose>0)
      fmt::print("     t = {} + max[{},{}] + {} = {}\n",t1,t2,s3,t3,t);
    return steps*t;
  };
};

class parallel {
private:
  std::vector< nodegraph > processor_graphs;
  nodegraph global_graph;
public:
  parallel(int nnodes,distribution blocked) {
    global_graph = nodegraph(blocked);
    for (int inode=0; inode<nnodes; inode++)
      processor_graphs.push_back( nodegraph(blocked) );
    for ( auto &p : processor_graphs )
      p.get_all_nodes =
	[this] () -> const std::vector<nodegraph>& { return this->processor_graphs; };
  };
  // lose this in time
  std::vector< nodegraph > &get_graphs() { return processor_graphs; };
  
  void graph_leveling( int cores,int verbose ) {
    if (verbose>0)
      fmt::print("%%%%\nSetting levels\n");
    for ( auto &g : processor_graphs )
      g.k0.set_parallel();
    for ( auto &g : processor_graphs )
      g.k1.set_levels(cores);
    for ( auto &g : processor_graphs )
      g.k2.set_levels(cores);
    for ( auto &g : processor_graphs )
      g.k3.set_levels(cores);
    if (verbose>0) {
      for ( auto &g : processor_graphs ) {
	fmt::print("Node {} on {} procs:\n",g.get_node(),cores);
	fmt::print("  k0: size={}, levels={}\n",g.k0.size(),g.k0.nlevels());
	fmt::print("  k1: size={}, levels={}\n",g.k1.size(),g.k1.nlevels());
	fmt::print("  k2: size={}, levels={}\n",g.k2.size(),g.k2.nlevels());
	fmt::print("  k3: size={}, levels={}\n",g.k3.size(),g.k3.nlevels());
      }
      fmt::print("---- levels\n");
    }
  };

  void graph_dotting( const distribution &blocked,bool dot ) {
    if (dot) {
      global_graph.dot();
      for (int in=0; in<blocked.nnodes(); in++) {
	processor_graphs.at(in).dot();
      }
    }
  };

  void graph_execution( int steps,int nlocal,int latency,int verbose ) {
    fmt::print("%%%% Executing:\n");
    int maxtime{0};
    for ( auto &g : processor_graphs ) {
      int t = g.execute(steps,nlocal,latency,verbose);
      if (t>maxtime) maxtime = t;
    }
    fmt::print("Parallel time: {}\n",maxtime);
    fmt::print("---- Executing.\n\n");
  };

  void graph_building( int verbose ) {
    int nnodes = processor_graphs.size();
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
	  fmt::print("Node {}, k4: {} nodes locally executable\n",g.get_node(),g.get_k4().size());
	if (verbose>1)
	  fmt::print("Node {}, k4={}\n",g.get_node(),g.get_k4().listing());
      }

      for ( auto &g : processor_graphs ) {
	g.build_k5();
	if (verbose)
	  fmt::print("Node {}, k5: {} nodes\n",g.get_node(),g.get_k5().size());
	if (verbose>1)
	  fmt::print("Node {}, k5={}\n",g.get_node(),g.get_k5().listing());
      }

      for ( auto &g : processor_graphs ) {
	g.build_k0fromothers();
	if (verbose)
	  fmt::print("remote k0: {} nodes contribute\n",g.get_k0fromothers().size());
      }

    } catch (std::string c) {
      throw( fmt::format("Error analyzing graph k4, k5: {}\n",c) );
    } catch (std::out_of_range o) {
      fmt::print("Error analyzing k4, k5\n"); throw(o);
    }
    
    try {
      // k1: one to benefit each neighbour
      for ( auto &g : processor_graphs ) {
	for ( const auto &h : processor_graphs ) {
	  if (g.get_node()==h.get_node()) continue;
	  auto k0k1 = g.build_k1(h);
	  auto k0o = std::get<0>(k0k1);
	  auto k1o = std::get<1>(k0k1);
	  if (k0o.size()>0)
	    g.k0s.push_back(k0o);
	  if (k1o.size()>0) {
	    g.k1s.push_back(k1o);
	    for ( auto t : k1o.bucket )
	      g.k1.adding(t);
	  }
	}
	if (verbose>0) { fmt::print("Node {}, ",g.get_node());
	  if (verbose==1) { fmt::print("halo for: k0[");
	    for ( auto k : g.get_k0s() )
	      if (k.size()>0)
		fmt::print(" {}",k.target);
	    fmt::print(" ] k1[");
	    for ( auto k : g.get_k1s() )
	      if (k.size()>0)
		fmt::print(" {}",k.target);
	    fmt::print(" ]\n");
	  }
	  if (verbose>1) {
	    for ( auto &h : processor_graphs ) {
	      int iother = h.get_node();
	      fmt::print("k1 for {}: {}; ",iother,g.get_k1s().at(iother).listing());
	    }
	    fmt::print("\n");
	    fmt::print("Node {}, k1={}\n",g.get_node(),g.get_k1().listing());
	  }
	}
      }

      // k1, as constructed by all neighbours
      for ( auto &g : processor_graphs ) {
	taskbucket k1s;
	for (int iother=0; iother<nnodes; iother++) {
	  if (g.get_node()==iother) continue;
	  for ( auto b : processor_graphs.at(iother).get_k1s() ) {
	    k1s.merge_in(b);
	  }
	}
	g.set_k1fromothers(k1s);
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
      throw(fmt::format("Error analyzing graph k1: {}\n",c));
    } catch (std::out_of_range o) {
      throw(fmt::format("Error out_of_range analyzing graph\n"));
    } catch (...) {
      throw(fmt::format("Error analyzing graph k1\n"));
    }

  };

  void make_1d(const distribution &blocked,int blocking)
  {
    int Nodes = blocked.nnodes(),ndomains = blocked.get_domain_side();
    for (int nn=0; nn<Nodes; nn++)
      processor_graphs.at(nn).set_node(nn);

    for (int is=0; is<=blocking; is++) {
      for (int ip=0; ip<ndomains; ip++) {
	auto t = std::shared_ptr<task>( new task(is,ip) );
	if (is>0) {
	  try {
	    t->add_predecessor( global_graph.find_task(is-1,ip) );
	    if (ip>0)
	      t->add_predecessor( global_graph.find_task(is-1,ip-1) );
	    if (ip<ndomains-1)
	      t->add_predecessor( global_graph.find_task(is-1,ip+1) );
	  } catch(...) { throw(fmt::format("Error adding predecessors\n")); }
	}
	// insert task in global graph and local graphs
	int in=-1; bool added{false};
	try {
	  global_graph.add_task(t);
	  for (in=0; in<Nodes; in++)
	    if (t->is_local_to_node(in,blocked)) {
	      processor_graphs.at(in).add_task(t); added = true; }
	} catch (...) { throw(fmt::format("Error inserting task in graph {}\n",in)); }
	if (!added) throw(fmt::format("Failed to insert task {},{}\n",is,ip));
      }
    }
  };

  void make_2d(const distribution &blocked,int blocking)
  {
    int ndomains = blocked.get_domain_side();

    {
      int nn=0;
      for ( auto &g : processor_graphs )
	g.set_node(nn++);
    }

    int
      node_side = blocked.get_node_side(),
      domain_side = blocked.get_domain_side();
    for (int is=0; is<=blocking; is++) {
      for (int ip=0; ip<ndomains; ip++) {
	for (int jp=0; jp<ndomains; jp++) {
	  int tcoord = PP(ip,jp,domain_side);
	  //fmt::print("Level {}, adding task ({},{})={}",is,ip,jp,tcoord);
	  auto t = std::shared_ptr<task>( new task(is,tcoord) );
	  if (is>0) {
	    try {
	      t->add_predecessor( global_graph.find_task(is-1,tcoord) );
	      if (ip>0) { tcoord = PP(ip-1,jp,domain_side);
		//fmt::print(", ({},{})={}",ip-1,jp,tcoord);
		t->add_predecessor( global_graph.find_task(is-1,tcoord) );
	      }
	      if (jp>0) { tcoord = PP(ip,jp-1,domain_side);
		//fmt::print(", ({},{})={}",ip,jp-1,tcoord);
		t->add_predecessor( global_graph.find_task(is-1,tcoord) );
	      }
	      if (ip<ndomains-1) { tcoord = PP(ip+1,jp,domain_side);
		//fmt::print(", ({},{})={}",ip+1,jp,tcoord);
		t->add_predecessor( global_graph.find_task(is-1,tcoord) );
	      }
	      if (jp<ndomains-1) { tcoord = PP(ip,jp+1,domain_side);
		//fmt::print(", ({},{})={}",ip,jp+1,tcoord);
		t->add_predecessor( global_graph.find_task(is-1,tcoord) );
	      }
	    } catch(...) { fmt::print("Error adding predecessors\n"); throw(0); }
	  }
	  //fmt::print("\n");
	  // insert task in global graph and local graphs
	  int in=-1;
	  try {
	    global_graph.add_task(t);
	    for (int nn=0; nn<blocked.nnodes(); nn++)
	      if (t->is_local_to_node(nn,blocked))
		processor_graphs.at(nn).add_task(t);
	  } catch (...) { fmt::print("Error inserting task in graph {}\n",in); throw(0); }
	}
      }
    }
  };
  void make_3d(const distribution &blocked,int blocking)
  {
    int ndomains = blocked.get_domain_side();

    {
      int nn=0;
      for ( auto &g : processor_graphs )
	g.set_node(nn++);
    }

    int
      node_side = blocked.get_node_side(),
      domain_side = blocked.get_domain_side();
    for (int ip=0; ip<ndomains; ip++) {
      for (int jp=0; jp<ndomains; jp++) {
	for (int kp=0; kp<ndomains; kp++) {
	  int tcoord = PPP(ip,jp,kp,domain_side);
	  auto t = std::shared_ptr<task>( new task(0,tcoord) );
	  for (int iq=0; iq<ip; iq++) {
	    for (int jq=0; jq<jp; jq++) {
	      for (int kq=0; kq<kp; kq++) {
		int scoord = PPP(iq,jq,kq,domain_side);
		t->add_predecessor( global_graph.find_task(0,scoord) );
	      }
	    }
	  }
	  global_graph.add_task(t);
	  for (int nn=0; nn<blocked.nnodes(); nn++)
	    if (t->is_local_to_node(nn,blocked))
	      processor_graphs.at(nn).add_task(t);
	}
      }
    }
  };
  
};
