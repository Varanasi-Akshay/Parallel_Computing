/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_base.cxx: Implementations of the base classes
 ****
 ****************************************************************/

#include "utils.h"
#include "imp_base.h"

#include <climits>

/****
 **** Basics
 ****/

/****
 **** Architecture
 ****/

std::vector<int> make_endpoint(int d,int s) {
  std::vector<int> endpoint;
  for (int id=0; id<d; id++)
    endpoint.push_back(1);
  endpoint[0] = s;
  for (int id=0; id<d-1; id++) { // find the largest factor of endpoint[id] and put in id+1
    for (int f=(int)(sqrt(endpoint[id])); f>=2; f--) {
      if (endpoint[id]%f==0) {
	endpoint[id] /= f; endpoint[id+1] = f;
	break; // end factor finding loop, go to next dimension
      }
    }
  }
  for (int id=0; id<d; id++)
    endpoint[id]--;
  return endpoint;
};

/*!
  Return the original processor; for now that's always zero. 
  See \ref architecture::get_proc_endpoint
*/
processor_coordinate *architecture::get_proc_origin(int d) {
  return new processor_coordinate( std::vector<int>(d,0) );
};

//! Multi-d descriptor of number of processes. This is actually the highest proc number.
//! \todo find a much better way of deducing or setting a processor grid
processor_coordinate *architecture::get_proc_endpoint(int d) {
  int P = nprocs()*get_over_factor();
  return new processor_coordinate( make_endpoint(d,P) );
  processor_coordinate *coord;
  if (d==1) {
    coord = new processor_coordinate(1); coord->set(0,P-1);
  } else if (d==2) {
    coord = new processor_coordinate(2);
    int ntids_i,ntids_j;
    for (int n=sqrt(P+1); n>=1; n--)
      if (P%n==0) { // real grid otherwise
	ntids_i = P/n; ntids_j = n;
	coord->set(0,ntids_i-1); coord->set(1,ntids_j-1); goto found;
      }
    coord->set(0,P-1); coord->set(1,0); // pencil by default
  } else throw(fmt::format("Can not make proc endpoint of d={}",d));
 found:
  return coord;
};

//! The size layout is one more than the endpoint \todo return as non-reference copy
processor_coordinate *architecture::get_proc_layout(int dim) {
  processor_coordinate *layout = get_proc_endpoint(dim);
  for (int id=0; id<dim; id++)
    layout->set(id, layout->coord(id)+1 );
  return layout;
};

//! Set the collective strategy by value.
void architecture::set_collective_strategy(collective_strategy s) {
  strategy = s;
};

//! Set the collective strategy by number; great for commandline arguments.
void architecture::set_collective_strategy(int s) {
  switch (s) {
  case 1 : set_collective_strategy( collective_strategy::ALL_PTP ); break;
  case 2 : set_collective_strategy( collective_strategy::GROUP ); break;
  case 3 : set_collective_strategy( collective_strategy::RECURSIVE ); break;
  case 4 : set_collective_strategy( collective_strategy::MPI ); break;
  case 0 :
  default :
    set_collective_strategy( collective_strategy::ALL_PTP ); break;
  }
};

std::string architecture_type_as_string( architecture_type type ) {
  switch (type) {
  case architecture_type::UNDEFINED : return std::string("Undefined architecture"); break;
  case architecture_type::SHARED : return std::string("Shared memory"); break;
  case architecture_type::SPMD : return std::string("MPI architecture"); break;
  case architecture_type::ISLANDS : return std::string("MPI+OpenMP") ; break;
  default : throw(std::string("Very undefined architecture")); break;
  }
};

//! Base case for architecture summary.
std::string architecture::summary() { fmt::MemoryWriter w;
  w.write(get_name());
  w.write(", protocol: {}",protocol_as_string());
  w.write(", collectives: {}",strategy_as_string());
  if (get_split_execution()) w.write(", split execution");
  if (has_random_sourcing()) w.write(", source randomization");
  if (get_can_embed_in_beta()) w.write(", object embedding");
  if (get_can_message_overlap()) w.write(", messages post early");
  if (algorithm::do_optimize) w.write(", task graph optimized");
  return w.str();
};

/****
 **** Environment
 ****/

//! \todo set the trace parameters by function call
environment::environment(int argc,char **argv) {
  entity::set_env(this);
  set_command_line(argc,argv);
  set_name("imp");
  debug_level = iargument("d",0);
  message_tag_admin_threshold = 1000;
  strategy = iargument("collective",0);
  if (has_argument("optimize"))
    algorithm::do_optimize = true;
  if (has_argument("queue_summary"))
    algorithm::queue_trace_summary = 1;
  if (has_argument("matrix_view"))
    sparse_matrix::sparse_matrix_trace = 1;
  if (has_argument("progress"))
    entity::add_trace_level(trace_level::PROGRESS);
  if (has_argument("reduct"))
    entity::add_trace_level(trace_level::REDUCT);
};

//! Reporting and cleanup
environment::~environment() {
  print_summary();
  close_ir_outputfile();
  delete_environment(); // left over stuff from derived environments
};

/*!
  Print general options.
  This routine will be augmented by the mode-specific calls such as 
  \ref mpi_environment::print_options. These will also typically abort.
*/
void environment::print_options() {
  printf("General options:\n");
  printf("  -optimize : optimize task graph\n");
  printf("  -queue_summary : summary task queue after execution\n");
  printf("  -progress/reduct : trace progress / reductions\n");
  printf("  -collective n where n=0 (ptp) 1 (ptp) 2 (group) 3 (recursive) 4 (MPI)\n");
};

bool environment::has_argument(const char *name) {
  std::string strname{name};
  bool has = hasarg_from_argcv(name,nargs,the_args)
    || hasarg_from_internal(strname);
  // if (get_is_printing_environment())
  //   printf("arg <%s>:%d\n",name,has);
  return has;
};

int environment::iargument(const char *name,int vdef) {
  int r = hasarg_from_argcv(name,nargs,the_args);
  if (r) {
    int v =iarg_from_argcv(name,vdef,nargs,the_args);
    // if (get_is_printing_environment())
    //   printf("arg <%s>:%d\n",name,v);
    // get_architecture()->print_trace
    //   (fmt::format("Argument <<{}>> supplied as <<{}>>",name,v));
    return v;
  } else return vdef;
};

void environment::push_entity( entity *e ) {
  list_of_all_entities.push_back(e);
};

int environment::n_entities() const {
  return list_of_all_entities.size();
};

//! A long listing of the names of all defined entities.
void environment::list_all_entities() {
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::OBJECT) {
      object *o = dynamic_cast<object*>(ent);
      fmt::print("Object: {}\n",o->get_name());
    } else if (c==entity_cookie::KERNEL) {
      kernel *k = dynamic_cast<kernel*>(ent);
      fmt::print("Kernel: {}\n",k->get_name());
    } else if (c==entity_cookie::TASK) {
      kernel *k = dynamic_cast<kernel*>(ent);
      task *t = dynamic_cast<task*>(k);
      if (t!=nullptr)
      	fmt::print("Task: {}\n",t->get_name());
    } else if (c==entity_cookie::DISTRIBUTION) {
      distribution *k = dynamic_cast<distribution*>(ent);
      fmt::print("Distribution: {}\n",k->get_name());
    }
  };
};

//! Count the allocated space of all objects
double environment::get_allocated_space() {
  double allocated = 0.; int nobjects = 0;
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::OBJECT) {
      nobjects++;
      object *o = dynamic_cast<object*>(ent);
      if (o!=nullptr) {
	double s = (*e)->get_allocated_space();
	allocated += s;
      }
    }
  }
  return allreduce_d(allocated);
};

//! A quick summary of all defined entities.
result_tuple *environment::local_summarize_entities() {
  auto results = new result_tuple;
  int n_message = 0, n_object = 0, n_kernel = 0, n_task = 0, n_distribution = 0;
  double flopcount = 0., duration = 0., analysis = 0.,
    msg_volume = 0.;
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (0) {
    } else if (c==entity_cookie::DISTRIBUTION) {
      distribution *k = dynamic_cast<distribution*>(ent);
      n_distribution++;
    } else if (c==entity_cookie::KERNEL) {
      //kernel *k = dynamic_cast<kernel*>(ent);
      n_kernel++;
    } else if (c==entity_cookie::TASK) {
      //kernel *k = dynamic_cast<kernel*>(ent);
      n_task++;
    } else if (c==entity_cookie::MESSAGE) {
      message *m = dynamic_cast<message*>(ent);
      if (m!=nullptr && m->get_sendrecv_type()==message_type::SEND) {
	//fmt::print("Counting message <<{}>> times {}\n",m->as_string(),m->how_many_times);
	n_message += m->how_many_times;
	msg_volume += m->volume() * m->how_many_times;
      }
    } else if (c==entity_cookie::OBJECT) {
      object *o = dynamic_cast<object*>(ent);
      n_object++;
    } else if (c==entity_cookie::QUEUE) {
      algorithm *q = dynamic_cast<algorithm*>(ent);
      duration += q->execution_event.get_duration();
      analysis += q->analysis_event.get_duration();
      flopcount += q->get_flop_count();
    }
  };
  architecture *arch = get_architecture();
  std::get<RESULT_OBJECT>(*results) = n_object;
  std::get<RESULT_KERNEL>(*results) = n_kernel;
  std::get<RESULT_TASK>(*results) = n_task; // no reduce!
  std::get<RESULT_DISTRIBUTION>(*results) = n_distribution;
  std::get<RESULT_ALLOCATED>(*results) = get_allocated_space();
  std::get<RESULT_DURATION>(*results) = duration;
  std::get<RESULT_ANALYSIS>(*results) = analysis;
  //  printf("found local nmessages %d\n",n_message);
  std::get<RESULT_MESSAGE>(*results) = n_message; // reduce
  std::get<RESULT_WORDSENT>(*results) = msg_volume; // reduce_d
  std::get<RESULT_FLOPS>(*results) = flopcount; // reduce_d
  return results;
};

/*!
  Convert the result of \ref environment::summarize_entities to a string.
 */
std::string environment::summary_as_string( result_tuple *results ) {
  fmt::MemoryWriter w;
  w.write("Summary: ");
  w.write("#objects: {}",std::get<RESULT_OBJECT>(*results));
  w.write(", #kernels: {}",std::get<RESULT_KERNEL>(*results));
  w.write(", #tasks: {}",std::get<RESULT_TASK>(*results));
  w.write(", |space|={}",(float)std::get<RESULT_ALLOCATED>(*results));
  w.write(", analysis time={}",std::get<RESULT_ANALYSIS>(*results));
  w.write(", runtime={}",std::get<RESULT_DURATION>(*results));
  // w.write(", analysis time={:9.5e}",std::get<RESULT_ANALYSIS>(*results));
  // w.write(", runtime={:9.5e}",std::get<RESULT_DURATION>(*results));
  w.write(", #msg={}",std::get<RESULT_MESSAGE>(*results));
  w.write(", #words sent={:7.2e}",std::get<RESULT_WORDSENT>(*results));
  w.write(", flops={:7.2e}",std::get<RESULT_FLOPS>(*results));
  return w.str();
};

int environment::nmessages_sent( result_tuple *results ) {
  return std::get<RESULT_MESSAGE>(*results);
};

void environment::print_summary() {
  if (has_argument("summary")) {
    fmt::MemoryWriter w;
    w.write("Summary:\n");

    // summary architecture and settings
    w.write("\n{}",arch->summary());

    // summary of entities
    auto summary = mode_summarize_entities();
    std::string summary_string = summary_as_string(summary);
    w.write("\n{}",summary_string);

    if (get_is_printing_environment())
      fmt::print("{}\n",summary_string);
  }
};

std::string environment::as_string() {
  return get_architecture()->as_string();
};

void environment::kernels_to_dot_file() {
  FILE *dotfile; std::string s;
  dotfile = fopen(fmt::format("{}-kernels.dot",get_name()).data(),"w");
  s = kernels_as_dot_string();
  fprintf(dotfile,"%s\n",s.data());
  fclose(dotfile);
};

/*!
  Make a really long string of all the kernels, with dependencies.
  \todo the way we get the algorithm name is not very elegant. may algorithm_as_dot_string, then call this?
 */
std::string environment::kernels_as_dot_string() { fmt::MemoryWriter w;
  w.write("digraph G {}\n",'{');
  for ( auto e : list_of_all_entities ) {
    entity_cookie c = e->get_cookie();
    if (c==entity_cookie::QUEUE) {
      algorithm *a = dynamic_cast<algorithm*>(e);
      if (a!=nullptr) {
	w.write("  label=\"{}\";\n",a->get_name());
	w.write("  labelloc=t;\n");
      }
    }
  }
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::KERNEL) {
      kernel *k = dynamic_cast<kernel*>(ent);
      std::string outname = k->get_out_object()->get_name();
      //w.write("  \"{}\" -> \"{}\";\n",k->get_name(),outname);
      auto deps = k->get_dependencies();
      for (auto d=deps.begin(); d!=deps.end(); ++d) {
    	w.write("  \"{}\" -> \"{}\";\n",
    		(*d)->get_in_object()->get_name(),outname
    		);
      }
    }
  }
  w.write("{}\n",'}');
  return w.str();
};

//! The basic case for single processor;
//! see \ref mpi_environment::tasks_to_dot_file
void environment::tasks_to_dot_file() {
  fmt::MemoryWriter w;
  {
    w.write("digraph G {}\n",'{');
    std::string s = tasks_as_dot_string();
    w.write("{}\n",s.data());
    w.write("{}\n",'}');
  }
  FILE *dotfile; 
  dotfile = fopen(fmt::format("{}-tasks.dot",get_name()).data(),"w");
  fprintf(dotfile,"%s\n",w.str().data());
  fclose(dotfile);
};

std::string environment::tasks_as_dot_string() { fmt::MemoryWriter w;
  for (auto e=list_of_all_entities.begin(); e!=list_of_all_entities.end(); ++e) {
    entity_cookie c = (*e)->get_cookie();
    entity *ent = (entity*)(*e);
    if (c==entity_cookie::TASK) {
      kernel *k = dynamic_cast<kernel*>(ent);
      task *t = dynamic_cast<task*>(k);
      for ( auto d : t->get_predecessor_coordinates() ) { //=deps->begin(); d!=deps->end(); ++d) {
    	w.write("  \"{}-{}\" -> \"{}-{}\";\n",
    		d->get_step(),d->get_domain().as_string(),
    		t->get_step(),t->get_domain().as_string()
    		);
      }
    }
  }
  return w.str();
};

//! Print a line plus indentation to stdout or the output file.
void environment::print_line( std::string c ) {
  if (!get_is_printing_environment()) return;
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"%s%s\n",indentation->data(),c.data());
};

void environment::open_bracket() { this->print_line( (char*)"<<" ); };
void environment::close_bracket() { this->print_line( (char*)">>" ); };

void environment::print_to_file( std::string s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"%s%s\n",indentation->data(),s.c_str());
};

void environment::print_to_file( const char *s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"%s%s\n",indentation->data(),s);
};

void environment::print_to_file( int p,std::string s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"[p%d] %s%s\n",p,indentation->data(),s.c_str());
};

void environment::print_to_file( int p,const char *s ) {
  if (!get_is_printing_environment()) return;
  // stdout or the file
  FILE *f = stdout; if (ir_outputfile!=NULL) f = ir_outputfile;
  fprintf(f,"[p%d] %s%s\n",p,indentation->data(),s);
};

//! Open a new output file, and close old one if needed.
void environment::set_ir_outputfile( const char *nam ) {
  if (!get_is_printing_environment()) return;
  if (ir_outputfile!=nullptr) fclose(ir_outputfile);
  fmt::MemoryWriter w;
  w.write("{}.ir",nam);
  ir_outputfilename = w.str();
  ir_outputfile = fopen(ir_outputfilename.data(),"w");
};

void environment::register_execution_time(double t) {
  execution_times.push_back(t);
}

void environment::record_task_executed() {
  // the openmp version has the same, but with a critical section
  ntasks_executed++;
};

void environment::register_flops(double f) {
  flops += f;
}

/*!
  Just about anything created in IMP is an entity, meaning that it's
  going to be pushed in a a list in the environment.
 */
//! This is the constructor that everyone should use
entity::entity( entity_cookie c ) {
  //  fmt::print("entity {}\n",(int)c);
  env->push_entity(this); set_cookie(c);
};

entity::entity( entity *e, entity_cookie c ) {
  if (e==nullptr)
    throw(std::string("Failed upcast probably"));
  env->push_entity(this); set_cookie(c);
};

//! Store the environment in which entities will recorded. Hm.
void entity::set_env( environment *e ) {
  if (env!=nullptr)
    throw(std::string("Can not reset entity environment"));
  env = e;
};

std::string entity::cookie_as_string() const {
  switch (typecookie) {
  case entity_cookie::UNKNOWN :      return std::string("unknown"); break;
  case entity_cookie::DISTRIBUTION : return std::string("distribution"); break;
  case entity_cookie::KERNEL :       return std::string("kernel"); break;
  case entity_cookie::TASK :         return std::string("task"); break;
  case entity_cookie::MESSAGE :      return std::string("message"); break;
  case entity_cookie::OBJECT :       return std::string("object"); break;
  case entity_cookie::OPERATOR :     return std::string("operator"); break;
  case entity_cookie::QUEUE :        return std::string("queue"); break;
  };
  return std::string("!!!undefined!!!");
};

/****
 **** Decomposition
 ****/

//! Default decomposition uses all procs of the architecture in one-d manner.
decomposition::decomposition( architecture *arch )
  : decomposition( arch, processor_coordinate
		   ( std::vector<int>{arch->nprocs()*arch->get_over_factor()} ) ) {
};

/*!
  In multi-d the user needs to indicate how the domains are laid out.
  The processor coordinate is a size specification, to make it compatible with nprocs
*/
//snippet decompfromcoord
decomposition::decomposition( architecture *arch,processor_coordinate &sizes )
  : architecture(arch),
    entity(/* dynamic_cast<entity*>(arch),*/entity_cookie::DECOMPOSITION) {
  int dim = sizes.get_dimensionality();
  if (dim<=0)
    throw(std::string("Non-positive decomposition dimensionality"));
  domain_layout = sizes; //new processor_coordinate(sizes);
};
//snippet end
decomposition::decomposition( architecture *arch,processor_coordinate &&sizes )
  : architecture(arch),
    entity(/* dynamic_cast<entity*>(arch),*/entity_cookie::DECOMPOSITION) {
  int dim = sizes.get_dimensionality();
  if (dim<=0)
    throw(std::string("Non-positive decomposition dimensionality"));
  domain_layout = sizes; //new processor_coordinate(sizes);
};

//! Copy constructor
decomposition::decomposition(decomposition *d)
  : decomposition(dynamic_cast<architecture*>(d),d->domain_layout) {
  for (int im=0; im<d->mdomains.size(); im++)
    mdomains.push_back( new processor_coordinate( d->mdomains.at(im) ) );
  copy_embedded_decomposition(d); copy_decomp_factory(d);
};

//! Get dimensionality.
int decomposition::get_dimensionality() {
  int dim = domain_layout.get_dimensionality();
  return dim;
};

//! Number of domains
int decomposition::domains_volume() {
  int p = domain_layout.volume();
  return p;
};

//! Get dimensionality, which has to be the same as something else.
int decomposition::get_same_dimensionality( int d ) {
  if (d!=domain_layout.get_dimensionality())
    throw(fmt::format("Coordinate dimensionality mismatch decomposition:{} vs layout{}",
		      d,domain_layout.as_string()));
  return d;
};

//! Add a bunch of 1d local domains
void decomposition::add_domains( indexstruct *d ) {
  for ( auto i=d->first_index(); i<=d->last_index(); i++ ) {
    int ii = (int)i; // std::static_cast<int>(i);
    add_domain( processor_coordinate( std::vector<int>{ ii } ) );
  }
};

//! Add a multi-d local domain.
void decomposition::add_domain( processor_coordinate d ) {
  int dim = d.get_same_dimensionality( domain_layout.get_dimensionality() );
  mdomains.push_back(d); };

//! Get the multi-dimensional coordinate of a linear one. \todo check this calculation
//! http://stackoverflow.com/questions/10903149/how-do-i-compute-the-linear-index-of-a-3d-coordinate-and-vice-versa
processor_coordinate decomposition::coordinate_from_linear(int p) {
  int dim = domain_layout.get_dimensionality();
  if (dim<=0) throw(std::string("Zero dim layout"));
  processor_coordinate pp = new processor_coordinate(dim);
  for (int id=dim-1; id>=0; id--) {
    int dsize = domain_layout.coord(id);
    if (dsize==0)
      throw(fmt::format("weird layout <<{}>>",domain_layout.as_string()));
    pp.set(id,p%dsize); p = p/dsize;
  };
  return pp;
};

processor_coordinate &decomposition::get_origin_processor() {
  if (closecorner.get_dimensionality()==0) {
    int d = get_dimensionality();
    closecorner = processor_coordinate( std::vector<int>(d,0) );
  }
  return closecorner;
};
processor_coordinate &decomposition::get_farpoint_processor() {
  if (farcorner.get_dimensionality()==0) {
    int d = get_dimensionality();
    int P = domains_volume();
    farcorner = processor_coordinate( make_endpoint(d,P) );
  }
  return farcorner;
};

// processor_coordinate *decomposition::get_closecorner() {
//   return get_proc_origin(get_dimensionality());
// };
// processor_coordinate *decomposition::get_farcorner() {
//     return get_proc_endpoint(get_dimensionality());
// };

/*!
  We begin iteration by giving the first coordinate.
  We store the current iterate as a private processor_coordinate: `cur_coord'.
  Iteration is done C-style: the last coordinate varies quickest.

  Note: iterating is only defined for bricks.
*/
decomposition &decomposition::begin() {
  cur_coord = get_origin_processor();
  //fmt::print("decomp::begin: {}\n",cur_coord.as_string());
  return *this;
};

/*!
  Since we are iterating C-style (row-major),
  the iteration endpoint is like the first coordinate but with the 
  zero component increased.
  In row major this would be the first iterated point that is not in the brick.
*/
decomposition &decomposition::end() {
  cur_coord = processor_coordinate( get_origin_processor() );
  cur_coord.at(0) = get_farpoint_processor()[0]+1;
  //fmt::print("decomp::end: {}\n",cur_coord.as_string());
  return *this;
};

/*!
  Here's how to iterate: 
  - from last to first dimensions, find the dimension where you are not at the far edge
  - increase the coordinate in that dimension
  - all higher dimensions are reset to the first coordinate.
*/
void decomposition::operator++() {
  int dim = get_origin_processor().get_dimensionality();
  for (int id=dim-1; id>=0; id--) {
    if (cur_coord[id]<get_farpoint_processor()[id] || id==0) {
      cur_coord.at(id)++; break;
    } else
      cur_coord.at(id) = get_origin_processor()[id];
  }
};

bool decomposition::operator!=( decomposition &other ) {
  bool
    f = !(get_origin_processor()==other.get_origin_processor()),
    l = !(get_farpoint_processor()==other.get_farpoint_processor()),
    c =	!(cur_coord==other.cur_coord); // what does this test?
  // fmt::print("decomp::neq me={},{},{}, end={},{},{}\n",
  // 	     get_origin_processor().at(0),cur_coord.at(0),get_farpoint_processor().at(0),
  // 	     other.get_origin_processor().at(0),other.cur_coord.at(0),other.get_farpoint_processor().at(0)
  // 	     );
  return f || l || c;
};

bool decomposition::operator==( decomposition &other ) {
  bool
    f = get_origin_processor()==other.get_origin_processor(),
    l = get_farpoint_processor()==other.get_farpoint_processor(),
    c = cur_coord==other.cur_coord;
  // fmt::print("mult:eq {}@{} vs {}@{} : {}, {}, {}\n",
  // 	     as_string(),cur_coord.as_string(),
  // 	     other.as_string(),other.cur_coord.as_string(),
  // 	     f,l,c);
  return f && l && c;
};

processor_coordinate &decomposition::operator*() {
  //fmt::print("decomp::deref: {}\n",cur_coord.as_string());
  return cur_coord;
};

/****
 **** Parallel indexstructs
 ****/

//! Basic constructor with nd domains.
parallel_indexstruct::parallel_indexstruct(int nd) {
  processor_structures.reserve(nd);
  for (int is=0; is<nd; is++)
    processor_structures.push_back
      ( std::shared_ptr<indexstruct>( new unknown_indexstruct() ) );
};

//!< This assumes creation is complete. \todo remove this in favour of the next function?
int parallel_indexstruct::size() const {
  return processor_structures.size();
};

//! This sounds collective but it's really a local function.
bool parallel_indexstruct::is_known_globally() {
  for ( auto p : processor_structures )
    if (!p->is_known()) return false;
  return true;
}

/*! Query the number of domains in this parallel indexstruct.
  (In multi-d the parallel indexstruct is defined on fewer than global ndomains.)
  Note that this can be used before the actual structures are set.
*/
int parallel_indexstruct::pidx_domains_volume() const {
  int p = processor_structures.size();
  return p;
};

/*! Copy constructor. Only the type is copied literally; the structures are recreated.
  \todo we're not handling the local_structure case yet.
  \todo we really should not copy undefined structures
  \todo can we use the copy constructor of std::vector for the structures?
*/
parallel_indexstruct::parallel_indexstruct( parallel_indexstruct *other ) {
  int P = other->pidx_domains_volume();
  // processor_structures = new std::vector< std::shared_ptr<indexstruct> >;
  processor_structures.reserve(P);

  for (int p=0; p<P; p++) {
    try {
      std::shared_ptr<indexstruct> otherstruct = other->get_processor_structure(p);
      processor_structures.push_back( otherstruct ); //->make_clone() );
    }
    catch (std::string c) {fmt::print("Error <<{}>> in get_processor_structure for copy\n",c);
      throw( fmt::format("Could not get pstruct {} from <<{}>>",p,other->as_string()) ); }
    catch (...) { fmt::print("Unknown error setting proc struct for p={}\n",p); }
  }
  auto old_t = other->get_type();
  if (old_t==distribution_type::UNDEFINED)
    throw(fmt::format("Pidx to copy from has undefined type: {}",other->as_string()));
  set_type(old_t);
};

//! Equality test by comparing pstructs \todo this does not account for localstruct
int parallel_indexstruct::equals(parallel_indexstruct *s) const {
  int t = 1;
  for (int p=0; p<pidx_domains_volume(); p++) {
    t = t && this->get_processor_structure(p)->equals(s->get_processor_structure(p));
    if (!t) return 0;
  }
  return t;
};

void psizes_from_global( std::vector<index_int> &sizes,int P,index_int gsize) {
  index_int blocksize = (index_int)(gsize/P);
  // first set all sizes equal
  for (int i=0; i<P; ++i)
    sizes[i] = blocksize;
  // then spread the remainder
  for (int i=0; i<gsize-P*blocksize; ++i)
    sizes[i]++;
}

/*!
  Create from an indexstruct. This can for instance be used to make an
  OpenMP parallel structure on an MPI numa domain.

  \todo make sure that the indexstruct is the same everywhere
*/
void parallel_indexstruct::create_from_indexstruct( std::shared_ptr<indexstruct> ind ) {
  if (!ind->is_strided())
    throw(fmt::format("Can not create pidx from indexstruct: {}",ind->as_string()));
  int P = pidx_domains_volume(); int stride = ind->stride();
  if (P==0)
    throw(std::string("parallel indexstruct does not have #domains set"));
  int Nglobal = ind->last_index()-ind->first_index()+1;
  index_int usize = Nglobal/P; std::vector<index_int> sizes(P);
  psizes_from_global(sizes,P,Nglobal);
  // now set first/last
  index_int sum = ind->first_index();
  for (int p=0; p<P; p++) {
    index_int first=sum, next=sum+sizes[p];
    if (next<first) throw(std::string("suspicious local segment derived"));
    if (stride==1)
      processor_structures.at(p) = 
	std::shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    else
      processor_structures.at(p) = 
	std::shared_ptr<indexstruct>( new strided_indexstruct(first,next-1,stride) );
    sum = next;
  }
  set_type(distribution_type::GENERAL);
};

/*!
  Usually we create parallel structures with indices 0..N-1
*/
void parallel_indexstruct::create_from_global_size(index_int gsize) {
  create_from_indexstruct
    ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,gsize-1) ) );
  set_type(distribution_type::CONTIGUOUS);
};

/*!
  Create a parallel index structure where every processor has the same
  number of points, allocated consecutively.

  \todo how do we do the case of mpi localsize?
*/
void parallel_indexstruct::create_from_uniform_local_size(index_int lsize) {
  int P = pidx_domains_volume();
  // now set first/last
  index_int sum = 0;
  for (int p=0; p<P; p++) {
    index_int first=sum, next=sum+lsize;
    processor_structures.at(p) =
      std::shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    sum = next;
  }
  set_type(distribution_type::CONTIGUOUS);
};

void parallel_indexstruct::create_from_local_sizes( std::vector<index_int> lsize) {
  // now set first/last
  int P = pidx_domains_volume();
  if (P!=lsize.size())
    throw(fmt::format("Global ndomains={} vs supplied vector of sizes: {}",P,lsize.size()));
  index_int sum = 0;
  for (int p=0; p<P; p++) {
    index_int s = lsize.at(p), first = sum, next = sum+s;
    processor_structures.at(p) =
      std::shared_ptr<indexstruct>( new contiguous_indexstruct(first,next-1) );
    sum = next;
  }
  set_type(distribution_type::CONTIGUOUS);
};

void parallel_indexstruct::create_from_replicated_local_size(index_int lsize) {
  create_from_replicated_indexstruct
    ( std::shared_ptr<indexstruct>( new contiguous_indexstruct(0,lsize-1) ) );
  set_type(distribution_type::REPLICATED);
};

//! \todo can we use the same pointer everywhere?
void  parallel_indexstruct::create_from_replicated_indexstruct(std::shared_ptr<indexstruct> idx)
{
  // now set first/last
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    processor_structures.at(p) = idx; //std::shared_ptr<indexstruct>( idx.make_clone() );
  }
  set_type(distribution_type::REPLICATED);
};

//! Why is that neg-neg exception not perculating up in unittest_distribution:[40]?
void parallel_indexstruct::create_cyclic(index_int lsize,index_int gsize) {
  int P = pidx_domains_volume();
  if (gsize<0) {
    if (lsize<0) {
      fmt::print("Need lsize or gsize for cyclic, proceeding with lsize=1\n");
      lsize = 1;
    }
    gsize = P*lsize;
  } else if (lsize<0) lsize = gsize/P;

  if (P*lsize!=gsize) {
    fmt::print("Incompatible lsize {} vs gsize {}",lsize,gsize);
    gsize = P*lsize;
  }
  for (int p=0; p<P; p++) {
    processor_structures.at(p) =
      std::shared_ptr<indexstruct>( new strided_indexstruct(p,gsize-1,P) );
  }
  set_type(distribution_type::CYCLIC);
};

//! \todo get rid of that clone
void parallel_indexstruct::create_blockcyclic(index_int bs,index_int nb,index_int gsize) {
  if (nb==1) {
    create_cyclic(bs,gsize); return; }

  int P = pidx_domains_volume();
  if (gsize>=0) { printf("gsize ignored in blockcyclic\n");
  }
  index_int lsize = bs*nb;
  gsize = P*lsize;

  if (P*lsize!=gsize) {
    fmt::print("Incompatible lsize {} vs gsize {}",lsize,gsize);
    gsize = P*lsize;
  }
  for (int p=0; p<P; p++) {
    index_int proc_first = p*bs;
    composite_indexstruct *local = new composite_indexstruct();
    for (index_int ib=0; ib<nb; ib++) {
      index_int block_first = proc_first+ib*P*bs;
      local->push_back
	( std::shared_ptr<indexstruct>
	  ( new contiguous_indexstruct(block_first,block_first+bs-1) ) );
    }
    processor_structures.at(p) = std::shared_ptr<indexstruct>( local->make_clone() );
  }
  set_type(distribution_type::CYCLIC);
};

void parallel_indexstruct::create_from_explicit_indices(index_int *lens,index_int **sizes) {
  // this is the most general case. for now only for replicated scalars
  index_int nmax = 0;
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    processor_structures.at(p) =
      std::shared_ptr<indexstruct>( new indexed_indexstruct(lens[p],sizes[p]) );
    for (index_int i=0; i<lens[p]; i++)
      nmax = MAX(nmax,sizes[p][i]);
  }
  set_type(distribution_type::GENERAL);
};

void parallel_indexstruct::create_from_function
        ( index_int(*pf)(int p,index_int i),index_int nlocal ) {
  index_int nmax = 0;
  int P = pidx_domains_volume();
  for (int p=0; p<P; p++) {
    index_int *ind = new index_int[nlocal];
    for (index_int i=0; i<nlocal; i++) {
      index_int n = (*pf)(p,i); nmax = MAX(nmax,n);
      ind[i] = n;
    }
    processor_structures.at(p) =
      std::shared_ptr<indexstruct>( new indexed_indexstruct( nlocal,ind ) );
  }
  set_type(distribution_type::GENERAL);
};

//! \todo instead of passing in an object, pass object data?
void parallel_indexstruct::create_by_binning( object *o, double mn, double mx, int id ) {
  throw(std::string("fix the binning routine"));
  set_type(distribution_type::GENERAL);
};

/*!
  Get the structure of processor p.
  In many cases, every processor has global knowledge, so regardless the address space
  you can indeed ask about any processor.
  However, it is allowed to return nullptr; see the copy constructor.
  \todo ugly. we should really fix the copy constructor.
*/
std::shared_ptr<indexstruct> parallel_indexstruct::get_processor_structure(int p) const {
  if (local_structure!=nullptr) return local_structure;
  int P = pidx_domains_volume();
  if (p<0 || p>=P)
    throw(fmt::format("Requested processor {} out of range 0-{}",p,P));
  else if (p>processor_structures.size())
    throw(fmt::format("No processor {} in structure of size {}",p,processor_structures.size()));
  std::shared_ptr<indexstruct> pstruct = processor_structures.at(p);
  if (pstruct==nullptr)
    throw(fmt::format("Found null processor structure at linear {}",p));
  return pstruct;
};

/*! Set a processor structure; this redefines the type as fully general
  \todo I don't like this push back stuff. Create the whole vector and just set
*/
void parallel_indexstruct::set_processor_structure(int p,std::shared_ptr<indexstruct> pstruct) {
  if (p<0 || p>processor_structures.size())
    throw(fmt::format("Setting pstruct #{} outside structures bound 0-{}",
		      p,processor_structures.size()));
  if (p<processor_structures.size()) {
    processor_structures.at(p) = pstruct;
  } else {
    throw(std::string("I don' like this pushback stuff"));
    processor_structures.push_back( pstruct );
  }
  uncompute_internal_quantities();
};

//! \todo should we throw an exception if there is a local struct?
index_int parallel_indexstruct::first_index( int p ) const {
  return get_processor_structure(p)->first_index();
};
//! \todo should we throw an exception if there is a local struct?
index_int parallel_indexstruct::last_index( int p ) const {
  return get_processor_structure(p)->last_index();
};
/*!
  Size of the pth indexstruct in a parallel structure,
  unless there is a local structure.
 */
index_int parallel_indexstruct::local_size( int p ) {
  try {
    std::shared_ptr<indexstruct> localstruct;
    if (local_structure!=nullptr)
      localstruct = local_structure;
    else
      localstruct = get_processor_structure(p);
    if (localstruct==nullptr)
      throw(std::string("null local struct"));
    if (!localstruct->is_known())
      throw(fmt::format("Should not ask local size of unknown struct"));
    else 
      return localstruct->local_size();
  } catch (std::string c) {
    throw(fmt::format("Trouble parallel indexstruct local size: {}",c));
  }
};

//! The very first index in this parallel indexstruct. This assumes the processors are ordered.
index_int parallel_indexstruct::global_first_index() const {
  return first_index(0);
};

//! The very last index in this parallel indexstruct. This assumes the processors are ordered.
index_int parallel_indexstruct::global_last_index() const {
  index_int g;
  try { g = last_index(pidx_domains_volume()-1);
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> in global_last_index",c));
  };
};

/*!
  Return a single indexstruct that summarizes the structure in one dimension.
  \todo we should really test for gaps
*/
std::shared_ptr<indexstruct> parallel_indexstruct::get_enclosing_structure() const {
  try {
    index_int f = global_first_index(), l = global_last_index();
    fmt::print("pidx {} gives f={}, l={}\n",as_string(),f,l);
    return std::shared_ptr<indexstruct>( new contiguous_indexstruct(f,l) );
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> getting pidx enclosing",c));
  }
};

//! From very first to very last index.
index_int parallel_indexstruct::outer_size() const {
  return global_last_index()-global_first_index()+1;
};

/*!
  Return if a processor contains a certain index. This routine throws an
  exception if the index is globally invalid.
 */
int parallel_indexstruct::contains_element(int p,index_int i) const {
  // this catches out of bound indices
  if (!is_valid_index(i)) {
    throw(std::string("Index globally out of bounds"));
  }
  return get_processor_structure(p)->contains_element( i );
};

/*!
  Return the number of any processor that contains the requested index, 
  or throw an exception if not found. Searching starts with processor p0.
*/
int parallel_indexstruct::find_index(index_int ind,int p0) {
  int P = pidx_domains_volume();
  for (int pp=0; pp<P; pp++) {
    int p = (p0+pp)%P;
    if (contains_element(p,ind))
      return p;
  }
  throw(std::string("Index not found on any process"));
};

/*!
  Same as parallel_indexstruct::find_index but start searching with the first processor
*/
int parallel_indexstruct::find_index(index_int ind) {
  return find_index(ind,0);
};

/*!
  Try to detect the type of a parallel structure.
  Some of the components can be nullptr, in which case we return false.
 */
bool parallel_indexstruct::can_detect_type(distribution_type t) {
  int P = pidx_domains_volume();
  if (t==distribution_type::CONTIGUOUS) {
    index_int prev_last = LONG_MAX;
    bool first{true};
    for ( auto struc : processor_structures ) {
      if (!struc->is_contiguous()) return false;
      if (!first && struc->first_index()!=prev_last+1) return false;
      prev_last = struc->last_index();
      first = false;
    }
    return true;
  } else if (t==distribution_type::BLOCKED) {
    for ( auto struc : processor_structures ) {
      if (!struc->is_contiguous()) return false;
    }
    return true;
  } else return false;
};

//! \todo rewrite this for multi_structures
int parallel_structure::can_detect_type(distribution_type t) {
  if (is_orthogonal) {
    for (auto s : get_dimension_structures() )
      if (s==nullptr || !s->can_detect_type(t)) return 0;
    return 1;
  } else return 0;
};

/*!
  Detect contiguous and blocked (locally contiguous) distribution types
  and set the type parameter accordingly.
  Otherwise we keep whatever is there.
*/
distribution_type parallel_indexstruct::infer_distribution_type() {
  auto t = get_type();
  if (has_type_replicated()) return t;
  if (t!=distribution_type::CONTIGUOUS && can_detect_type(distribution_type::CONTIGUOUS)) {
    t = distribution_type::CONTIGUOUS;
  } else if (t!=distribution_type::BLOCKED && can_detect_type(distribution_type::BLOCKED)) {
    t = distribution_type::BLOCKED;
  } else
    t = distribution_type::GENERAL;
  set_type(t);
  return t;
}

//! \todo rewrite this for multi_structures
distribution_type parallel_structure::infer_distribution_type() {
  if (is_orthogonal) {
    auto t = distribution_type::UNDEFINED;
    for ( auto s : get_dimension_structures() )
      t = max_type(t,s->infer_distribution_type());
    if (t==distribution_type::UNDEFINED)
      throw(fmt::format
	    ("Inferring undefined for orthogonal parallel structure {}",as_string()));
    return t;
  } else
    return distribution_type::GENERAL;
};

//! Compute the more general of two types. Weed out the undefined case.
distribution_type max_type(distribution_type t1,distribution_type t2) {
  if (t1==distribution_type::UNDEFINED)
    return t2;
  else if (t2==distribution_type::UNDEFINED)
    return t1;
  else if ((int)t1<(int)t2)
    return t1;
  else return t2;
};

/*!
  Create a new parallel index structure by operating on this one.
  We do this by operating on each processor structure independently.

  \todo can we be more intelligently about how to set the type of the new structure?
  \todo write "operate_by" for indexstruct, then use copy constructor & op_by for this
 */
std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate( const ioperator &op) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  set_type(newstruct->infer_distribution_type());
  return newstruct;
};

std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate( const ioperator &&op) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  set_type(newstruct->infer_distribution_type());
  return newstruct;
};

std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate
    ( ioperator &op,std::shared_ptr<indexstruct> trunc ) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = processor_structures.at(p),
      out = in->operate(op,trunc)->force_simplify();
    newstruct->set_processor_structure(p,out);    
  }
  set_type(newstruct->infer_distribution_type());
  return newstruct;
};

//! \todo this should become the default one.
// std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate
//     ( std::shared_ptr<sigma_operator> op ) {
//   return operate(op.get());
// };

//! \todo we can really lose that star
//! \todo make domain_volume const, and then make this routine const
std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate( const sigma_operator &op ) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    auto
      in = get_processor_structure(p),
      out = in->operate(op)->force_simplify();
    newstruct->set_processor_structure(p,out);
  }
  set_type(newstruct->infer_distribution_type());
  return newstruct;
};

//! Operate, but keep the first index of each processor in place.
//! This probably doesn't make sense for shifting, but it's cool for multiplying and such.
std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate_base( const ioperator &op ) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate base on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    if (processor_structures.at(p)==nullptr) {
      throw(std::string("No index struct?")); }
    index_int newfirst = op.operate( processor_structures.at(p)->first_index() );
    ioperator back("shift",-newfirst), forth("shift",newfirst);
    auto
      t1struct = processor_structures.at(p)->operate( back ),
      t2struct = t1struct->operate(op)->force_simplify();
    newstruct->set_processor_structure( p, t2struct->operate( forth ) );
  }
  newstruct->set_type(this->type);
  return newstruct;
}

std::shared_ptr<parallel_indexstruct> parallel_indexstruct::operate_base( const ioperator &&op ) {
  auto newstruct = std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  if (local_structure) throw(std::string("Can not operate base on locally defined pstruct"));
  for (int p=0; p<pidx_domains_volume(); p++) {
    if (processor_structures.at(p)==nullptr) {
      throw(std::string("No index struct?")); }
    index_int newfirst = op.operate( processor_structures.at(p)->first_index() );
    ioperator back("shift",-newfirst), forth("shift",newfirst);
    auto
      t1struct = processor_structures.at(p)->operate( back ),
      t2struct = t1struct->operate(op)->force_simplify();
    newstruct->set_processor_structure( p, t2struct->operate( forth ) );
  }
  newstruct->set_type(this->type);
  return newstruct;
}

std::shared_ptr<parallel_indexstruct> parallel_indexstruct::struct_union
( std::shared_ptr<parallel_indexstruct> merge ) {
  auto newstruct =  std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct( this ) );
  for (int p=0; p<pidx_domains_volume(); p++) {
    newstruct->processor_structures.at(p)
      = std::shared_ptr<indexstruct>
      ( this->processor_structures.at(p)->struct_union
	(merge->processor_structures.at(p).get()) );
  }
  return newstruct;  
}

void parallel_indexstruct::extend_pstruct(int p,indexstruct *i) {
  std::shared_ptr<indexstruct>
    bef = processor_structures.at(p),
    ext = std::shared_ptr<indexstruct>( bef->struct_union(i) );
  //fmt::print("Extended <<{}>> to <<{}>>\n",bef->as_string(),ext->as_string());
  set_processor_structure(p,ext);
};

/****************
 **** Parallel structure
 ****************/

parallel_structure *parallel_structure::operate( const ioperator &op ) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  auto rstruct = new parallel_structure(decomp);

  auto orth = get_is_orthogonal();
  rstruct->set_is_orthogonal(orth);

  if (orth) {
    for (int is=0; is<get_dimensionality(); is++) {
      auto base_structure = get_dimension_structure(is);
      auto operated_structure = base_structure->operate(op);
      rstruct->set_dimension_structure(is,operated_structure);
    }
    rstruct->set_is_converted(false);
  } else {
    rstruct->set_is_converted(false);
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      auto base_structure = get_processor_structure(dcoord);
      auto operated_structure = base_structure->operate(op);
      auto simplified_structure = operated_structure->force_simplify();
      rstruct->set_processor_structure(dcoord,simplified_structure);
    }
  }
  
  if (op.is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( rstruct->infer_distribution_type() );

  return rstruct;
};

parallel_structure *parallel_structure::operate( const ioperator &&op ) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  auto rstruct = new parallel_structure(decomp);

  auto orth = get_is_orthogonal();
  rstruct->set_is_orthogonal(orth);

  if (orth) {
    for (int is=0; is<get_dimensionality(); is++) {
      auto base_structure = get_dimension_structure(is);
      auto operated_structure = base_structure->operate(op);
      rstruct->set_dimension_structure(is,operated_structure);
    }
    rstruct->set_is_converted(false);
  } else {
    rstruct->set_is_converted(false);
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      auto base_structure = get_processor_structure(dcoord);
      auto operated_structure = base_structure->operate(op);
      auto simplified_structure = operated_structure->force_simplify();
      rstruct->set_processor_structure(dcoord,simplified_structure);
    }
  }
  
  if (op.is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( rstruct->infer_distribution_type() );

  return rstruct;
};

//! \todo unify with the non-truncating version
parallel_structure *parallel_structure::operate
    ( ioperator &op,std::shared_ptr<multi_indexstruct> trunc ) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  auto rstruct = new parallel_structure(decomp);

  if (get_is_orthogonal()) {
    for (int id=0; id<get_dimensionality(); id++) {
      rstruct->set_dimension_structure
	(id,get_dimension_structure(id)->operate(op,trunc->get_component(id)));
    }
    rstruct->set_is_orthogonal();
    rstruct->set_is_converted(false);
    //    rstruct->convert_to_multi_structures();
  } else {
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct->set_processor_structure
	(dcoord,get_processor_structure(dcoord)->operate(op,trunc)->force_simplify());
    }
    rstruct->set_is_orthogonal(false);
    rstruct->set_is_converted(false);
  }
  
  if (op.is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( rstruct->infer_distribution_type() );

  return rstruct;
};

/*!
  Apply a multi operator to a multi structure by applying the components
  of the operator to the components of the structure.
  \todo can we collapse this with the single ioperator case?
*/
parallel_structure *parallel_structure::operate( multi_ioperator *op ) {
  int dim = get_same_dimensionality(op->get_dimensionality());
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *rstruct = new parallel_structure(decomp);

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct->set_dimension_structure
	(is,get_dimension_structure(is)->operate(op->get_operator(is)));
    }
    rstruct->set_is_orthogonal();
    rstruct->convert_to_multi_structures();
  } else {
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct->set_processor_structure
	(dcoord,
	 get_processor_structure(dcoord)->operate(op)->force_simplify());
    }
    rstruct->set_is_orthogonal(false);
    rstruct->set_is_converted(false);
  }
  
  if (op->is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( infer_distribution_type() );

  return rstruct;
};

//! \todo write unit test
parallel_structure *parallel_structure::operate(multi_sigma_operator *op) {
  int dim = get_same_dimensionality(op->get_dimensionality());
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *rstruct = new parallel_structure(decomp);

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct->set_dimension_structure
	(is,get_dimension_structure(is)->operate(op->get_operator(is)));
    }
    rstruct->set_is_orthogonal();
    rstruct->convert_to_multi_structures();
  } else {
    for (int ds=0; ds<multi_structures.size(); ds++) {
      auto dcoord = coordinate_from_linear(ds);
      rstruct->set_processor_structure
	(dcoord,
	 get_processor_structure(dcoord)->operate(op)->force_simplify());
    }
    rstruct->set_is_orthogonal(false);
    rstruct->set_is_converted(false);
  }
  
  if (op->is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( infer_distribution_type() );

  return rstruct;
};

//! Apply \ref parallel_indexstruct::operate_base to each dimension
parallel_structure *parallel_structure::operate_base( const ioperator &op ) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *rstruct = new parallel_structure(decomp);

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct->set_dimension_structure
	(is,get_dimension_structure(is)->operate_base(op));
    }
    rstruct->set_is_orthogonal();
    rstruct->convert_to_multi_structures();
  } else
    throw(std::string("Can not operate base unless orthogonal"));

  if (op.is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( infer_distribution_type() );

  return rstruct;
};

parallel_structure *parallel_structure::operate_base( const ioperator &&op ) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *rstruct = new parallel_structure(decomp);

  if (get_is_orthogonal()) {
    for (int is=0; is<get_dimensionality(); is++) {
      rstruct->set_dimension_structure
	(is,get_dimension_structure(is)->operate_base(op));
    }
    rstruct->set_is_orthogonal();
    rstruct->convert_to_multi_structures();
  } else
    throw(std::string("Can not operate base unless orthogonal"));

  if (op.is_shift_op())
    rstruct->set_type( get_type() );
  else
    rstruct->set_type( infer_distribution_type() );

  return rstruct;
};

/*! Merge two parallel structures by doing \ref parallel_indexstruct::struct_union
  on the dimension components.
  \todo this actually gives the convex hull of the union. good? bad?
*/
parallel_structure *parallel_structure::struct_union( parallel_structure *merge) {
  int dim = get_dimensionality();
  //  if (dim!=1) throw(std::string("need 1-d for par struct union"));
  parallel_structure *return_struct = new parallel_structure(this);
  for (int id=0; id<dim; id++)
    return_struct->set_dimension_structure
      ( id,get_dimension_structure(id)->struct_union( merge->get_dimension_structure(id) ) );
  return return_struct;
};

bool parallel_indexstruct::is_valid_index(index_int i) const {
  return (i>=first_index(0)) && (i<=last_index(pidx_domains_volume()-1));
};

bool parallel_structure::is_valid_index(domain_coordinate &i) {
  return (i>=global_first_index()) && (i<=global_last_index());
};

bool parallel_structure::is_valid_index(domain_coordinate &&i) {
  return (i>=global_first_index()) && (i<=global_last_index());
};

std::string parallel_structure::header_as_string() {
  return fmt::format("#doms={}, type={}, globally known={}",
		     domains_volume(),type_as_string(),is_known_globally());
};

std::string parallel_indexstruct::as_string() const {
  fmt::MemoryWriter w;
  for (int p=0; p<pidx_domains_volume(); p++)
    w.write(" {}:{}",
	    p,get_processor_structure(p)->as_string());
  w.write(" ]");
  return w.str();
};

std::string parallel_structure::as_string() {
  fmt::MemoryWriter w;
  w.write("{}: [",header_as_string() );
  if (get_is_converted()) {
    //w.write("converted (stuff missing), ");
    w.write("P={} : ",multi_structures.size());
    for (int is=0; is<multi_structures.size(); is++)
      w.write("<<p={}: {}>>,",is,multi_structures[is]->as_string());
  } else {
    w.write("unconverted (stuff missing), ");
    w.write("Dim={} : ",get_same_dimensionality(get_dimension_structures().size()));
    // for ( int d=0; d<get_dimension_structures().size(); d++ )
    //   w.write("d={} : {}",d,get_dimension_structure(d)->as_string());
  }
  w.write(" ]");
  return w.str();
};

/****
 **** Mask & coordinate
 ****/

//! Create an empty processor_coordinate object of given dimension
//snippet pcoorddim
processor_coordinate::processor_coordinate(int dim) {
  for (int id=0; id<dim; id++)
    coordinates.push_back(-1);
};
//snippet end

//! Create coordinate from linearized number, against decomposition \todo unnecessary?
processor_coordinate::processor_coordinate(int p,decomposition &dec)
  : processor_coordinate(dec.get_dimensionality()) {
  int dim = dec.get_dimensionality();
  auto pcoord = dec.coordinate_from_linear(p);
  for (int d=0; d<dim; d++)
    set(d,pcoord.coord(d));
};

//! Constructor from explicit vector of sizes
processor_coordinate::processor_coordinate( std::vector<int> dims )
  : processor_coordinate(dims.size()) {
  for (int id=0; id<dims.size(); id++)
    set(id,dims[id]);
};
processor_coordinate::processor_coordinate( std::vector<index_int> dims )
  : processor_coordinate(dims.size()) {
  for (int id=0; id<dims.size(); id++)
    set(id,dims[id]);
};

//! Copy constructor
processor_coordinate::processor_coordinate( processor_coordinate *other ) {
  for (int id=0; id<other->coordinates.size(); id++)
    coordinates.push_back( other->coordinates[id] ); };

//! Create from the dimensionality of a decomposition.
processor_coordinate_zero::processor_coordinate_zero(decomposition &d)
  :  processor_coordinate_zero(d.get_dimensionality()) {};

/*!
  Get the dimensionality by the size of the coordinate vector.
  The dimension zero case corresponds to the default constructor,
  which is used for processor coordinate objects stored in a decomposition object.
*/
int processor_coordinate::get_dimensionality() const {
  int s = coordinates.size();
  if (s<0)
    throw(std::string("Non-positive processor-coordinate dimensionality"));
  return s;
};

//! Get the dimensionality, and it should be the same as someone else's.
int processor_coordinate::get_same_dimensionality( int d ) const {
  int rd = get_dimensionality();
  if (rd!=d)
    throw(fmt::format("Non-conforming dimensionalities {} vs {}",rd,d));
  return rd;
};

void processor_coordinate::set(int d,int v) {
  if (d<0 || d>=get_dimensionality())
    throw(fmt::format("Can not set dimension {}",d));
  coordinates.at(d) = v;
};

//! Get one component of the coordinate.
int processor_coordinate::coord(int d) const {
  if (d<0 || d>=coordinates.size() )
    throw(fmt::format("dimension {} out of range for coordinate <<{}>>",d,as_string()));
  return coordinates.at(d);
};

/*! Compute volume by multiplying all coordinates
  \todo is this a bad name? what do we use it for?
*/
int processor_coordinate::volume() const {
  int r = 1;
  for (int id=0; id<coordinates.size(); id++) r *= coordinates[id]; return r;
};

//! Equality test
bool processor_coordinate::operator==( processor_coordinate &&other ) {
  int dim = get_same_dimensionality(other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)!=other.coord(id)) return false;
  return true;
};
bool processor_coordinate::operator==( processor_coordinate &other ) {
  int dim = get_same_dimensionality(other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)!=other.coord(id)) return false;
  return true;
};

//! Equality test by ref
// bool processor_coordinate::equals( processor_coordinate *other ) {
//   return equals(*other);
// };

bool processor_coordinate::equals( processor_coordinate &other ) {
  int dim = get_same_dimensionality(other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)!=other[id]) return 0;
  return 1;
};

bool processor_coordinate::equals( processor_coordinate &&other ) {
  int dim = get_same_dimensionality(other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)!=other[id]) return 0;
  return 1;
};

//! Equality to zero
bool processor_coordinate::is_zero() { auto z = 1;
  for ( auto c : coordinates ) z = z && c==0;
  return z;
};

bool processor_coordinate::operator>( processor_coordinate other ) {
  int dim = get_same_dimensionality( other.get_dimensionality());
  for (int id=0; id<dim; id++)
    if (coord(id)<=other.coord(id)) return false;
  return true;
};

bool processor_coordinate::operator>( index_int other) {
  int dim = get_dimensionality();
  for (int id=0; id<dim; id++)
    if (coord(id)<=other) return false;
  return true;
};

//! Operate plus with second coordinate an integer
processor_coordinate processor_coordinate::operator+( index_int iplus) {
  int dim = get_dimensionality();
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)+iplus);
  return pls;
};

//! Operate plus with second coordinate a coordinate
processor_coordinate processor_coordinate::operator+(processor_coordinate &cplus) {
  int dim = get_same_dimensionality(cplus.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)+cplus.coord(id));
  return pls;
};

//! Operate minus with second coordinate an integer
processor_coordinate processor_coordinate::operator-(index_int iminus) {
  int dim = get_dimensionality();
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)-iminus);
  return pls;
};

//! Operate minus with second coordinate a coordinate
processor_coordinate processor_coordinate::operator-(processor_coordinate &cminus) {
  int dim = get_same_dimensionality(cminus.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id,coord(id)-cminus.coord(id));
  return pls;
};

//! Module each component wrt a layout vector \todo needs unittest
processor_coordinate processor_coordinate::operator%(processor_coordinate modvec) {
  int dim = get_same_dimensionality(modvec.get_dimensionality());
  processor_coordinate pls(dim);
  for (int id=0; id<dim; id++)
    pls.set(id, coord(id)%modvec.coord(id));
  return pls;
};

//! Rotate a processor coordinate in a grid \todo needs unittest
processor_coordinate processor_coordinate::rotate( std::vector<int> v,processor_coordinate m) {
  int dim = get_same_dimensionality(v.size());
  auto pv = processor_coordinate(v);
  return ( (*this)+pv )%m;
};

/*
 * operations
 */
domain_coordinate processor_coordinate::operate( const ioperator &op ) {
  int dim = get_dimensionality();
  domain_coordinate opped(dim); // = new processor_coordinate(dim);
  for (int id=0; id<dim; id++)
    opped.set(id, op.operate(coord(id)) );
  return opped;
};

domain_coordinate processor_coordinate::operate( const ioperator &&op ) {
  int dim = get_dimensionality();
  domain_coordinate opped(dim); // = new processor_coordinate(dim);
  for (int id=0; id<dim; id++)
    opped.set(id, op.operate(coord(id)) );
  return opped;
};

//! Unary minus
processor_coordinate processor_coordinate::negate() {
  int dim = get_dimensionality();
  processor_coordinate n(dim);
  for (int id=0; id<dim; id++)
    n.set(id,-coord(id));
  return n;
};

// //! Get a linear number wrt a surrounding cube.
// int processor_coordinate::linearize( processor_coordinate *layout ) {
//   int dim = get_same_dimensionality(layout->get_dimensionality());
//   int s = coord(0);
//   for (int id=1; id<dim; id++)
//     s = s*layout->coord(id) + coord(id);
//   return s;
// };

/*!
  Get a linear number wrt a surrounding cube.
  \todo this gets called with farpoint. not the same as layout
*/
int processor_coordinate::linearize( const processor_coordinate &layout ) const {
  int dim = get_same_dimensionality(layout.get_dimensionality());
  int s = coord(0);
  for (int id=1; id<dim; id++)
    s = s*layout[id] + coord(id);
  return s;
};

//! Get a linear number wrt to the layout of a procstruct
int processor_coordinate::linearize( const decomposition *procstruct ) const {
  return linearize( procstruct->get_domain_layout() );
};

/*! Construct processor coordinate that is identical to self,
  but zero in the indicated dimension.
  The `farcorner' argument is not used, but specified for symmetry 
  with \ref processor_coordinate::right_face_proc.
  \todo this gets called with farpoint, should be origin
*/
processor_coordinate processor_coordinate::left_face_proc
    (int d,processor_coordinate &&farcorner) {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate left(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) left.set(id,coord(id));
    else left.set(id,0);
  return left;
};

processor_coordinate processor_coordinate::left_face_proc
    (int d,processor_coordinate &farcorner) {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate left(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) left.set(id,coord(id));
    else left.set(id,0);
  return left;
};

/*! Construct processor coordinate that is identical to self,
  but maximal in the indicated dimension.
  The maximality is given by the `farcorner' argument.
*/
processor_coordinate processor_coordinate::right_face_proc
    (int d,processor_coordinate &farcorner) {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate right(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) right.set(id,coord(id));
    else right.set(id,farcorner.coord(id));
  return right;
};
processor_coordinate processor_coordinate::right_face_proc
    (int d,processor_coordinate &&farcorner) {
  int dim = get_same_dimensionality( farcorner.get_dimensionality() );
  processor_coordinate right(dim);
  for (int id=0; id<dim; id++)
    if (id!=d) right.set(id,coord(id));
    else right.set(id,farcorner.coord(id));
  return right;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_left_face( decomposition *procstruct ) const {
  auto origin = procstruct->get_origin_processor();
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)==origin[id]) return true;
  return false;
  // processor_coordinate *farcorner = procstruct->get_farcorner();
  // for (int id=0; id<get_dimensionality(); id++)
  //   if (coord(id)==0) return true;
  // return false;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_right_face( decomposition *procstruct ) const {
  auto farcorner = procstruct->get_farpoint_processor();
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)==farcorner[id]) return true;
  return false;
  // processor_coordinate *farcorner = procstruct->get_farcorner();
  // for (int id=0; id<get_dimensionality(); id++)
  //   if (coord(id)==farcorner->coord(id)) return true;
  // return false;
};

//! Is this coordinate on any face of the processor brick? \todo farcorner by reference
bool processor_coordinate::is_on_face( decomposition *procstruct ) const {
  return is_on_left_face(procstruct) || is_on_right_face(procstruct);
};
bool processor_coordinate::is_on_face( std::shared_ptr<object> proc ) const {
  return is_on_face(proc.get());
};

//! Is this coordinate the origin?
bool processor_coordinate::is_null() const {
  for (int id=0; id<get_dimensionality(); id++)
    if (coord(id)!=0) return false;
  return true;
};

//! Multiply processor coordinate with index_int gives a \ref domain_coordinate
// domain_coordinate processor_coordinate::operator*(index_int i) {
//   int dim = get_dimensionality(); domain_coordinate c(dim);
//   for (int id=0; id<dim; id++)
//     c.set(id,coord(id)*i);
//   return c;
// };


// //! Subtract two processor coordinates. This is often applied to a farcorner.
// processor_coordinate processor_coordinate::operator-( processor_coordinate other ) {
//   int dim = get_dimensionality();
//   processor_coordinate m(dim);
//   for (int id=0; id<dim; id++)
//     m.set(id, coord(id)-other.coord(id) );
//   return m;
// };

std::string processor_coordinate::as_string() const {
  fmt::MemoryWriter w;
  w.write("P[");
  for ( int i=0; i<coordinates.size(); i++ )
    w.write("{},",coordinates.at(i));
  w.write("]");
  return w.str();
};

//! Mask constructor. Right now only for 1d and 2d
processor_mask::processor_mask(decomposition *d)
  : decomposition(*d),
    entity(dynamic_cast<entity*>(d),entity_cookie::MASK) {
  int dim = get_dimensionality(), np = domains_volume();
  included.reserve(np);
  for (int p=0; p<np; p++)
    included.push_back(Fuzz::NO);
};

//! Create a mask from a list of integers.
processor_mask::processor_mask( decomposition *d, std::vector<int> procs )
  : processor_mask(d) {
  for ( auto p : procs )
    included[p] = Fuzz::YES;
};

//! Copy constructor.
processor_mask::processor_mask( processor_mask& other )
  : decomposition(other),
    entity(entity_cookie::MASK) {
  throw(std::string("no processor mask copy constructor"));
}
//   int dim = get_dimensionality(), np = domains_volume(); auto P = d->get_farcorner();
//   //  if (dim==1) { int np  = P->coord(0)+1;
//     include1d = new Fuzz[np];
//     for (int p=0; p<P->coord(0); p++) include1d[p] = other.include1d[p];
//   // } else
//   //   throw(std::string("Can not copy mask in multi-d"));
// };

// ! create a mask with the first P processors added
processor_mask::processor_mask( decomposition *d,int P ) : processor_mask(d) {
  for (int p=0; p<P; p++)
    included[p] = Fuzz::YES;
};
//   if (get_dimensionality()!=1)
//     throw(std::string("Can not add linear procs to mask in multi-d"));
//   for (int p=0; p<P; p++) {
//     processor_coordinate *c = new processor_coordinate(1);
//     c->set(0,p); add(c);
//   };
// };

//! Add a processor to the mask
void processor_mask::add(processor_coordinate &p) {
  int plin = p.linearize(this);
  included[plin] = Fuzz::YES;
};

//! Render mask as list of integers. This is only used in \ref mpi_distribution::add_mask.
std::vector<int> processor_mask::get_includes() { std::vector<int> includes;
  throw(std::string("get includes is totally wrong"));
  // int dim = get_dimensionality();
  // if (dim==1) {
  //   for (int i=0; i<domains_volume(); i++)
  //     includes.push_back( include1d[i]==Fuzz::YES );
  //   return includes;
  // } else
  //   throw(std::string("Can not get includes in multi-d"));
};

//! Test alivesness of a process.
int processor_mask::lives_on(processor_coordinate &p) {
  int plin = p.linearize(this);
  return included[plin]==Fuzz::YES;
};

//! Remove a processor from the mask; this only makes sense for the constructor from P.
void processor_mask::remove(int p) {
  included[p] = Fuzz::NO;
};

/****
 **** Parallel structure
 ****/

//! Constructor.
parallel_structure::parallel_structure(decomposition *d)
  : decomposition(d),entity(entity_cookie::UNKNOWN) {
  allocate_structure();
};

//! Shortcut for one-dimensional structure
parallel_structure::parallel_structure
    (decomposition *d,std::shared_ptr<parallel_indexstruct> pidx)
  : parallel_structure(d) {
  if (d->get_dimensionality()>1)
    throw(std::string("One dimensional constructor only works in 1D"));
  set_dimension_structure(0,pidx);
};

//! Copy constructor.
//! \todo we shouldn't have to infer the pidx type: should have been set when p was created
//! \todo right now we reuse the pointers. that's dangerous
parallel_structure::parallel_structure(parallel_structure *p)
  : parallel_structure( dynamic_cast<decomposition*>(p) ) {
  if (p->has_type(distribution_type::UNDEFINED))
    throw(std::string("Can not copy undefined parallel structure"));

  int dim = get_dimensionality();

  if (p->is_orthogonal) {
    is_orthogonal = true;
    try {
      for (int id=0; id<dim; id++) {
	auto old_struct = p->get_dimension_structure(id);
	auto dim_struct =
	  std::shared_ptr<parallel_indexstruct>
	      ( new parallel_indexstruct( old_struct.get() ) );
	set_dimension_structure(id,dim_struct);
      }
    } catch (std::string c) {
      throw(fmt::format("Parallel structure by copying orthogonal: {}",c));
    }
  }
  if (p->is_converted) {
    is_converted = true;
    try {
      for (int is=0; is<domains_volume(); is++) {
	auto pcoord = p->coordinate_from_linear(is);
	auto p_structure = p->get_processor_structure(pcoord);
	set_processor_structure(pcoord,p_structure);
      }
    } catch (std::string c) {
      throw(fmt::format("Parallel structure by copying converted: {}",c));
    }
  }
  is_orthogonal = p->is_orthogonal; is_converted = p->is_converted;
  known_globally = p->known_globally;
  set_type(p->get_type());
  if (p->global_structure_is_locked())
    set_global_structure( p->get_global_structure());
};

/*!
  We create an array of \ref multi_indexstruct, one for each global domain.
  Translation from linear to multi-d goes through \ref decomposition::get_farcorner.
*/
void parallel_structure::allocate_structure() {
  decomposition *d = dynamic_cast<decomposition*>(this);
  if (d==nullptr) throw(std::string("Could not cast to decomposition"));
  int dim = d->get_dimensionality();

  // set unknown multi structures
  try {
    for (int id=0; id<d->domains_volume(); id++)
      multi_structures.push_back
	( std::shared_ptr<multi_indexstruct>( new unknown_multi_indexstruct(dim) ) );
  } catch (std::string c) {
    throw(fmt::format("Trouble creating multi structures: {}",c));
  }

  // set empty dimension structures
  try {
    for (int id=0; id<dim; id++)
      dimension_structures.push_back(nullptr);
    for (int id=0; id<dim; id++) {
      auto dimsize = d->get_size_of_dimension(id);
      auto dimstruct =
	std::shared_ptr<parallel_indexstruct>( new parallel_indexstruct(dimsize) );
      set_dimension_structure( id,dimstruct );
    }
  } catch (std::string c) {
    throw(fmt::format("Trouble creating dimension structures: {}",c));
  }
};

//! Initially creating the dimension structures.
void parallel_structure::push_dimension_structure(std::shared_ptr<parallel_indexstruct> pidx) {
  dimension_structures.push_back(pidx);
};

/*!
  Multi-d parallel structure can be set;
  this assumes the location in the \ref structure vector is already created.
*/
void parallel_structure::set_dimension_structure
    (int d,std::shared_ptr<parallel_indexstruct> pidx) {
  if (d<0 || d>=get_dimensionality())
    throw(fmt::format("Invalid dimension <<{}>> to set structure",d));
  dimension_structures.at(d) = pidx;
};

//! Get the parallel structure in a specific dimension.
std::shared_ptr<parallel_indexstruct> parallel_structure::get_dimension_structure(int d) {
  if (d<0 || d>=get_dimensionality())
    throw(fmt::format("Invalid dimension <<{}>> to request structure",d));
  auto rstruct = dimension_structures.at(d);
  if (rstruct==nullptr)
    throw(fmt::format("Parallel indexstruct in dimension <<{}>> uninitialized",d));
  return rstruct;
};

//! Get the multi_indexstruct of a multi-d processor coordinate
std::shared_ptr<multi_indexstruct> parallel_structure::get_processor_structure
    ( processor_coordinate &p ) {
  if (!get_is_converted()) {
    //fmt::print("{}: distr was not converted: converting.\n",p.as_string());
    convert_to_multi_structures();
    //fmt::print("done\n");
  }
  decomposition *d = dynamic_cast<decomposition*>(this);
  if (d==nullptr)
    throw(fmt::format("Could not upcast to decomposition"));
  auto layout = d->get_domain_layout();
  int plinear = p.linearize(layout);
  if (plinear>=multi_structures.size())
    throw(fmt::format("Coordinate {} linear {} out of bound {} for <<{}>>",
		      p.as_string(),plinear,multi_structures.size(),layout.as_string()));
  auto rstruct = multi_structures.at(plinear);
  if (rstruct==nullptr)
    throw(fmt::format("Found null processor structure at linear {}",p.as_string()));
  return rstruct;
};

std::shared_ptr<multi_indexstruct> parallel_structure::get_processor_structure
    ( processor_coordinate *p ) {
  return get_processor_structure(*p);
};

/*!
  Set a processor structure, general multi-d case.
  But if this is one-dimensional we work on the dimension structure
  so that we maintain orthogonality.

  \todo can we be more clever about setting taht structure type?
*/
void parallel_structure::set_processor_structure
    ( processor_coordinate &p,std::shared_ptr<multi_indexstruct> pstruct) {
  if (!get_is_converted())
    convert_to_multi_structures();
  decomposition *d = dynamic_cast<decomposition*>(this);
  auto layout = d->get_domain_layout();
  int plinear = p.linearize(layout);
  multi_structures.at(plinear) = pstruct;
  set_is_converted(); set_is_orthogonal(false); 

  set_type( infer_distribution_type() );
  enclosing_structure_is_computed = memoization_status::UNSET;
  return;
};

void parallel_structure::set_processor_structure
    ( processor_coordinate *p,std::shared_ptr<multi_indexstruct> pstruct) {
  set_processor_structure(*p,pstruct);
};

//! Set a processor structure, shortcut for one-d. \todo dstructure only set if converted?
void parallel_structure::set_processor_structure(int p,std::shared_ptr<indexstruct> pstruct) {
  if (get_dimensionality()>1)
    throw(std::string("Can not set unqualified pstruct in multi-d pidx"));
  get_dimension_structure(0)->set_processor_structure(p,pstruct);
  multi_structures.at(p) = std::shared_ptr<multi_indexstruct>( new multi_indexstruct(pstruct) );
  set_type( distribution_type::GENERAL );
};

/*
 * Creation
 */
//! Create from global size
void parallel_structure::create_from_global_size(index_int gsize) {
  get_dimension_structure(0)->create_from_global_size(gsize);
  set_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
};
//! Create from global size, multi-d
void parallel_structure::create_from_global_size(std::vector<index_int> gsizes) {
  for (int id=0; id<gsizes.size(); id++)
    get_dimension_structure(id)->create_from_global_size(gsizes[id]);
  set_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
};

//! Create from indexstruct
void parallel_structure::create_from_indexstruct( std::shared_ptr<indexstruct> idx) {
  get_dimension_structure(0)->create_from_indexstruct(idx);
  set_type( distribution_type::BLOCKED );
  //  convert_to_multi_structures(); // VLE this was at one point needed for Lulesh, it seems
};

//! Create from indexstruct, multi-d
void parallel_structure::create_from_indexstruct(multi_indexstruct *idx) {
  int dim = get_same_dimensionality( idx->get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_indexstruct(idx->get_component(id));
  set_type( distribution_type::BLOCKED );
  set_is_known_globally();
};

//! Create from indexstruct, multi-d
void parallel_structure::create_from_indexstruct( std::shared_ptr<multi_indexstruct> idx) {
  int dim = get_same_dimensionality( idx->get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_indexstruct(idx->get_component(id));
  set_type( distribution_type::BLOCKED );
  set_is_known_globally();
};

//! Create from replicated indexstruct
void parallel_structure::create_from_replicated_indexstruct( std::shared_ptr<indexstruct> idx) {
  get_dimension_structure(0)->create_from_replicated_indexstruct(idx);
  set_type( distribution_type::REPLICATED );
  set_is_known_globally();
}
//! Create from replicated indexstruct, multi-d
void parallel_structure::create_from_replicated_indexstruct(multi_indexstruct *idx) {
  int dim = get_same_dimensionality( idx->get_dimensionality() );
  for (int id=0; id<dim; id++)
    get_dimension_structure(id)->create_from_replicated_indexstruct
      (idx->get_component(id));
  set_type( distribution_type::REPLICATED );
  set_is_known_globally();
};

void parallel_structure::create_from_uniform_local_size(index_int lsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_uniform_local_size(lsize);
  set_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
};
void parallel_structure::create_from_local_sizes( std::vector<index_int> szs ) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_local_sizes(szs);
  set_type( distribution_type::CONTIGUOUS );
  set_is_known_globally();
  //fmt::print("created from local sizes, known: {}\n",is_known_globally());
};
void parallel_structure::create_from_replicated_local_size(index_int lsize) {
  get_same_dimensionality(1);
  get_dimension_structure(0)->create_from_replicated_local_size(lsize);
  set_type( distribution_type::REPLICATED );
  set_is_known_globally();
}

void parallel_structure::create_cyclic(index_int lsize,index_int gsize) {
  get_dimension_structure(0)->create_cyclic(lsize,gsize);
  set_type( distribution_type::CYCLIC );
  convert_to_multi_structures();
  set_is_known_globally();
};
void parallel_structure::create_blockcyclic(index_int bs,index_int nb,index_int gsize) {
  get_dimension_structure(0)->create_blockcyclic(bs,nb,gsize);
  set_type( distribution_type::GENERAL);
  convert_to_multi_structures();
};
void parallel_structure::create_from_explicit_indices(index_int *nidx,index_int **idx) {
  get_dimension_structure(0)->create_from_explicit_indices(nidx,idx);
  set_type( distribution_type::GENERAL);
  convert_to_multi_structures();
};
void parallel_structure::create_from_function( index_int(*f)(int,index_int),index_int n) {
  // from p,i
  get_dimension_structure(0)->create_from_function(f,n);
  set_type( distribution_type::GENERAL);
  convert_to_multi_structures();
};

//! \todo add explicit bins to the create call; min and max become kernels
void parallel_structure::create_by_binning(object *o) {
  if (get_dimensionality()>1) throw(std::string("Can not bin in more than one d"));
  double mn = o->get_min_value(),mx = o->get_max_value();
  get_dimension_structure(0)->create_by_binning(o,mn-0.5,mx+0.5,0);
  set_type( distribution_type::GENERAL );
  convert_to_multi_structures();
};

//! Make \ref multi_structures from an orthogonally specified structure.
//! We don't do this if the structure is no longer orthgonal
void parallel_structure::convert_to_multi_structures() {
  if (is_converted) return;
  int dim = get_dimensionality();
  try {
    if (dim==1) {
      auto istruct = get_dimension_structure(0);
      int iprocs = istruct->size();
      if (multi_structures.size()<iprocs)
	throw(fmt::format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs));
      for (int ip=0; ip<iprocs; ip++)
	multi_structures.at(ip) = std::shared_ptr<multi_indexstruct>
	  ( new multi_indexstruct
	    ( istruct->get_processor_structure(ip) ) );
    } else if (dim==2) {
      auto 
	istruct = get_dimension_structure(0),
	jstruct = get_dimension_structure(1);
      int iprocs = istruct->size(), jprocs = jstruct->size();
      if (multi_structures.size()<iprocs*jprocs)
	throw(fmt::format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs*jprocs));
      int is=0;
      for (int ip=0; ip<iprocs; ip++)
	for (int jp=0; jp<jprocs; jp++)
	  multi_structures.at(is++) = std::shared_ptr<multi_indexstruct>
	    ( new multi_indexstruct
	      ( std::vector<std::shared_ptr<indexstruct>>{
		istruct->get_processor_structure(ip),jstruct->get_processor_structure(jp)
		  } ) );
    } else if (dim==3) {
      auto
	istruct = get_dimension_structure(0),
	jstruct = get_dimension_structure(1),
	kstruct = get_dimension_structure(2);
      int iprocs = istruct->size(), jprocs = jstruct->size(), kprocs = kstruct->size();
      if (multi_structures.size()<iprocs*jprocs*kprocs)
	throw(fmt::format("Insufficient multi_structures {} for dimension size {}",
			  multi_structures.size(),iprocs*jprocs*kprocs));
      int is=0;
      for (int ip=0; ip<iprocs; ip++)
	for (int jp=0; jp<jprocs; jp++)
	  for (int kp=0; kp<kprocs; kp++)
	    multi_structures.at(is++) = std::shared_ptr<multi_indexstruct>
	      ( new multi_indexstruct
		( std::vector<std::shared_ptr<indexstruct>>{
		  istruct->get_processor_structure(ip),
		    jstruct->get_processor_structure(jp),
		    kstruct->get_processor_structure(kp)
		    } ) );
    } else
      throw(std::string("Can not convert to multi_structures in dim>3"));
  } catch (std::string c) { fmt::print("Error <<{}>> converting\n",c);
    throw(std::string("Could not convert to multi_structures"));
  } catch (...) { fmt::print("Unknown error converting\n");
    throw(std::string("Could not convert to multi_structures"));
  }
  is_converted = true; is_orthogonal = true;
};

//! The vector of local sizes of a processor
domain_coordinate &parallel_structure::local_size_r(processor_coordinate p) {
  try {
    return get_processor_structure(p)->local_size_r();
  } catch (std::string c) {
    throw(fmt::format("Trouble parallel structure local size: {}",c));
  }
};

//! Global size as a \ref processor_coordinate
domain_coordinate &parallel_structure::global_size() {
  try {
    return get_enclosing_structure()->local_size_r();
  } catch (std::string c) {
    throw(fmt::format("Trouble parallel structure global size: {}",c));
  }
};

// //! Local number of points in the structure
// index_int parallel_structure::volume( processor_coordinate *p ) {
//   return volume(*p);
// };

/*!
  Local number of points in the structure
*/
index_int parallel_structure::volume( processor_coordinate &&p ) {
  try {
    auto struc = get_processor_structure(p);
    index_int
      vol = struc->volume();
    if (vol<0)
      throw(fmt::format("Negative volume for p={}",p.as_string()));
    return vol;
  } catch ( std::string c ) {
    throw(fmt::format("Problem parallel structure volume: {}",c));
  }
};

index_int parallel_structure::volume( processor_coordinate &p ) {
  try {
    auto struc = get_processor_structure(p);
    index_int
      vol = struc->volume();
    if (vol<0)
      throw(fmt::format("Negative volume for p={}",p.as_string()));
    return vol;
  } catch ( std::string c ) {
    throw(fmt::format("Problem parallel structure volume: {}",c));
  }
};

//! Total number of points in the structure
//! \todo memo'ize this number
//! \todo make shared two coordinates and one struct
index_int parallel_structure::global_volume() {
  auto enclosing = get_enclosing_structure();
  return enclosing->volume();
};

/****
 **** Distribution
 ****/

//snippet distributiondef
//! The default constructor does not set the parallel_indexstruct objects:
//! that's done in the derived distributions.
distribution::distribution(decomposition *d)
  : parallel_structure(d),
    entity(entity_cookie::DISTRIBUTION) {
  set_name("some-distribution"); set_dist_factory();
  int np;
  try { np = d->domains_volume();
  } catch (std::string c) {
    throw(fmt::format("decomposition volume problem: {}",c));
  }
  linear_sizes = std::vector<int>(np);
  linear_starts = std::vector<int>(np);
  linear_offsets = std::vector<int>(np);
};
//snippet end

//! Constructor from parallel structure
distribution::distribution(parallel_structure *struc)
  : parallel_structure(struc),
    entity(entity_cookie::DISTRIBUTION) {
  set_name(fmt::format("distribution-from-{}",struc->get_name())); set_dist_factory();
  int np;
  try { np = domains_volume();
  } catch (std::string c) {
    throw(fmt::format("decomposition volume problem: {}",c));
  }
  linear_sizes = std::vector<int>(np);
  linear_starts = std::vector<int>(np);
  linear_offsets = std::vector<int>(np);
};

//! Constructor from explicitly specified parallel_indexstruct.
distribution::distribution(decomposition *d,std::shared_ptr<parallel_indexstruct> struc)
  : distribution(d) {
  set_dimension_structure(0,struc);
};

//! Copy constructor \todo why doesn't copying the numa structure work?
distribution::distribution( distribution *d )
  : parallel_structure(d),
    entity(entity_cookie::DISTRIBUTION) {
  add_mask( d->mask );
  // done in decomposition copy: copy_communicator(dynamic_cast<communicator*>(d));
  has_linear_data = d->has_linear_data;
  linear_sizes = d->linear_sizes;
  linear_starts = d->linear_starts;
  linear_offsets = d->linear_offsets;
  //
  compute_numa_structure = d->compute_numa_structure;
  set_type(d->get_type()); set_name(d->get_name());
  set_orthogonal_dimension( d->get_orthogonal_dimension() );
  copy_dist_factory(d); copy_operate_routines(d); copy_communicator(d);
  compute_global_first_index = d->compute_global_first_index; // in copy_operate_routines?
  compute_global_last_index = d->compute_global_last_index;
  try {
    memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in distribution copy from {}",d->get_name()));
  }
};

void distribution::create_from_unique_local( std::shared_ptr<multi_indexstruct> strct) {
  index_int lsize;
  try {
    get_same_dimensionality(1);
    int P = domains_volume(); lsize = strct->local_size(0);
    std::vector<index_int> sizes(P); gather64(lsize,sizes);
    create_from_local_sizes(sizes);
  } catch (std::string c) {
    throw(fmt::format("Error creating from unique local size {}: {}",lsize,c));
  } catch (...) {
    throw(fmt::format("Unknown error creating from unique local size {}",lsize)); }    
};

//! Test whether a coordinate lives on a processor
bool distribution::contains_element(processor_coordinate &p,domain_coordinate &&i) {
  auto pstruct = get_processor_structure(p);
  return pstruct->contains_element(i);
};

//! Test whether a coordinate lives on a processor
bool distribution::contains_element(processor_coordinate &p,domain_coordinate &i) {
  auto pstruct = get_processor_structure(p);
  return pstruct->contains_element(i);
};

//! Local allocation of a distribution is local size of the struct times orthogonal dimension
index_int distribution::local_allocation_p( processor_coordinate &p ) {
  return get_orthogonal_dimension()*volume(p);
};

/*!
  Get the multi-dimensional enclosure of the parallel structure.
  This is memo-ized.
*/
std::shared_ptr<multi_indexstruct> parallel_structure::get_enclosing_structure() {
  try {
    if (enclosing_structure_is_computed<memoization_status::SET) {
      auto enclose = compute_global_structure();
      //fmt::print("Computed global structure as <<{}>>\n",enclose->as_string());
      set_enclosing_structure(enclose);
    }
    return enclosing_structure;
  } catch (std::string c) {
    throw(fmt::format("Error getting enclosing for pstruct: <<{}>>",c));
  }
};

/*!
  Get the multi-dimensional enclosure of the parallel structure.
  This is memo-ized.
*/
std::shared_ptr<multi_indexstruct> parallel_structure::get_global_structure() {
  if (global_structure_is_computed<memoization_status::SET)
    set_global_structure( compute_global_structure() );
  return global_structure;
};

/*!
  We do something funky with this in the product case.
 */
void parallel_structure::set_enclosing_structure(std::shared_ptr<multi_indexstruct> struc) {
  if (enclosing_structure_is_computed==memoization_status::LOCKED)
    throw(std::string("Can not set locked enclosing structure"));
  enclosing_structure = struc; enclosing_structure_is_computed = memoization_status::SET;
};

void parallel_structure::set_global_structure(std::shared_ptr<multi_indexstruct> struc) {
  if (global_structure_is_computed==memoization_status::LOCKED)
    throw(std::string("Can not set locked global structure"));
  global_structure = struc; global_structure_is_computed = memoization_status::SET;
};

/*!
  We do something funky with this in the product case.
 */
void parallel_structure::lock_global_structure(std::shared_ptr<multi_indexstruct> struc) {
  if (global_structure_is_computed==memoization_status::LOCKED)
    throw(std::string("Can not lock locked global structure"));
  global_structure = struc; global_structure_is_computed = memoization_status::LOCKED;
};

/*
  Construct the enclosing structure so that it can be saved.
  Many routines use this enclosing structure, so this saves a ton of recomputation.
*/
std::shared_ptr<multi_indexstruct> parallel_structure::compute_global_structure() {
  int dim = get_dimensionality();
  try {
    auto first = global_first_index(), last = global_last_index();
    return std::shared_ptr<multi_indexstruct>( new contiguous_multi_indexstruct(first,last) );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
      throw(fmt::format("Could not compute global structure for <<{}>>",as_string()));
  }
};

/*!
  Compute the starts and sizes of the processor blocks.
  See also \ref distribution::compute_linear_offsets.
*/
void distribution::compute_linear_sizes() {
  if (has_linear_data) return;
  if (the_communicator_mode==communicator_mode::OMP)
    throw(std::string("Can not yet compute linear sizes for OMP"));
  index_int myfirst;
  try {
    myfirst = linearize(first_index_r(proc_coord(*this)));
  } catch (std::string c) {
    fmt::print("Error: {}\n",c);
    throw(std::string("Could not linearize first index")); }
  try {
    gather32(myfirst,linear_starts);
    gather32(volume(proc_coord(*this)),linear_sizes);
  } catch (std::string c) {
    fmt::print("Error: {}\n",c);
    throw(std::string("Could not gather linear")); }
  has_linear_data = true;
};

std::vector<int> &distribution::get_linear_sizes() {
  compute_linear_sizes();
  return linear_sizes;
};

/*!
  Compute the offsets of the processor blocks.
  The call to \ref distribution::get_linear_sizes
  ensures that the starts and linear sizes have been computed.
*/
void distribution::compute_linear_offsets() {
  compute_linear_sizes();
  //linear_offsets = new std::vector<int>;
  int scan = 0;
  int
    remember_start = linear_starts.at(0)-1,
    remember_size = linear_sizes.at(0)-1;
  int nprocs = linear_sizes.size();
  //  fmt::MemoryWriter w; w.write("Offsets:");
  for (int iproc=0; iproc<nprocs; iproc++) {
    int
      start = linear_starts.at(iproc),
      size = linear_sizes.at(iproc);
    //w.write(" {}: {}@{} -> {} ...",iproc,size,start,scan);
    linear_offsets.at(iproc) = scan; //->push_back(scan);
    if (start==remember_start && size==remember_size)
      processor_skip.push_back(true);
    else {
      processor_skip.push_back(false);
      scan += size;
    }
    remember_start = start; remember_size = size;
  }
};

std::vector<int> &distribution::get_linear_offsets() {
  compute_linear_offsets();
  return linear_offsets;
};

//! Give the linear location of a domain coordinate
index_int parallel_structure::linearize( domain_coordinate &coord ) { 
  return coord.linear_location_in( global_first_index(),global_last_index() ); };

/*!
  Most of the factory routines are completely mode-dependent, so there
  are no initial values.
  Here we only set the initial message factory.
  (The \ref mpi_message object contains some MPI-specific stuff.)
*/
void distribution::set_dist_factory() {
  new_message = 
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	    std::shared_ptr<multi_indexstruct> g) -> message* {
    throw(std::string("No default new_message")); };
  new_embed_message = 
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	    std::shared_ptr<multi_indexstruct> e,
	    std::shared_ptr<multi_indexstruct> g) -> message* {
    throw(std::string("No default new_embed_message")); };
  location_of_first_index =
    [] (distribution &d,processor_coordinate &p) -> index_int {
        throw(std::string("imp_base.h local_of_first_index")); };
  location_of_last_index =
    [] (distribution &d,processor_coordinate &p) -> index_int {
        throw(std::string("imp_base.h local_of_last_index")); };
  numa_first_index =
    [] (void) -> domain_coordinate& { throw(std::string("imp_base.h numa_first_index")); };
  numa_local_size =
    [] (void) -> index_int { throw(std::string("imp_base.h numa_local_size")); };
};

distribution *distribution::operate( const ioperator &op) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(std::string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( fmt::format("{}-operated(pi)",get_name()) );
  return operated;
};

distribution *distribution::operate( const ioperator &&op) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(std::string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( fmt::format("{}-operated(pi)",get_name()) );
  return operated;
};

distribution *distribution::operate( multi_ioperator *op ) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(std::string("Could not upcast to par structure"));
  auto operated_structure = base_structure->operate(op);
  auto operated = new_distribution_from_structure(operated_structure);
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( fmt::format("{}-operated(mi)",get_name()) );
  return operated;
};

//! Operate on distribution by lambda generation on operated structure
distribution *distribution::operate(multi_sigma_operator *op) {
  auto operated = new_distribution_from_structure
    ( dynamic_cast<parallel_structure*>(this)->operate(op) );
  operated->set_name( fmt::format("{}-operated(ms)",get_name()) );
  return operated;
};

/*!
  Operate on a distribution.
  - global operator is applied as such
  - every other type loops over processors, so this requires 
    a globally known distribution
 */
distribution *distribution::operate(distribution_sigma_operator *op) {
  decomposition *decomp = dynamic_cast<decomposition*>(this);
  if (decomp==nullptr)
    throw(std::string("Could not cast to decomposition"));

  if (op->is_global_based()) {
    try {
      return op->operate(this);
    } catch (std::string c) { fmt::print("Error in global dist_sigma operate: {}\n",c);
      throw(fmt::format("Could not global operate on {}",as_string()));
    }
  } else {
    if (!is_known_globally())
      throw(fmt::format("Can not operate non-global dist_sigma_op if not globally known: {}",
			as_string()));
    try {
      auto structure = new parallel_structure(decomp);
      for ( auto me : *decomp ) {
	std::shared_ptr<multi_indexstruct> new_pstruct;
	try {
	  new_pstruct = op->operate(this,me);
	} catch (std::string c) {
	  throw(fmt::format("Operate dist_sig_op failed: {}",c)); };
	structure->set_processor_structure(me,new_pstruct);
      }
      structure->set_type( structure->infer_distribution_type() );
      structure->set_is_known_globally();

      auto operated_dist = new_distribution_from_structure(structure);
      //fmt::print("operated dist: {}\n",operated_dist->as_string());
      return operated_dist;
    } catch (std::string c) {
      throw(fmt::format("Error in non-global dist_sig_op: {}",c));
    }
  }
};

//! Operate on the base. This mostly differs for multiplication operations.
distribution *distribution::operate_base( const ioperator &op ) {
  decomposition *base_decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr) throw(std::string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() )) throw(std::string("Type got lost"));
  auto operated_structure = base_structure->operate_base(op);
  auto operated_distro = new_distribution_from_structure(operated_structure);
  operated_distro->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  return operated_distro;
};

distribution *distribution::operate_base( const ioperator &&op ) {
  decomposition *base_decomp = dynamic_cast<decomposition*>(this);
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr) throw(std::string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() )) throw(std::string("Type got lost"));
  auto operated_structure = base_structure->operate_base(op);
  auto operated_distro = new_distribution_from_structure(operated_structure);
  operated_distro->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  return operated_distro;
};

//! Operate and truncate
distribution *distribution::operate_trunc
    ( ioperator &op,std::shared_ptr<multi_indexstruct> trunc ) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(std::string("Could not upcast to par structure"));
  if (!base_structure->has_type( this->get_type() ))
    throw(std::string("Type got lost"));
  distribution *operated =
    new_distribution_from_structure(base_structure->operate(op,trunc));
  operated->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  operated->set_name( fmt::format("{}-operated(tr)",get_name()) );
  return operated;
};

distribution *distribution::distr_union( distribution *other ) {
  if (get_orthogonal_dimension()!=other->get_orthogonal_dimension())
    throw(fmt::format("Incompatible orthogonal dimensions: this={}, other={}",
		      get_orthogonal_dimension(),other->get_orthogonal_dimension()));
  int dim = get_dimensionality();
  auto union_struct = new parallel_structure(this);
  if (get_is_orthogonal() && other->get_is_orthogonal()) {
    for (int id=0; id<dim; id++) {
      auto new_pidx =
	get_dimension_structure(id)->struct_union(other->get_dimension_structure(id));
      if (new_pidx->outer_size()==0)
	throw(fmt::format("Made empty in dim {} from <<{}>> and <<{}>>",
			  id,get_dimension_structure(id)->as_string(),
			  get_dimension_structure(id)->as_string()));
      union_struct->set_dimension_structure(id,new_pidx);
    }
    union_struct->set_is_orthogonal(true); union_struct->set_is_converted(false);
  } else {
    //printf("Union of multi_structures\n");
    if (get_is_orthogonal()) convert_to_multi_structures();
    if (other->get_is_orthogonal()) other->convert_to_multi_structures();
    for (int is=0; is<domains_volume(); is++) {
      auto pcoord = coordinate_from_linear(is);
      union_struct->set_processor_structure
	( pcoord, get_processor_structure(pcoord)
	  ->struct_union(other->get_processor_structure(pcoord))->force_simplify() );
    }
    union_struct->set_is_orthogonal(false); union_struct->set_is_converted(false);
  }
  union_struct->set_type(distribution_type::GENERAL);
  auto union_d = new_distribution_from_structure(union_struct);
  union_d->set_orthogonal_dimension( get_orthogonal_dimension() );
  try {
    union_d->memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in unioned distr {}",union_d->get_name()));
  }
  return union_d;
};

//! Copy the factory routines.
void distribution::copy_dist_factory(distribution *d) {
  new_distribution_from_structure = d->new_distribution_from_structure;
  new_distribution_from_unique_local = d->new_distribution_from_unique_local;
  new_scalar_distribution = d->new_scalar_distribution;
  new_object = d->new_object; new_object_from_data = d->new_object_from_data;
  new_kernel_from_object = d->new_kernel_from_object;
  kernel_from_objects = d->kernel_from_objects;

  numa_first_index = d->numa_first_index; numa_local_size = d->numa_local_size;
  location_of_first_index = d->location_of_first_index;
  location_of_last_index = d->location_of_last_index;

  local_allocation = d->local_allocation; get_visibility = d->get_visibility;
  new_message = d->new_message; new_embed_message = d->new_embed_message;
};

//! Test on the requested type, and throw an exception if not.
void parallel_structure::require_type(distribution_type t) {
  if (get_type()!=t) {
    throw(fmt::format("Type is <<{}>> should be <<{}>>",
		      type_as_string(),distribution_type_as_string(t)));
  }
};

std::string distribution_type_as_string(distribution_type t) {
  if (t==distribution_type::UNDEFINED)       return std::string("undefined");
  else if (t==distribution_type::CONTIGUOUS) return std::string("contiguous");
  else if (t==distribution_type::BLOCKED)    return std::string("blocked");
  else if (t==distribution_type::REPLICATED) return std::string("replicated");
  else if (t==distribution_type::CYCLIC)     return std::string("cyclic");
  else if (t==distribution_type::GENERAL)    return std::string("general");
  else return std::string("unknown");
};

//! \todo make sure this is collective!
distribution *distribution::extend
    ( processor_coordinate *p,std::shared_ptr<multi_indexstruct> i ) {
  extend(*p,i);
};

distribution *distribution::extend
    ( processor_coordinate ep,std::shared_ptr<multi_indexstruct> i ) {
  parallel_structure *base_structure = dynamic_cast<parallel_structure*>(this);
  if (base_structure==nullptr)
    throw(std::string("Could not upcast to parallel structure"));
  // auto extended_structure = new parallel_structure( base_structure );
  ioperator no_op("none");
  auto extended_structure = base_structure->operate(no_op);

  auto extended_proc_struct = get_processor_structure(ep)->struct_union(i)->force_simplify();
  extended_structure->set_processor_structure(ep,extended_proc_struct);

  auto edist = new_distribution_from_structure(extended_structure);
  edist->set_orthogonal_dimension( this->get_orthogonal_dimension() );
  fmt::print("extended distribution: {}\n",edist->as_string());
  return edist;
};

//! \todo is this ever used? it's largely empty now....
int distribution::equals(distribution *d) {
  if (!(domains_volume()==d->domains_volume()))
    printf("Different numbers of procs\n");
  return 1;
};

// //! Get first index by reference, from C-pointer to processor \todo make this true reference
// domain_coordinate &parallel_structure::first_index_r(processor_coordinate *p) {
//   return get_processor_structure(p)->first_index_r();
// };

// //! Get last index by reference, from C-pointer to processor \todo make this true reference
// domain_coordinate &parallel_structure::last_index_r(processor_coordinate *p) {
//   return get_processor_structure(p)->last_index_r();
// };

//! Get first index by reference, from processor \todo this needs to go: it copies too much
domain_coordinate &parallel_structure::first_index_r(processor_coordinate &p) {
  return get_processor_structure(p)->first_index_r();
};

//! Get last index by reference, from processor \todo this needs to go: it copies too much
domain_coordinate &parallel_structure::last_index_r(processor_coordinate &p) {
  return get_processor_structure(p)->last_index_r();
};

//! Get first index by reference, from processor \todo this needs to go: it copies too much
domain_coordinate &parallel_structure::first_index_r(processor_coordinate &&p) {
  return get_processor_structure(p)->first_index_r();
};

//! Get last index by reference, from processor \todo this needs to go: it copies too much
domain_coordinate &parallel_structure::last_index_r(processor_coordinate &&p) {
  return get_processor_structure(p)->last_index_r();
};

// domain_coordinate parallel_structure::last_index(processor_coordinate p) {
//   return get_processor_structure(&p)->last_index();
// };

/*! Get the global first index, which is supposed to have been set by 
  the distribution creation routines.
*/
const domain_coordinate &parallel_structure::global_first_index() {
  // int dim = get_dimensionality();
  // if (stored_global_first_index==nullptr)
  //   throw(fmt::format("Distribution <<{}>> has no stored global first index",as_string()));
  return stored_global_first_index;
};

/*! Get the global last index, which is supposed to have been set by 
  the distribution creation routines.
*/
const domain_coordinate &parallel_structure::global_last_index() {
  // int dim = get_dimensionality();
  // if (stored_global_last_index==nullptr)
  //   throw(fmt::format("Distribution <<{}>> has no stored global last index",as_string()));
  return stored_global_last_index;
};

std::shared_ptr<indexstruct> distribution_sigma_operator::operate
    (int dim,distribution *d,processor_coordinate &p) const {
  if (!sigma_based)
    throw(fmt::format("Need to be of sigma type for this operate"));
  return sigop.operate( d->get_processor_structure(p)->get_component(dim) );
};

//! \todo the processor coordinate needs ot be passed by reference and be const
std::shared_ptr<indexstruct> distribution_sigma_operator::operate
    (int dim,distribution *d,processor_coordinate p) const {
  if (!sigma_based)
    throw(fmt::format("Need to be of sigma type for this operate"));
  return sigop.operate( d->get_processor_structure(p)->get_component(dim) );
};

std::shared_ptr<multi_indexstruct> distribution_sigma_operator::operate
    (distribution *d,processor_coordinate &p) const {
  if (!lambda_based)
    throw(fmt::format("Need to be of lambda type for this operate"));
  return dist_sigma_f(d,p);
};

/*!
  This is the variant to use if the operate, eh, operation contains collectives
*/
distribution *distribution_sigma_operator::operate(distribution *d) const {
  if (!global_based)
    throw(fmt::format("Need to be of global type for this operate"));
  if (dist_global_f==nullptr)
    throw(fmt::format("Dist global f is null"));
  try {
    return dist_global_f(d);
  } catch (std::string c) { //fmt::print("dist_sig_op global operate failed: {}\n",c);
    throw(fmt::format("Dist_sig_op::operate failed: {}",c));
  } catch (...) { //fmt::print("dist_sig_op global operate failed\n");
    throw(std::string("Dist_sig_op::operate failed"));
  }
};


/*
 * Numa stuff
 */

void distribution::set_numa_structure( std::shared_ptr<multi_indexstruct> n ) {
  set_numa_structure(n,n->first_index_r(),n->volume());
};

void distribution::set_numa_structure
    ( std::shared_ptr<multi_indexstruct> n,const domain_coordinate &f,index_int s) {
  the_numa_structure = n->force_simplify();
  the_numa_first_index = f; the_numa_local_size = s;
  numa_structure_is_computed = true;
};

const domain_coordinate distribution::offset_vector() {
  auto nstruct = get_numa_structure();
  auto gstruct = get_global_structure();
  return nstruct->first_index_r() - gstruct->first_index_r();
};

const domain_coordinate &distribution::numa_size_r() {
  return get_numa_structure()->local_size_r();
};

void distribution::memoize() {
  // compute global first/last, store in parallel_structure member
  if (compute_global_first_index==nullptr)
    throw(std::string("No compute global first defined"));
  try {
    stored_global_first_index = compute_global_first_index(this);
  } catch (std::string c) {
    throw(fmt::format("Could not memo'ize first of {} <<{}>>",as_string(),c));
  }
  // if (stored_global_first_index==nullptr)
  //   throw(std::string("Failed to compute global first"));

  if (compute_global_last_index==nullptr)
    throw(std::string("No compute global last defined"));
  try  {
    stored_global_last_index = compute_global_last_index(this);
  } catch (std::string c) {
    throw(fmt::format("Error memo'izing l of {}: <<{}>>",get_name(),c)); };
  // if (stored_global_last_index==nullptr)
  //   throw(std::string("Failed to compute global last"));
};

//! Get the numa structure on which this process is defined.
std::shared_ptr<multi_indexstruct> distribution::get_numa_structure() {
  try {
    if (!numa_structure_is_computed) {
      compute_numa_structure(this);
    }
  } catch (std::string c) {
    throw(fmt::format("Get numa structure failed with <<{}>> for <<{}>>",c,get_name()));
  }
  if (the_numa_structure==nullptr)
    throw(std::string("Failed to compute numa structure"));
  return the_numa_structure;
};

//! Get the numa structure of a specific dimension
indexstruct *distribution::get_numa_structure(int id) {
  try {
    return get_numa_structure()->get_component(id).get();
  } catch (std::string c) {
    throw(fmt::format("Error <<{}>> getting numa struct in dimension {}",c,id));
  }
};

//snippet analyzepatterndependence
/*!
  This routine is called on an alpha distribution.
  For a requested segment of the beta structure, belonging to "mytid",
  find messages to mytid.
  This constructs:
  - the global_struct in global coordinates with the right sender
  - the local_struct expressed relative to the beta structure

  We can deal with a beta that sticks out 0--gsize, but the resulting
  global_struct will still be properly limited.

 \todo why doesn't the containment test at the end succeed?
 \todo can the tmphalo be completely conditional to the local address space?
*/
std::vector<message*> distribution::messages_for_segment
    ( processor_coordinate &mycoord,self_treatment doself,
      std::shared_ptr<multi_indexstruct> beta_block,
      std::shared_ptr<multi_indexstruct> halo_struct ) {
  int dim = get_dimensionality();
  int P = this->domains_volume();
  auto farcorner = get_farpoint_processor();
  int mytid = mycoord.linearize(this);
  std::vector<message*> messages; // = new std::vector<message*>;
  messages.reserve(P);
  auto buildup = std::shared_ptr<multi_indexstruct>( new multi_indexstruct(dim) );
  int pstart = mytid; if (has_random_sourcing()) pstart += rand()%P;
  for (int ip=0; ip<P; ip++) {
    int p = (pstart+ip)%P; // start with self first, or random
    //if (!lives_on(p)) continue; // deal with masks
    auto pcoord = coordinate_from_linear(p);
    if (doself==self_treatment::ONLY && !mycoord.equals(pcoord))
      continue;
    auto pstruct = get_processor_structure(pcoord);
    auto mintersect = beta_block->intersect(pstruct);
    // compute the intersect of the beta struct and alpha[p] in global coordinates
    if (!mintersect->is_empty() && !(doself==self_treatment::EXCLUDE && p==mytid)
        && !buildup->contains(mintersect) ) {
      auto simstruct = mintersect->force_simplify();
      message *m = new_message(pcoord,mycoord,simstruct); m->set_receive_type();
      if (beta_has_local_addres_space)
	m->relativize_to(halo_struct->force_simplify());
      messages.push_back( m );
      buildup = buildup->struct_union(mintersect)->force_simplify();
      if (buildup->contains(beta_block)) goto covered;
    }
  }
//snippet end

  for (int id=0; id<dim; id++) { // does the halo stick out in dimension id?
    std::shared_ptr<multi_indexstruct> intersect; index_int g = global_size().at(id);
    // is the halo sticking out to the right?
    //snippet mpiembedmessage
    auto pleft = mycoord.left_face_proc(id,farcorner);
    intersect = beta_block->intersect
      (get_processor_structure(pleft)->operate(shift_operator(id,g)));
    if ( !intersect->is_empty()
         && !(doself==self_treatment::EXCLUDE && pleft.equals(mycoord)) ) {
      buildup = buildup->struct_union(intersect);
      message *m = new_embed_message
        (pleft,mycoord,intersect,intersect->operate(shift_operator(id,-g)));
      if (beta_has_local_addres_space) {
        auto tmphalo = halo_struct->operate(shift_operator(id,-g));
        m->relativize_to(tmphalo);
      }
      messages.push_back(m);
    }
    //snippet end
    // is the halo sticking out to the left
    auto pright = mycoord.right_face_proc(id,farcorner);
    intersect = beta_block->intersect
      ( get_processor_structure(pright)->operate(shift_operator(id,-g)) );
    if ( !intersect->is_empty()
         && !(doself==self_treatment::EXCLUDE && pright.equals(mycoord)) ) {
      buildup = buildup->struct_union(intersect);
      message *m = new_embed_message
	(pright,mycoord,intersect,intersect->operate(shift_operator(id,g)));
      if (beta_has_local_addres_space) {
	auto tmphalo = halo_struct->operate(shift_operator(id,g));
	m->relativize_to(tmphalo);
      }
      messages.push_back(m);
    }
  }
  // if (!buildup->contains(beta_block))
  //   throw(fmt::format("message buildup {} does not cover beta {}",
  // 		      buildup->as_string(),beta_block->as_string()));
 covered:
  for ( auto m : messages )
    m->set_halo_struct(halo_struct);
  return messages;
};

std::string distribution::as_string() {
  parallel_structure *pstruct = dynamic_cast<parallel_structure*>(this);
  if (pstruct==nullptr)
    throw(std::string("Could not cast distribution to parallel_structure"));
  return fmt::format("{}:[{}]",get_name(),pstruct->parallel_structure::as_string());
};

/****
 **** Sparse stuff
 ****/

/*!
  Extract the indices from a sparse row
  \todo this can make a really large indexed before it simplifies.
*/
std::shared_ptr<indexstruct> sparse_row::all_indices() {
  auto all = std::shared_ptr<indexstruct>( new indexed_indexstruct() ); // empty?
  for (auto e=r.begin(); e!=r.end(); ++e)
    all->addin_element( (*e).get_index() );
  //all = all->add_element( (*e).get_index() );
  return all->force_simplify();
};
/*!
  Sparse inner product of a sparse row and a dense row from an object.
  \todo omp struct 4 somehow takes the general path
 */
double sparse_row::inprod( std::shared_ptr<object> o,processor_coordinate &p ) {
  double s = 0.;
  if (o->has_type_blocked()) {
    auto odata = o->get_data(p);
    index_int shift_to_local = o->location_of_first_index(*o.get(),p);
    for (index_int icol=0; icol<size(); icol++) { 
      auto e = r[icol]; index_int col = e.get_index();
      double x1 = e.get_value(), x2 = odata[col-shift_to_local];
      s += x1*x2;
    }
  } else {
    for (index_int icol=0; icol<size(); icol++) {
      auto e = r[icol]; index_int col = e.get_index();
      double x1 = e.get_value(), x2 = o->get_element_by_index(col,p);
      s += x1*x2;
    }
  }
  return s;
};

/*! Sparse matrix constructor by adding a \ref sparse_rowi for each 
  row in the owned range.
  \todo we should at some point handle non-contiguous owned ranges
*/
sparse_matrix::sparse_matrix( indexstruct *idx ) : sparse_matrix() {
  m.reserve(idx->local_size());
  if (idx->is_contiguous()) {
    for (index_int i=idx->first_index(); i<=idx->last_index(); ++i)
      insert_row( std::shared_ptr<sparse_rowi>( new sparse_rowi(i) ) );
  } else throw(fmt::format("Can not create spmat from type {}",idx->type_as_string()));
  set_row_creation_locked();
};

/*!
  Add a new row to a matrix, keeping the rows sorted by global number.
  \todo throw error if already found
*/
void sparse_matrix::insert_row( std::shared_ptr<sparse_rowi> row ) {
  if (m.size()==0) {
    m.push_back(row);
  } else {
    int newrow = row->get_row_number(), oldrow;
    auto pos = m.begin();
    for ( ; pos!=m.end(); ++pos ) {
      oldrow = (*pos)->get_row_number();
      if (newrow<oldrow) break;
    }
    if (newrow==oldrow)
      throw(fmt::format("matrix already has a row #{}",row->get_row_number()));
    m.insert(pos,row);
  }
};

//! Test whether an (i,j) index is already present.
bool sparse_matrix::has_element( index_int i,index_int j ) {
  if (get_row_creation_locked() && (i<first_row_index() || i>last_row_index()) )
    throw(fmt::format("Row {} out of bounds",i));
  auto row = get_row_index_by_number(i);
  if (row<0)
    return false;
  return m.at(row)->has_element(j);
};

/*!
  Set an element. This will create a row if needed.
*/
void sparse_matrix::add_element( index_int i,index_int j,double v ) {
  if (j<0 || (globalsize>=0 && j>=globalsize) )
    throw(fmt::format("Column {} exceeds global size 0-{}",j,globalsize));
  auto row = get_row_index_by_number(i);
  if (row>=0) { // row already exists
    m.at(row)->add_element(j,v);
  } else { // row does not exist yet
    if (get_row_creation_locked())
      throw(fmt::format("Can not create row {} in locked matrix: indices={}",
			i,row_indices()->as_string()));
    auto newrow = std::shared_ptr<sparse_rowi>( new sparse_rowi(i) );
    newrow->add_element(j,v);
    insert_row( newrow );
  }
};

/*!
  Get a row by its absolute number. Return nullptr if not found.
  \todo keep a seek pointer.
 */
int sparse_matrix::get_row_index_by_number(index_int idx) {
  for (int irow=0; irow<m.size(); irow++ ) {
    index_int row_no = m[irow]->get_row_number();
    //fmt::print("compare {} to {}\n",idx,row_no);
    if (idx==row_no)
      return irow;
  }
  return -1;
};

/*!
  Get all the row indices. This routines should not be used computationally.
 */
std::shared_ptr<indexstruct> sparse_matrix::row_indices() {
  auto idx = std::shared_ptr<indexstruct>{ new indexed_indexstruct() };
  for (auto row : m )
    idx->addin_element( row->get_row_number() );
  return idx->force_simplify();
};

std::string sparse_element::as_string() {
  return fmt::format("{}:{}",get_index(),get_value());
};

std::string sparse_row::as_string() {
  fmt::MemoryWriter w;
  for (auto e=r.begin(); e!=r.end(); ++e) w.write("{} ",(*e).as_string());
  return w.str();
};

std::string sparse_rowi::as_string() {
  return fmt::format("<{}>: {}",get_row_number(),get_row()->as_string());
};

/*!
  All columns from the local matrix.
*/
std::shared_ptr<indexstruct> sparse_matrix::all_columns() {
  auto all = std::shared_ptr<indexstruct>{ new empty_indexstruct() };
  for (auto row : m) {
    all = all->struct_union( row->all_indices() );
  }
  auto r_all = all->force_simplify();
  // fmt::print("Sparse matrix columns <<{}>> simplified to <<{}>>\n",
  // 	     all->as_string(),r_all->as_string());
  return r_all;
};

/*!
  All columns by request.
  \todo get that indexstruct iterator to work
*/
std::shared_ptr<indexstruct> sparse_matrix::all_columns_from
    ( std::shared_ptr<multi_indexstruct> multi_wanted ) {
  if (multi_wanted->get_dimensionality()>1)
    throw(std::string("Can not get all columns from in multi-d"));
  auto wanted = multi_wanted->get_component(0);
  if (wanted->is_empty())
    throw(fmt::format("Matrix seems to be empty on this proc: {}",as_string()));
  auto all = std::shared_ptr<indexstruct>{ new empty_indexstruct() };
  for (int iirow=0; iirow<wanted->local_size(); iirow++) {
    try {
      int irow = wanted->get_ith_element(iirow);
      auto row = get_row_index_by_number(irow);
      if (row<0)
	throw(fmt::format("Could not locally get row {}",irow));
      all = all->struct_union( m.at(row)->all_indices() );
    } catch (std::string c) {
      throw(fmt::format("Error processing row #{} of {}: {}",iirow,wanted->as_string(),c));
    }
  }
  return all->force_simplify();
};

sparse_matrix *sparse_matrix::transpose() const {
  throw(std::string("sparse matrix transpose not implemented"));
};

/*! 
  Product of a sparse matrix into a blocked vector.
  \todo write iterator over all rows, inprod returning sparse_element
*/
void sparse_matrix::multiply
    ( std::shared_ptr<object> in,std::shared_ptr<object> out,processor_coordinate &p) {
  if (out->get_dimensionality()>1)
    throw(std::string("spmvp not in multi-d"));
  if (out->has_type_locally_contiguous()) {
    auto data = out->get_data(p);
    index_int
      tar0 = out->location_of_first_index(*out.get(),p),
      len = out->volume(p);
    for (index_int i=0; i<len; i++) {
      index_int rownum = out->first_index_r(p).coord(0)+i;
      auto sprow = get_row_index_by_number(rownum);
      double v = m.at(sprow)->inprod(in,p);
      data[tar0+i] = v;
    }
  } else {
    auto outstruct = out->get_processor_structure(p)->get_component(0);
    throw(fmt::format("Can only multiply into blocked type, not <<{}>>",
		      outstruct->as_string()));
  }
};

std::string sparse_matrix::as_string() {
  return fmt::format("Sparse matrix on {}, nnzeros={}",
		     row_indices()->as_string(),nnzeros());
};

std::string sparse_matrix::contents_as_string() {
  fmt::MemoryWriter w; w.write("{}:\n",as_string());
  for (auto row=m.begin(); row!=m.end(); ++row)
    w.write("{}\n",(*row)->as_string());
  return w.str();
};

/****
 **** Object data
 ****/

/*!
  Create data pointers/offset/size for each domain.
  Offsets are mostly useful for OpenMP.
  This is called in the object_data constructor
*/
void object_data::create_data_pointers(int ndom) {
  domain_data_pointers = std::vector<double*>(ndom);
  for ( auto &p : domain_data_pointers )
    p = nullptr;
  data_offsets = std::vector<index_int>(ndom);
  data_sizes = std::vector<index_int>(ndom);
};

void object_data::set_domain_data_pointer( int n,double *dat,index_int s,index_int o) {
  domain_data_pointers.at(n) = dat; data_sizes.at(n) = s; data_offsets.at(n) = o;
};
  
//! Allocate data and store the pointer as the numa data
void object_data::create_data(index_int s, std::string c) {
  if (s<0)
    throw(fmt::format("Negative {} malloc for <<{}>>",s,c));
  std::shared_ptr<double> dat;
  if (s==0)
    dat = std::shared_ptr<double>( new double[1] );
  else
    dat = std::shared_ptr<double>( new double[s] );
  //  set_data_pointer(0,dat,s);
  numa_data_pointer = dat; numa_data_size = s*sizeof(double);
  data_status = object_data_status::ALLOCATED;
  create_data_count += numa_data_size;
  if (get_trace_create_data())
    fmt::print("Create {} for <<{}>>, reaching {}\n",s,c,create_data_count);
};

//! Create unnamed data
void object_data::create_data(index_int s) {
  create_data(s,std::string("Unknown object"));
};

std::shared_ptr<double> object_data::get_numa_data_pointer() const {
  return numa_data_pointer;
};

double *object_data::get_raw_data() const {
  return numa_data_pointer.get();
};

index_int object_data::get_raw_size() const {
  return numa_data_size;
};

//! Register someone else's data as mine; use offset. \todo add size argument
void object_data::inherit_data( std::shared_ptr<object> o,index_int offset,processor_coordinate &p ) {
  //throw(fmt::format("inherit at offset is dangerous\n"));
  auto dat = o->get_data(p);
  // the data is a bare pointer, so we can do arithmetic on it.
  set_domain_data_pointer(0, dat+offset, o->volume(p));
  data_status = object_data_status::INHERITED;
};

//! Inherit someone's data; base addresses align. \\! \todo should we lose the processor?
void object_data::inherit_data( std::shared_ptr<object> o,processor_coordinate &p ) {
  //  inherit_data(o,0,p);
  numa_data_pointer = o->numa_data_pointer;
  numa_data_size = o->numa_data_size;
  domain_data_pointers = o->domain_data_pointers;
  data_sizes = o->data_sizes;
  data_offsets = o->data_offsets;
};

//! Get the data pointer by local number, computed by \ref get_domain_local_number.
double* object_data::get_data_pointer( int n ) {
  auto p = domain_data_pointers.at(n);
  if (p==nullptr)
    throw(fmt::format("Data pointer #{} is null",n));
  return domain_data_pointers.at(n); };

//! Get the data size by local number, computed by \ref get_domain_local_number.
index_int object_data::get_data_size( int n ) {
  return data_sizes.at(n);
};

//! Set an object to a constant value. \todo domains \todo extend to orthogonal dimensions
void object_data::set_value( std::shared_ptr<double> x ) {
  if (!has_data_status_allocated())
    throw(std::string("Can not set value for unallocated"));
  for (int idom=0; idom<domain_data_pointers.size(); idom++) {
    auto dat = get_data_pointer(idom); index_int siz = get_data_size(idom);
    for (index_int i=0; i<siz; i++)
      dat[i] = *x;
  }
};

void object_data::set_value(double x) {
  std::shared_ptr<double> xx{ new double[1] }; xx.get()[0] = x;
  set_value(xx);
};

double object_data::get_max_value() {
  double mx = *( domain_data_pointers.at(0) );
  for (int p=0; p<domain_data_pointers.size(); p++) {
    auto dat = domain_data_pointers[p]; int s = data_sizes[p];
    for (int i=0; i<s; i++) {
      double v = dat[i];
      if (v>mx) mx = v;
    }
  }
  return mx;
};

double object_data::get_min_value() {
  if (domain_data_pointers.size()==0)
    throw(std::string("Can not get min value from non-existing data"));
  double mn = *( domain_data_pointers[0] );
  for (int p=0; p<domain_data_pointers.size(); p++) {
    auto dat = domain_data_pointers[p]; int s = data_sizes[p];
    for (int i=0; i<s; i++) {
      double v = dat[i];
      if (v<mn) mn = v;
    }
  }
  return mn;
};

std::string object_data::data_status_as_string() {
  if (data_status==object_data_status::UNALLOCATED) return std::string("UNALLOCATED");
  if (data_status==object_data_status::ALLOCATED) return std::string("ALLOCATED");
  if (data_status==object_data_status::INHERITED) return std::string("INHERITED");
  if (data_status==object_data_status::REUSED) return std::string("REUSED");
  if (data_status==object_data_status::USER) return std::string("USER");
  throw(std::string("Invalid data status"));
};

/****
 **** Object
 ****/

//snippet getelement
/*!
  Find an element by index. 
  This uses \ref get_numa_structure since for OMP we can see everything, for MPI only local.
  This is so far only used in the sparse matrix product.
  \todo broken in multi-d
*/
double object::get_element_by_index(index_int i,processor_coordinate &p) {
  auto localstruct = get_numa_structure();
  index_int locali = localstruct->linearfind(i);
  if (locali<0 || locali>=numa_local_size())
    throw(fmt::format("Found {} as {}: out of bounds of <<{}>>",
		      i,locali,localstruct->as_string()));
  return get_data(p)[locali];
};
//snippet end

/*! Render object information as std::string. See also \ref object::values_as_string.
  \todo this becomes a circular call */
std::string object::as_string() {
  return fmt::format("{}:Distribution:<<{}>>",
		     get_name(),dynamic_cast<distribution*>(this)->get_name());
};

std::string object::values_as_string(processor_coordinate &p) {
  fmt::MemoryWriter w; w.write("{}:",get_name());
  if (get_orthogonal_dimension()>1)
    throw(std::string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_data(p);
  index_int f = location_of_first_index(*this,p), s = volume(p);
  for (index_int i=0; i<s; i++)
    w.write(" {}:{}",i+f,data[i]);
  return w.str();
};

std::string object::values_as_string(processor_coordinate &&p) {
  fmt::MemoryWriter w; w.write("{}:",get_name());
  if (get_orthogonal_dimension()>1)
    throw(std::string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_data(p);
  index_int f = location_of_first_index(*this,p), s = volume(p);
  for (index_int i=0; i<s; i++)
    w.write(" {}:{}",i+f,data[i]);
  return w.str();
};

/****
 **** Message
 ****/

/*!
  Message creation. We set the local struct to global; this will be relativized in MPI.
  - global struct : global indexes of the message
  - embed struct : global but before twe wrap halos, so it can be -1:0 and such.

  \todo can be omit the cloning? the outer structs are temporary enough
  \todo I wish we could make the local_struct through relativizing, but there's too much variability in how that is done.
*/
message::message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
		 std::shared_ptr<multi_indexstruct> &e,std::shared_ptr<multi_indexstruct> &g)
  : decomposition(d),
    entity(dynamic_cast<entity*>(d),entity_cookie::MESSAGE) {
  sender = snd; receiver = rcv;
  local_struct = g->make_clone();
  global_struct = g->make_clone();
  embed_struct = e->make_clone();
  set_name(fmt::format("message-{}->{}",snd.as_string(),rcv.as_string()));
};

//! Return the message sender.
processor_coordinate &message::get_sender() {
  if (sender.get_dimensionality()<=0)
    throw(std::string("Invalid sender"));
  return sender; };

//! Return the message receiver.
processor_coordinate &message::get_receiver() {
  if (receiver.get_dimensionality()<=0)
    throw(std::string("Invalid receiver"));
  return receiver; };

std::shared_ptr<multi_indexstruct> message::get_global_struct() {
  if (global_struct==nullptr) throw(std::string("msg has no global struct"));
  return global_struct; };

std::shared_ptr<multi_indexstruct> message::get_embed_struct() {
  if (embed_struct==nullptr) throw(std::string("msg has no embed struct"));
  return embed_struct; };

std::shared_ptr<multi_indexstruct> message::get_local_struct() {
  if (local_struct==nullptr) throw(std::string("msg has no local struct"));
  return local_struct; };

std::shared_ptr<object> message::get_in_object( ) {
  if (in_object==nullptr) throw(std::string("Message has no in object"));
  return in_object; };

std::shared_ptr<object> message::get_out_object( ) {
  if (out_object==nullptr) throw(std::string("Message has no out object"));
  return out_object; };

//! Set the outputobject of a message.
void message::set_out_object( std::shared_ptr<object> out ) { out_object = out; };

//! Set the input object of a message.
void message::set_in_object( std::shared_ptr<object> in ) { in_object = in; };

//snippet subarray
void message::compute_subarray
(std::shared_ptr<multi_indexstruct> outer,std::shared_ptr<multi_indexstruct> inner,int ortho) {
  int dim = outer->get_same_dimensionality(inner->get_dimensionality());
  numa_sizes = new int[dim+1]; struct_sizes = new int[dim+1]; struct_starts = new int[dim+1];
  //  annotation.write("tar subarray:");
  auto loc = outer /*halo*/->location_of(inner);
  for (int id=0; id<dim; id++) {
    numa_sizes[id] = outer->local_size_r().at(id);
    struct_sizes[id] = inner->local_size_r().at(id);
    struct_starts[id] = loc->at(id);
    annotation.write(" {}:{}@{}in{}",id,struct_sizes[id],struct_starts[id],numa_sizes[id]);
  }
  // if (ortho>1)
  //   annotation.write(" (k={})",ortho);
  numa_sizes[dim] = ortho; struct_sizes[dim] = ortho; struct_starts[dim] = 0;
};
//snippet end

/*!
  Where does the send buffer fit in the input object?
  This is called by \ref message::set_in_object.
  This will be extended by \ref mpi_message::compute_src_index.
*/
//snippet impsrcindex
void message::compute_src_index() {
  if (src_index!=-1)
    throw(fmt::format("Can not recompute message src index in <<{}>>",get_name()));
  try {
    auto send_struct = get_global_struct(); // local ???
    auto proc_struct = get_in_object()->get_processor_structure(get_sender());
    src_index = send_struct->linear_location_in(proc_struct);
  } catch (std::string c) { throw(fmt::format("Error <<{}>> setting src_index",c)); }

  auto outer = get_in_object()->get_numa_structure();
  auto inner = get_global_struct();
  int
    ortho = get_in_object()->get_orthogonal_dimension();
  try {
    compute_subarray(outer,inner,ortho);
  } catch (std::string c) {
    throw(fmt::format("Could not compute src subarray for <<{}>> in <<{}>>: {}",
		      inner->as_string(),outer->as_string(),c));
  }
};
//snippet end

/*!
  Where does the receive buffer fit in the output object?
  This is call by \ref message::set_out_object.
  This will be extended by \ref mpi_message::compute_tar_index.
*/
//snippet imptarindex
void message::compute_tar_index() {
  if (tar_index!=-1)
    throw(fmt::format("Can not recompute message tar index in <<{}>>",get_name()));

  // computing target index. is that actually used?
  try {
    auto local_struct = get_local_struct();
    auto processor_struct = get_out_object()->get_processor_structure(get_receiver());
    tar_index = local_struct->linear_location_in(processor_struct);
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not compute target index"));
  }

  auto outer = get_out_object()->get_numa_structure();
  auto inner = get_embed_struct();
  try {
    int
      ortho = get_out_object()->get_orthogonal_dimension();
    compute_subarray(outer,inner,ortho);
  } catch (std::string c) {
    throw(fmt::format("Could not compute tar subarray for <<{}>> in <<{}>>: {}",
		      inner->as_string(),outer->as_string(),c));
  }
};
//snippet end

index_int message::get_src_index() {
  if (src_index==-1)
    throw(std::string("Src index not yet computed"));
  return src_index;
};

index_int message::get_tar_index() {
  if (tar_index==-1)
    throw(std::string("Tar index not yet computed"));
  return tar_index;
};


//snippet msgrelativize
/*!
  Express the local structure of a message as the global struct
  relative to a halo structure.
*/
void message::relativize_to(std::shared_ptr<multi_indexstruct> container) { 
  local_struct = global_struct->relativize_to(container);
};
//snippet end

//! \todo is this actually used? use linearize for the sender/receiver or use fmt::print
void message::as_char_buffer(char *buf,int *len) {
  if (*len<14) {
    throw(std::string("Insufficient buffer provided for message->as_string"));
  }
  int dim = global_struct->get_dimensionality();
  if (dim>1)
    throw(fmt::format("Can not convert msg to buffer in {} dim\n",dim));
  sprintf(buf,"%2d->%2d:[%2ld-%2ld]",
	  get_sender()[0],get_receiver()[0],
	  global_struct->first_index(0),global_struct->last_index(0));
  *len = 14;
};

void message::set_send_type() {
  sendrecv_type = message_type::SEND;
};
void message::set_receive_type() {
  sendrecv_type = message_type::RECEIVE;
};
message_type message::get_sendrecv_type() const {
  if (sendrecv_type==message_type::NONE)
    throw(std::string("Message has sendrecv type none"));
  return sendrecv_type;
};
std::string message::sendrecv_type_as_string() const {
  if (get_sendrecv_type()==message_type::SEND)
    return std::string("S;");
  else if (get_sendrecv_type()==message_type::RECEIVE)
    return std::string("R;");
  else
    return std::string("U;");
};

void message::set_tag_by_content
    (processor_coordinate &snd,processor_coordinate &rcv,int step,int np) {
  auto farcorner = get_farpoint_processor();
  tag = new message_tag(snd.linearize(farcorner),rcv.linearize(farcorner),step,np);
};

std::string message::as_string() {
  fmt::MemoryWriter w;
  w.write("{}:",sendrecv_type_as_string());
  w.write("{}->{}:{}",sender.as_string(),receiver.as_string(),global_struct->as_string());
  return w.str();
};

/****
 **** Signature function
 ****/

//
// sigma from operators
//

/*!
  Add an operator to a signature function.
 */
void signature_function::add_sigma_operator( multi_ioperator *op ) {
  set_type_operator_based();
  operators.push_back(op);
};

//
// signature from function
//

/*!
  Supply the int->indexstruct function; we wrap it in an \index ioperator,
  which we apply to whole domains.
*/
void signature_function::set_signature_function_function( multi_sigma_operator *op ) {
  set_type_function(); func = op;
};
//! The one-d special case is wrapped in the general mechanism
void signature_function::set_signature_function_function( const sigma_operator &op ) {
  set_signature_function_function( new multi_sigma_operator(op) );
};
//! Defined a signature function from a pointwise function \todo add multi_idx->multi_idx func
void signature_function::set_signature_function_function
    ( std::function< std::shared_ptr<indexstruct>(index_int) > f ) {
  set_signature_function_function( sigma_operator(f) );
};

//
// signature from explicit beta
//

//! \todo test that this does not contain multi blocks?
void signature_function::set_explicit_beta_distribution( distribution *d ) {
  explicit_beta_distribution = d; set_type_explicit_beta();
};

distribution *signature_function::get_explicit_beta_distribution() {
  if (!has_type_initialized(signature_type::EXPLICIT))
    throw(std::string("is not of explicit beta type"));
  if (explicit_beta_distribution==nullptr) throw(std::string("has no explicit beta"));
  return explicit_beta_distribution;
};

/*!
  Create the parallel_indexstruct corresponding to the beta distribution.

  This outputs a new object, even if it is just copying a gamma distribution.

  \todo can we get away with passing the pointer if it's an explicit beta?
  \todo int he operator-based case, is the truncation properly multi-d?
  \todo we need to infer the distribution type
  \todo efficiency: do this only on local domains
  \todo make shared pointer
*/
parallel_structure *signature_function::derive_beta_structure
    (distribution *gamma,std::shared_ptr<multi_indexstruct> truncation) {
  parallel_structure *beta_struct{nullptr};
  if (has_type_uninitialized()) {
    throw(std::string("Can not derive_beta_structure from uninitialized signature"));
  } else if (has_type_explicit_beta()) {
    distribution *x = get_explicit_beta_distribution();
    if (x->has_type(distribution_type::UNDEFINED))
      throw(std::string("Explicit beta should not have type undefined\n"));
    beta_struct = new parallel_structure(x);
    beta_struct->set_type(x->get_type());
  } else { // case: pattern, operator, function
    beta_struct = new parallel_structure( dynamic_cast<decomposition*>(gamma) );
    beta_struct->set_type(distribution_type::GENERAL);
    if (has_type_operator_based()) { // case: operators
      beta_struct = derive_beta_structure_operator_based(gamma,truncation);
    } else if (has_type_pattern()) { // case: sparsity pattern
      beta_struct = derive_beta_structure_pattern_based(gamma);
    } else if (has_type_function()) { // case: signature function explicit
      beta_struct->set_type( gamma->get_type() );
      parallel_structure *gamma_struct = dynamic_cast<parallel_structure*>(gamma);
      for (int p=0; p<gamma->domains_volume(); p++) {
	auto pcoord = gamma->coordinate_from_linear(p);
	auto gamma_p = gamma_struct->get_processor_structure(pcoord);
	auto newstruct = gamma_p->operate(func);
	beta_struct->set_processor_structure( pcoord,newstruct );
      }
    } else throw(std::string("Can not derive beta for this type"));
  }

  //  beta_struct->set_global_structure( gamma->get_enclosing_structure() );
  beta_struct->deduce_global_structure(); // noop for now
  return beta_struct;
};

parallel_structure *signature_function::derive_beta_structure_operator_based
    (distribution *gamma,std::shared_ptr<multi_indexstruct> truncation) {
  parallel_structure *beta_struct = 
    new parallel_structure( dynamic_cast<decomposition*>(gamma) );
  std::shared_ptr<multi_indexstruct> newstruct;
  beta_struct->set_type( gamma->get_type() );
  if (gamma->is_known_globally()) {
    for (int p=0; p<gamma->domains_volume(); p++) {
      auto pcoord = gamma->coordinate_from_linear(p); // identical block below. hm.
      try {
	newstruct = make_beta_struct_from_ops
	  (pcoord,gamma->get_processor_structure(pcoord),get_operators(),truncation);
      } catch (std::string c) {
	throw(fmt::format("Error <<{}>> in beta struct for {}\n",c,pcoord.as_string())) ; }
      if (newstruct->volume()==0)
	throw(fmt::format("Somehow made empty beta struct for coord {}",pcoord.as_string()));
      if (newstruct->is_multi())
	newstruct = newstruct->enclosing_structure();
      beta_struct->set_processor_structure( pcoord,newstruct );
    }
  } else {
    try {
      auto pcoord = gamma->proc_coord(*gamma);
      newstruct = make_beta_struct_from_ops
	(pcoord,gamma->get_processor_structure(pcoord),get_operators(),truncation);
      if (newstruct->volume()==0)
	throw(fmt::format("Somehow made empty beta struct for coord {}",pcoord.as_string()));
      if (newstruct->is_multi())
	newstruct = newstruct->enclosing_structure();
      beta_struct->set_processor_structure( pcoord,newstruct );
    } catch (std::string c) {
      throw(fmt::format("Deriving beta struct operator based: {}",c));
    }
  }
  return beta_struct;
};

//! \todo How much do we lose by taking that surrounding contiguous?
parallel_structure *signature_function::derive_beta_structure_pattern_based
    (distribution *gamma) {
  parallel_structure *beta_struct = 
    new parallel_structure( dynamic_cast<decomposition*>(gamma) );
  if (gamma->get_dimensionality()>1)
    throw(std::string("Several bugs in beta from type pattern"));
  //snippet pstructfrompattern
  for (auto dom : gamma->get_domains()) {
    auto base = gamma->get_processor_structure(dom);
    std::shared_ptr<indexstruct> columns,simple_columns;
    try {
      columns = pattern->all_columns_from(base);
    } catch (std::string c) {
      throw(fmt::format("Could not get columns for domain {}: {}",dom.as_string(),c)); }
    try {
      simple_columns = columns->over_simplify();
    } catch (std::string c) {
      throw(fmt::format("Could not simplify columns for domain {}: {}",dom.as_string(),c)); }
    // fmt::print("Matrix colums padded from <<{}>> to <<{}>>\n",
    // 	       columns->as_string(),simple_columns->as_string());
    beta_struct->set_processor_structure
      ( dom,std::shared_ptr<multi_indexstruct>( new multi_indexstruct(simple_columns) ) );
  }
  //snippet end
  return beta_struct;
};

// //! Construct the beta structure from gamma object \todo can this go?
// parallel_structure *signature_function::derive_beta_structure
//     (std::shared_ptr<object> gamma,std::shared_ptr<multi_indexstruct> enclosing) {
//   return derive_beta_structure(dynamic_cast<distribution*>(gamma.get()),enclosing);
// };

/*!
  For an operator-based \ref signature_function, create \f$ \beta(p) \f$ by union'ing
  the \ref indexstruct objects from applying the \ref ioperator objects contained 
  in #ops.

  For the case of modulo-based operators, the result is truncated. This 
  is a different kind of truncation than going on in distribution::messages_for_segment.

  An operator result is allowed to be empty, for instance from a shift followed by
  truncation.

 */
std::shared_ptr<multi_indexstruct> signature_function::make_beta_struct_from_ops
( processor_coordinate &pcoord, // this argument only for tracing
  std::shared_ptr<multi_indexstruct> gamma_struct,
  std::vector<multi_ioperator*> &ops, std::shared_ptr<multi_indexstruct> truncation ) {
  int dim = gamma_struct->get_dimensionality();
  if (truncation!=nullptr && truncation->is_empty())
    throw(std::string("Found empty truncation structure"));
  if (ops.size()==0)
    throw(std::string("Somehow no operators"));
  if (gamma_struct->is_empty())
    throw(fmt::format("Finding empty processor structure"));

  auto halo_struct = std::shared_ptr<multi_indexstruct>( new empty_multi_indexstruct(dim) );
  // fmt::MemoryWriter w;
  // w.write("Gamma struct {}\n",gamma_struct->as_string());
  for ( auto beta_op : ops ) {
    std::shared_ptr<multi_indexstruct> beta_struct;
    if (beta_op->is_modulo_op() || truncation==nullptr) {
      beta_struct = gamma_struct->operate(beta_op);
    } else {
      beta_struct = gamma_struct->operate(beta_op,truncation);
    }
    //w.write("operator {} gives {}, ",beta_op->as_string(),beta_struct->as_string());
    if (!beta_struct->is_empty()) {
      halo_struct = halo_struct->struct_union(beta_struct);
      //w.write("union={}\n",halo_struct->as_string());
    }
  }

  if (halo_struct->is_empty()) {
    fmt::MemoryWriter w;
    w.write("Make empty beta struct from {} by applying:",gamma_struct->as_string());
    for ( auto o : ops )
      w.write(" {},",o->as_string());
    throw(w.str());
  }
  halo_struct = halo_struct->force_simplify();
  return halo_struct;
};

/****
 **** Dependency
 ****/

//! Most of the time we create a dependency and later set the signature function
dependency::dependency(std::shared_ptr<object> in)
  : entity(dynamic_cast<entity*>(in.get()),entity_cookie::SIGNATURE) {
  in_object = in;
  set_name( fmt::format("dependency on object <<{}>>",in->get_name()) );
};

std::shared_ptr<object> dependency::get_in_object() {
  if (in_object==nullptr)
    throw(std::string("Dependency has no in object"));
  return in_object;
};

void dependency::set_beta_distribution( distribution *b) {
  beta_distribution = b;
};

distribution *dependency::get_beta_distribution() {
  if (beta_distribution==nullptr)
    throw(fmt::format("Dependency <<{}>> has no beta distribution",get_name()));
  return beta_distribution; };

int dependency::has_beta_distribution() {
  return beta_distribution!=nullptr;
};

void dependency::set_beta_object( std::shared_ptr<object> h ) {
  if (beta_object!=nullptr)
    throw(fmt::format("Can not override beta in dependency <<{}>>",get_name()));
  beta_object = h; };

std::shared_ptr<object> dependency::get_beta_object() {
  if (!has_beta_object())
    throw(std::string("No beta object to be got"));
  return beta_object;
};

/*!
  Allocate a halo based on a distribution. This creates the object and
  stores it in the halo member of the dependency.

  We use the \ref distribution::new_object factory routine for creating
  the actual object because dependencies are mode-independent.
*/
void dependency::create_beta_vector(std::shared_ptr<object> out) {
  if (has_beta_object()) return;
  ensure_beta_distribution(out);
  distribution *beta = get_beta_distribution();
  auto halo = std::shared_ptr<object>( beta->new_object(beta) );
  halo->allocate();
  halo->set_name(fmt::format("[{}]:halo",this->get_name()));
  set_beta_object(halo);
};

/*!
  Make sure that this dependency has a beta distribution.

  \todo is that mask on the beta actually used?
*/
void dependency::ensure_beta_distribution(std::shared_ptr<object> outvector) {
  if (!has_beta_distribution()) {
    auto invector = get_in_object();
    parallel_structure *pstruct;
    try {
      distribution *alpha = dynamic_cast<distribution*>(invector.get()),
	*gamma = dynamic_cast<distribution*>(outvector.get());
      pstruct = derive_beta_structure(gamma,alpha->get_enclosing_structure());
    } catch (std::string c) {
      fmt::print("Error <<{}>> for in={} [{}] out={} [{}]\n",
		 c,invector->get_name(),invector->as_string(),
		 outvector->get_name(),outvector->as_string() );
      throw(fmt::format("Could not ensure beta"));
    } catch (...) { throw(std::string("derive beta struct: other error")); }
    pstruct->set_type( pstruct->infer_distribution_type() );
    
    //snippet ensurebeta
    beta_distribution = outvector->new_distribution_from_structure(pstruct);
    if (outvector->has_mask())
      beta_distribution->add_mask(outvector->get_mask());
    //snippet end
    beta_distribution->set_name(fmt::format("beta-for-<<{}>>",get_name()));
    beta_distribution->set_orthogonal_dimension( invector->get_orthogonal_dimension() );
  }
};

/*!
  Get a vector of beta objects to pass to the local execute function.
  We really a vector, but this is only constructed once (per task execution),
  so the waste is not too bad.
*/
std::vector<std::shared_ptr<object>> dependencies::get_beta_objects() {
  std::vector<std::shared_ptr<object>> objs;
  for ( auto d : the_dependencies ) 
    objs.push_back( d->get_beta_object() );
  return objs;
};

/****
 **** Task
 ****/

std::vector<message*> &task::get_receive_messages() {
  return recv_messages;
};
std::vector<message*> &task::get_send_messages()   {
  return send_messages;
};
void task::set_receive_messages( std::vector<message*> &msgs) {
  recv_messages = msgs;
};
void task::set_send_messages( std::vector<message*> &msgs) {
  send_messages = msgs;
};
//! Take the receive msgs so that you can really give them to another task
std::vector<message*> task::lift_recv_messages() {
  auto msgs = recv_messages; recv_messages.clear();
  return msgs;
};
//! Take the send msgs so that you can really give them to another task
std::vector<message*> task::lift_send_messages() {
  auto msgs = send_messages; send_messages.clear();
  return msgs;
};

void task::add_post_messages( std::vector<message*> &msgs ) {
  for ( auto m : msgs ) {
    post_messages.push_back(m);
  }
}

std::vector<message*> &task::get_post_messages() {
  return post_messages;
};

void task::add_xpct_messages( std::vector<message*> &msgs ) {
  for ( auto m : msgs ) 
    xpct_messages.push_back(m);
};

std::vector<message*> &task::get_xpct_messages() {
  return xpct_messages;
};

/*!
  Find the corresponding send message for a receive message,
  otherwise return nullptr. This can only work for OpenMP:
  see \ref omp_request_vector::wait.
*/
message *task::matching_send_message(message *rmsg) {
  auto smsgs = get_send_messages();
  if (smsgs.size()==0)
    throw(fmt::format("Task <<{}>> somehow has no send messages",as_string()));
  for (auto smsg : smsgs) {
    //fmt::print("Comparing smsg=<<{}>> rmsg=<<{}>>\n",smsg->as_string(),rmsg->as_string());
    if ( smsg->get_sender().coord(0)==rmsg->get_sender().coord(0) &&
	 smsg->get_receiver().coord(0)==rmsg->get_receiver().coord(0) ) {
      return smsg;
    }
  }
  return nullptr;
};

/*!
  This does the following:
  - create the recv messages, locally by inspecting the beta structure
  - create the send message, which can be complicated
  - allocate the halo, for modes that need this
  - construct the list of predecessor tasks, given in step/domain coordinates

  \todo make a unit test for the predecessors
  \todo the send structure routine is pure virtual, so belongs to task; move to dependency?
  \todo see allocate_halo_vector: probably not store the halo in the dependency. but see kernel::analyze_dependencies
*/
void task::analyze_dependencies() {
  int step = get_step(); auto dom = get_domain();
  if (get_has_been_analyzed()) return;
  if (!has_type_origin()) {
    auto out = get_out_object();
    for ( auto d : get_dependencies() ) {
      // allocate the halo, this will be included in the messages as out_object
      try { 
	d->ensure_beta_distribution(out); d->create_beta_vector(out);
      } catch (std::string c) { fmt::print("Error <<{}>> on p={}\n",c,dom.as_string());
	throw(fmt::format("Could not make beta & halo for dep <<{}>> of task <<{}>>",
			  d->get_name(),this->as_string()));
      }
    }
    // create receive messages, these will carry the in object number
    try {
      derive_receive_messages();
    } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
      throw(fmt::format("Could not analyze task {}",get_name())); }
    // attach in/out objects to messages
    for ( auto m : get_receive_messages() ) {
      int innum = m->get_in_object_number();
      // record predecessor relations
      predecessor_coordinates.push_back( new task_id(innum,m->get_sender()) );
    }
    // create the send message structure
    try { derive_send_messages(); } catch (std::string c) {
      fmt::print("Error <<{}>> during derive send messages\n",c);
      throw(fmt::format("Task analyze dependencies failed for <<{}>>",as_string()));
    }
  }
  try { local_analysis(); }
  catch (std::string c) { fmt::print("Error <<{}>> in <<{}>>\n",c,get_name());
    throw(fmt::format("Local analysis failed")); }
  set_has_been_analyzed();
};

/*! 
  The receive structure of a task is spread over its dependencies,
  so the global call simply loops over the dependencies.
*/
void task::derive_receive_messages()
{
  int step = get_step(); auto pid = get_domain();
  //  if (recv_messages==nullptr) throw(std::string("We don't have a recv messages vector"));
  auto deps = get_dependencies();
  for ( int id=0; id<deps.size(); id++) { // we need the dependency number in the message
    auto d = deps.at(id);
    if (!d->has_beta_distribution())
      throw(std::string("dependency needs beta dist for create recv"));
    auto invector = d->get_in_object(), halo = d->get_beta_object();
    int ndomains = invector->domains_volume();
    distribution
      *beta_dist = d->get_beta_distribution();
    //snippet msgsforbeta
    auto beta_block = beta_dist->get_processor_structure(pid);
    auto numa_block = beta_dist->get_numa_structure(); // to relativize against
    self_treatment doself;
    if ( invector->has_collective_strategy(collective_strategy::MPI)
	 && d->get_is_collective() )
      doself = self_treatment::ONLY;
    else
      doself = self_treatment::INCLUDE;
    std::vector<message*> msgs;
    try {
      msgs = invector->messages_for_segment( pid,doself,beta_block,numa_block );
    } catch (std::string c) { fmt::print("Error: {}\n",c);
      throw(fmt::format("Could not derive messages for segment {} on numa {}",
			beta_block->as_string(),numa_block->as_string()));
    }
    //snippet end
    for ( auto msg : msgs ) {
      try {
        msg->set_name( fmt::format("recv-msg-{}-{}",
				   invector->get_object_number(),halo->get_object_number()) );
	msg->set_in_object(invector); msg->set_out_object(halo);
	msg->set_dependency_number(id); msg->set_receive_type();
        try {
          msg->compute_tar_index();
        } catch ( std::string c ) {
          fmt::print("Error computing tar index <<{}>>\n",c);
          throw(fmt::format("Could not localize recv message {}->{}",
			    msg->get_sender().as_string(),msg->get_receiver().as_string()));
	}
        msg->set_tag_from_kernel_step(step,ndomains);
        msg->set_is_collective( d->get_is_collective() );
	msg->add_trace_level( this->get_trace_level() );
        recv_messages.push_back( msg );
      } catch(std::string c) { fmt::print("Error <<{}>>\n",c);
	throw(fmt::format("Could not process message {} for dep <<{}>> in task <<{}>>",
			  msg->as_string(),d->as_string(),this->as_string()));
      }
    }
  }
};

/*!
  Convert knowledge of what I receive into who is sending to me.
  This uses a collective of some sort; based on MPI we call this
  `reduce-scatter'.
  \todo this calls reduce_scatter once for each task. not right with domains
*/
int task::get_nsends() {
  auto out = get_out_object();
  // first get my list of senders
  int ntids = out->domains_volume(), nrecvs=0,nsends;
  std::vector<int> my_senders; my_senders.reserve(ntids);
  auto layout = out->get_domain_layout();
  for (int i=0; i<ntids; ++i) my_senders.push_back(0);
  for ( auto msg : get_receive_messages() ) {
    int s = msg->get_sender().linearize(layout);
    if (s<0 || s>=ntids)
      throw(fmt::format("Invalid linear dom={} for layout <<{}>>",s,layout.as_string()));
    my_senders[s]++; nrecvs++;
  }

  // to invert that, first find out how many procs want my data
  try {
    int lineardomain = get_domain().linearize(layout);
    nsends = get_out_object()->reduce_scatter(my_senders.data(),lineardomain);
  } catch (std::string c) {
    fmt::print("Error <<{}>> doing reduce-scatter\n",c);
    throw(fmt::format("Task <<{}>> computing <<{}>>, failing in get_nsends",
		      get_name(),get_out_object()->as_string()));
  }

  return nsends;
};

dependency *task::find_dependency_for_object_number(int innum) {//( object *in ) {
  //  int innum = in->get_object_number();
  for ( auto dep : get_dependencies() ) {
    auto obj = dep->get_in_object();
    if (obj->get_object_number()==innum) {
      return dep;
    }
  }
};

/*!
  Post all messages and store the requests.
  The \ref notifyReadyToSendMsg call is pure virtual and implemented in each mode.
*/
void task::notifyReadyToSend( std::vector<message*> &msgs,request_vector *requests ) {
  for ( auto msg : msgs ) {
    if (msg->get_is_collective())
      return;
    auto newreq = notifyReadyToSendMsg(msg);
    if (newreq==nullptr) {
      msg->set_status( message_status::SKIPPED );
    } else {
      msg->set_status( message_status::POSTED );
      requests->add_request( newreq );
    }
  }
};

//snippet taskexecute
/*!
  Do task synchronization and local execution of a task.

  Masking is tricky.
  - Of course we skip execution if the output does not live on this task.
  - If we have output, we do sends and receives, but
  - we skip execution if any halos are missing.
 */
void task::execute() {
  auto d = get_domain(); auto outvector = get_out_object();

  // allocate the output if not already; maybe it's embedded.
  try { outvector->allocate();
  } catch (std::string c) {
    throw(fmt::format("Error during outvector allocate: {}",c)); }

  if (get_has_been_executed()) {
    fmt::print("Not executing {}\n",this->get_name());
    if (get_exec_trace_level()>=exec_trace_level::EXEC)
      fmt::print("Task bypassed <<{}>>\n",this->get_name());
    return;
  }
  if (!outvector->lives_on(get_domain()))
    return;

  bool trace = get_exec_trace_level()>=exec_trace_level::EXEC;
  if (trace)
    fmt::print("Task execute <<{}>>\n",this->get_name());

  if (!get_has_been_optimized()) {
    try { notifyReadyToSend(get_send_messages(),requests);
    } catch (std::string c) { fmt::print("Error <<{}>> in notifyReadyToSend\n",c);
      throw(fmt::format("Send posting failed for <<{}>>",get_name()));
    }
    try { acceptReadyToSend(get_receive_messages(),requests);
    } catch (std::string c) { fmt::print("Error <<{}>> in acceptReadyToSend\n",c);
      throw(fmt::format("Recv posting failed for <<{}>>",get_name()));
    }
  }

  auto objs = this->get_beta_objects();
  if (!has_type_origin() && objs.size()==0)
    throw(fmt::format("Non-origin task <<{}>> has zero inputs",get_name()));
  if (all_betas_live_on(d)) {
    if (outvector->get_split_execution())
      try { // the local part that can be done without communication.
	local_execute(objs,outvector,localexecutectx,unsynctest);
      } catch (std::string c) { fmt::print("Error <<{}>> in split exec1\n",c);
	throw(fmt::format("Task local execution failed before sync")); }
  } else fmt::print("skip pre-exec for missing halo\n");

  if (trace) fmt::print("Waiting for {} requests\n",requests->size());
  try { requests->wait(); }
  catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Task <<{}>> request wait failed",this->get_name())); }
  catch (...) { fmt::print("Unknown error waiting for requests\n");
    throw(fmt::format("Task <<{}>> request wait failed",this->get_name())); }
  record_nmessages_sent( requests->size() );

  if (all_betas_live_on(get_domain()) && all_betas_filled_on(d)) {
    try { // for product this executes the embedded queue; otherwise just local stuff
      if (trace) fmt::print("local execute\n");
      if (!outvector->get_split_execution())
	local_execute(objs,outvector,localexecutectx);
      else
	local_execute(objs,outvector,localexecutectx,synctest);
    } catch (std::string c) { fmt::print("Error <<{}>> in split exec2\n",c);
      throw(fmt::format("Task local execution failed after sync")); }
  } // else fmt::print("skip pre-exec on {} for missing halo\n",d);
 
  // Post the sends and receives of the task that will receive our data.
  // The wait for these requests happens elsewhere
  notifyReadyToSend(get_post_messages(),pre_requests);
  acceptReadyToSend(get_xpct_messages(),pre_requests);

  result_monitor(this->shared_from_this());
  set_has_been_executed();

}
//snippet end

/*!
  Conditional execution with an all-or-nothing test. 
  In the product mode we override this with a fine-grained test.
*/
void task::local_execute
    (std::vector<std::shared_ptr<object>> &beta_objects,std::shared_ptr<object> outobj,void *ctx,
     int(*tasktest)(std::shared_ptr<task> t)) {
  if (localexecutefn==nullptr)
    throw(fmt::format("Hm. no local function...."));
  if (outobj->has_data_status_unallocated())
    throw(fmt::format("Object <<{}>> has no data",outobj->get_name()));
  if ( tasktest==nullptr || (*tasktest)(this->shared_from_this()) ) {
    int s = get_step_counter(); //get_step(); 
    auto p = get_domain();
    double flopcount = 0.;
    try {
      localexecutefn(s,p,beta_objects,outobj,&flopcount);
    } catch (std::string c) {
      throw(fmt::format("Task tested local exec <<{}>> failed: {}",get_name(),c));
    }
    set_flop_count(flopcount);
  }
};

//! Unconditional local execution.
void task::local_execute
    (std::vector<std::shared_ptr<object>> &beta_objects,std::shared_ptr<object> outobj,void *ctx) {
  if (localexecutefn==nullptr)
    throw(fmt::format("Hm. no local function...."));
  {
    int s = get_step_counter(); 
    auto p = get_domain();
    double flopcount = 0.;
    if (!p.get_same_dimensionality(outobj->get_dimensionality()))
      throw(fmt::format("Random sanity check: p={}, obj={}",
			p.as_string(),outobj->get_name()));
    try {
      localexecutefn(s,p,beta_objects,outobj,&flopcount);
    } catch (std::string c) { 
      throw(fmt::format("Task untested local exec <<{}>> failed: {}",get_name(),c));
    }
    set_flop_count(flopcount);
  }
};

void task::clear_has_been_executed() {
  done = 0; delete_requests_vector(); make_requests_vector();
  for ( auto m : get_send_messages() )
    m->clear_was_sent();
  for ( auto m : get_receive_messages() )
    m->clear_was_sent();
};

/*!
  Check that all tasks have been executed exactly one.
 */
int algorithm::get_all_tasks_executed() {
  int all = 1;
  for ( auto t : get_tasks() )
    if (!t->get_has_been_executed()) { all = 0; break; }
  return allreduce_and(all);
};

/*!
  If we want to re-execute an algorithm, we need to clear the executed status.
*/
void algorithm::clear_has_been_executed() {
  for ( auto t : get_tasks() ) {
    if (!t->has_type_origin())
      t->clear_has_been_executed();
  }
};

std::string task::as_string() {
  fmt::MemoryWriter w;
  w.write("{}[s={},p={}",get_name(),get_step(),get_domain().as_string());
  if (get_is_synchronization_point()) w.write(",sync");
  if (has_type_origin())              w.write(",origin");
  w.write("]");
  if (!has_type_origin()) {
    auto preds = get_predecessors();
    w.write(", #preds={}: [",preds.size());
    for (auto p : preds ) // (auto p=preds->begin(); p!=preds->end(); ++p)
      w.write("{} ",p->get_name());
    w.write(" ]");
  }
  return w.str();
};

/****
 **** Kernel
 ****/

//! Copy constructor; this is only called when creating tasks.
kernel::kernel( kernel *old ) 
  : dependencies(static_cast<dependencies>(*old)),entity(entity_cookie::KERNEL) {
  set_name(static_cast<entity*>(old)->get_name());
  out_object = old->out_object; type = old->type; 
  localexecutefn = old->localexecutefn; localexecutectx = old->localexecutectx;
};

/*!
  We get a vector of \ref task objects from \ref kernel::split_to_tasks;
  here is where we store it. See also \ref kernel::addto_kernel_tasks.
  \todo make reference
*/
void kernel::set_kernel_tasks(std::vector< std::shared_ptr<task> > tt) {
  if (kernel_has_tasks())
    throw(fmt::format("Can not set tasks for already split kernel <<{}>>",get_name()));
  addto_kernel_tasks(tt); was_split_to_tasks = 1;
};

/*!
  With composite kernels each of the component kernels makes a vector of tasks;
  we gradually add them to the surrounding kernel.
  \todo make reference
*/
void kernel::addto_kernel_tasks(std::vector< std::shared_ptr<task> > tt) {
  for (auto t : tt ) //=tt.begin(); t!=tt.end(); ++t)
    kernel_tasks.push_back(t);
  was_split_to_tasks = 1;
};

const std::vector< std::shared_ptr<task> > &kernel::get_tasks() const {
  if (!kernel_has_tasks())
    throw(fmt::format("Kernel <<{}>> was not yet split to tasks",get_name()));
  return kernel_tasks;
};

/*!
  Analyzing kernel dependencies is mostly delegating the analysis
  to the kernel tasks. By dependencies we mean dependency on other kernels
  that originate the input objects for this kernel. 

  Origin tasks have no dependencies, so we skip them. Note that 
  ultimately they may still have outgoing messages; however
  that is set after queue optimization.
  \todo the ensure_beta_distribution call is also in task::analyze_dependencies. lose this one?
*/
void kernel::analyze_dependencies() {
  if (!kernel_has_tasks())
    this->split_to_tasks();

  if (get_has_been_analyzed()) return;
  auto outobject = get_out_object();
  if (!has_type_origin()) {
    auto the_dependencies = get_dependencies();
    for (auto d : the_dependencies ) {
      try { d->ensure_beta_distribution(outobject); }
      catch (std::string c) { fmt::print("Error <<{}>>\n",c);
	throw(fmt::format("Could not ensure beta in kernel<<{}>> for dependency <<{}>>",
			  get_name(),d->get_name())); }
    }
  }
  auto tsks = this->get_tasks();
  for ( auto t : tsks ) {
    t->analyze_dependencies();
  }
  if (!has_type_origin()) {
    auto the_dependencies = get_dependencies();
    for (auto d : the_dependencies ) {
      if (!d->get_beta_object()->global_structure_is_locked())
	d->get_beta_object()->set_global_structure
	  (d->get_in_object()->get_global_structure());
    }
  }
  set_has_been_analyzed();
};

/*!
  We generate a vector of tasks, which inherit a bunch of things from the surrounding kernel
  - the beta definition
  - the local function
  - the function context
  - the name (plus a unique id)
  of the kernel.
*/
void kernel::split_to_tasks() {
  if (kernel_has_tasks()) return;
  auto outvector = get_out_object();
  kernel_tasks.reserve( outvector->domains_volume() );

  auto domains = outvector->get_domains();
  if (domains.size()==0) printf("WARNING zero domains in kernel <<%s>>\n",get_name().data());
  if (tracing_progress())
    fmt::print("Kernel <<{}>> splitting to {} domains\n",get_name(),domains.size());
  for (auto dom : domains) {
    std::shared_ptr<task> t = make_task_for_domain(dom); // pure virtual kernel method
    t->set_localexecutefn( this->localexecutefn );
    t->set_localexecutectx( this->localexecutectx );
    t->find_kernel_task_by_domain =
      [&,outvector] (processor_coordinate &d) -> std::shared_ptr<task> {
      return get_tasks().at( d.linearize(dynamic_cast<distribution*>(outvector.get())) ); };
    t->set_name( fmt::format("T[{}]-{}-{}",this->get_name(),
			     this->get_step(),dom.as_string()) );
    kernel_tasks.push_back(t);
  }
  if (kernel_tasks.size()==0)
    fmt::print("Suspiciously zero tasks\n");
  was_split_to_tasks = 1;
};

void kernel::analyze_contained_kernels( kernel *k1,kernel *k2 ) {
  this->split_to_tasks();
  k1->analyze_dependencies();
  k2->analyze_dependencies();
};

void kernel::analyze_contained_kernels( kernel *k1,kernel *k2,kernel *k3 ) {
  this->split_to_tasks();
  k1->analyze_dependencies();
  k2->analyze_dependencies();
  k3->analyze_dependencies();
};

void kernel::split_contained_kernels( kernel *k1,kernel *k2 ) {
  if (kernel_has_tasks()) return;
  try {
    k1->split_to_tasks();  set_kernel_tasks( k1->get_tasks() );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Fail to split kernel <<{}>>",k1->get_name())); }
  try {
    k2->split_to_tasks(); addto_kernel_tasks( k2->get_tasks() );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Fail to split kernel <<{}>>",k2->get_name())); }
};

void kernel::split_contained_kernels( kernel *k1,kernel *k2,kernel *k3 ) {
  if (kernel_has_tasks()) return;
  try {
    k1->split_to_tasks();  set_kernel_tasks( k1->get_tasks() );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Fail to split kernel <<{}>>",k1->get_name())); }
  try {
    k2->split_to_tasks(); addto_kernel_tasks( k2->get_tasks() );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Fail to split kernel <<{}>>",k2->get_name())); }
  try {
    k3->split_to_tasks(); addto_kernel_tasks( k3->get_tasks() );
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Fail to split kernel <<{}>>",k3->get_name())); }
};

/*!
  Kernel tasks are independent, so why have a kernel execute function that
  executes the tasks of that kernel in sequence. For MPI and BSP purposes
  this suffices; however, for general task models we need to execute
  tasks in the context of a task queue.
*/
void kernel::execute() {
  if (kernel_tasks.size()==0)
    throw(std::string("kernel should have tasks"));
// #ifdef VT
//   VT_begin(vt_kernel_class);
// #endif
  auto tsks = get_tasks();
  for ( auto t : tsks ) {
    if (t==nullptr)
      throw(fmt::format("Finding nullptr to task in kernel: {}",as_string()));
    t->execute();
  }
// #ifdef VT
//   VT_end(vt_kernel_class);
// #endif
};

std::string kernel::as_string() { fmt::MemoryWriter w;
  auto o = get_out_object();
  w.write("K[{}]-out:<<{}#{}>>",get_name(),o->get_name(),o->global_volume());
  auto deps = get_dependencies();
  for ( auto d : deps ) { //.begin(); d!=deps.end(); ++d) {
    auto o = d->get_in_object();
    w.write("-in:<<{}#{}>>",o->get_name(),o->global_volume());
  }
  return w.str();
};

//! Count how many messages the tasks in this kernel receive.
int kernel::local_nmessages() {
  auto tsks = get_tasks();
  int nmessages = 0;
  for ( auto t : tsks ) {// (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    nmessages += t->get_receive_messages().size();
  }
  return nmessages;
}

/****
 **** Queue
 ****/

/*!
  Add a kernel to the internal list of kernels. This is a user
  level routine, and does not involve any sort of analysis, which
  waits until algorithm::analyze_dependencies.
 */
void algorithm::add_kernel(kernel *k) {
  k->set_step_counter( get_kernel_counter_pp() );
  //  fmt::print("adding kernel as count={}\n",k->get_step_counter());
  all_kernels->push_back( k );
};

/*! 
  Push the tasks of a kernel onto this queue. This is an internal routine
  called from algorithm::analyze_dependencies.
  This uses an overridable kernel::get_tasks

  \todo is this the right way, or should it be kernel::addtasks( tasks_queue* ) ?
*/
void algorithm::add_kernel_tasks_to_queue( kernel *k ) {
  for ( auto t : k->get_tasks() ) {
    this->add_task(t);
  }
};

void algorithm::add_task( std::shared_ptr<task> &t ) {
  tasks.push_back(t);
};

std::shared_ptr<task>& algorithm::get_task(int n) {
  if (n<0 || n>=tasks.size()) throw(std::string("Invalid task number; forgot origin kernel?"));
  return tasks.at(n); };

//! Queue split to tasks is splitting all kernels. Nothing deeper than that.
void algorithm::split_to_tasks() {
  if (get_has_been_split_to_tasks()) return;
  for ( auto k : *get_kernels() ) {
    k->split_to_tasks();
    for ( auto t : k->get_tasks() ) {
      this->add_task(t);
      t->find_other_task_by_coordinates =
	[&] (int s,processor_coordinate &d) -> std::shared_ptr<task>{
	return find_task_by_coordinates(s,d); };
      t->record_flop_count =
	[&] (double c) { record_flop_count(c); };
    }
  }
  set_has_been_split_to_tasks();
};

/*!
  We analyze a task queue by analyzing all its kernels, which recursively
  analyzes their tasks. The tasks are inserted into the queue.

  \todo increase message_tag_admin_threshold by max kernel number
  \todo figure out how to do the kernels dot file here.
 */
void algorithm::analyze_dependencies() {
  if (get_has_been_analyzed()) throw(std::string("Can not analyze twice"));
  analysis_event.begin();
  try {
    split_to_tasks();
    analyze_kernel_dependencies();
    find_predecessors(); // make task-task graph
    find_exit_tasks();
    mode_analyze_dependencies(); // OMP: split into locally executable & not
    if (get_can_message_overlap()) optimize();
    for ( auto t : get_tasks() ) {
      t->set_sync_tests(unsynctest,synctest); // Product: inherit tests on sync execution
      // immediately throw exception if circular?
      if (t->check_circularity(t)) {
	fmt::print("Task <<{}>> root of circular dependency path\n",t->get_name());
	set_circular();
      }
    }
  } catch (std::string c) { // at top level print and throw int
    fmt::print("Queue analysis failed: {}\n{}\n",c,this->contents_as_string()); throw(-1);
  }
  if (do_optimize)
    optimize();
  set_has_been_analyzed();
  analysis_event.end();
  if (get_trace_summary())
    fmt::print("Queue contents:\n{}\n",this->contents_as_string());
};

/*!
  We analyze the kernels, doing local analysis between kernel and predecessors.
  Recursive global analysis is done in \ref algorithm::analyze_dependencies.
*/
void algorithm::analyze_kernel_dependencies() {
  for ( auto k : *get_kernels() ) {
    k->analyze_dependencies();
    for ( auto d : k->get_dependencies() ) {
      d->get_in_object()->set_has_successor();
    };
  }
};

/*! 
  Find the tasks that have no successor. Traversing all tasks twice is no big deal.
*/
void algorithm::find_exit_tasks() {
  for ( auto t : get_tasks() ) {
    for ( auto p : t->get_predecessors() ) {
      p->set_is_non_exit();
    }
  }
  for ( auto t : get_tasks() ) {
    if (!t->is_non_exit()) {
      exit_tasks.push_back(t);
    }
  }
};

/*!
  Depth-first checking if a root task appears in its tree of dependencies.
*/
int task::check_circularity( std::shared_ptr<task> root ) {
  if (has_type_origin() || is_not_circular()) {
    return 0;
  } else {
    for ( auto t : get_predecessors() ) {
      //(auto t=get_predecessors()->begin(); t!=get_predecessors()->end(); ++t) {
      if (t->get_step()==root->get_step() && t->get_domain().equals(root->get_domain())) {
	fmt::print("Task <<{}>> is tail of circular dependency path\n",t->get_name());
	return 1;
      } else {
	int c = t->check_circularity(root);
	if (c) {
	  fmt::print("Task <<{}>> is on circular dependency path\n",t->get_name());
	  return 1;
	}
      }
    }
    set_not_circular();
  }
  return 0;
};

/*!
  Find a task number in the queue given its step/domain coordinates.
*/
std::shared_ptr<task> algorithm::find_task_by_coordinates( int s,processor_coordinate &d ) {
  // for (int it=0; it<tasks.size(); it++) {
  //   auto t = tasks.at(it);
  for ( auto t : tasks ) {
    if (t->get_step()==s && t->get_domain()==d) //.equals(d))
      return t;
  }
  throw(fmt::format("Could not find task by coordinates {},{}",s,d.as_string()));
};

/*!
  Find a task number in the queue given its step/domain coordinates.
*/
int algorithm::find_task_number_by_coordinates( int s,processor_coordinate &d ) {
  for (int it=0; it<tasks.size(); it++) {
    auto t = tasks.at(it);
    if (t->get_step()==s && t->get_domain()==d) //.equals(d))
      return it;
  }
  throw(std::string("Could not find task number by coordinate"));
};

/*!
  For each task find its predecessor ids.
  Every task already has the 
  predecessors by step/domain coordinate; here we convert that to linear
  coordinates. 

  Note that tasks may live on a different address space. Therefore the 
  \ref declare_dependence_on_task routine is pure virtual. The MPI version 
  is interesting.
 */
void algorithm::find_predecessors() {
  //  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
  for ( auto t : tasks ) {
    auto task_ids = t->get_predecessor_coordinates();
    for ( auto id : task_ids ) {
      try {
	t->declare_dependence_on_task( id );
      } catch (std::string c) {
	fmt::print("Error: {}\nQueue: {}\n",c,this->header_as_string());
	throw(fmt::format("Queue problem"));
      }
    }
  };
};

/*!
  After the standard analysis
  we split tasks in ones that are a synchronization point
  or dependent on one, and ones that are not and therefore can be
  executed locally without synchronizing in hybrid context.
*/
void algorithm::determine_locally_executable_tasks() {
  int nsync = 0;
  for ( auto t : get_tasks() ) {
    t->check_local_executability();
    task_local_executability lsync = t->get_local_executability();
    nsync += lsync==task_local_executability::NO;
  } 
};

/*!
  We declare the left and right OpenMP task in the origin kernel 
  to be a synchronization point. It's a first order approximation.
 */
void algorithm::set_outer_as_synchronization_points() {
  if (!get_has_been_split_to_tasks())
    split_to_tasks();
  //auto tsks = get_tasks();
  int cnt=0, org=0;
  for ( auto t : get_tasks() ) {
    if (t->has_type_origin()) { org++;
      auto d = t->get_domain();
      if (d.is_on_face( t->get_out_object().get() )) {
        t->set_is_synchronization_point(); cnt++;
      }
    }
  } 
  set_has_synchronization_tasks(cnt);
};

/*!
  Count the number of synchronization tasks,
  or return the number if this has already been determined.
 */
int algorithm::get_has_synchronization_tasks() {
  if (has_synchronization_points<0) {
    int count = 0;
    for ( auto t : get_tasks() ) {
      count += t->get_is_synchronization_point();
    }
    set_has_synchronization_tasks(count);
  }
  return has_synchronization_points;
};

//snippet queueoptimize
/*!
  We perform an optimization on the queue to get messages posted as early as possible.
  This is every so slightly tricky.

  For now we don't do this automatically: it has to be called by the user.
 */
void algorithm::optimize() {
  //  for (auto t=tasks.begin(); t!=tasks.end(); ++t) {
  for ( auto t : tasks ) {
    if (!t->has_type_origin()) {
      auto domain = t->get_domain();
      auto deps = t->get_dependencies();
      for (auto d=deps.begin(); d!=deps.end(); ++d) {
	int originkernel = (*d)->get_in_object()->get_object_number();
	auto othertsk = find_task_by_coordinates(originkernel,domain);
	{ auto lift = t->lift_send_messages();
	  othertsk->add_post_messages(lift); }
	{ auto lift = t->lift_recv_messages();
	  othertsk->add_xpct_messages(lift); }
	othertsk->set_pre_requests( t->get_requests() );
	t->set_has_been_optimized();
      }
    }
  }
}
//snippet end

/*!
  For a given object, find the kernel that produces it, and all the kernels
  that consume it. We throw an exception if we find more than one source.

  \todo is there a use for this?
*/
void algorithm::get_data_relations 
        (const char *object_name,kernel **src,std::vector<kernel*> **tar ) {
  kernel *src_kernel = NULL;
  std::vector<kernel*> *targets = new std::vector<kernel*>;
  for (auto k=all_kernels->begin(); // std::vector<kernel*>::iterator
       k!=all_kernels->end(); ++k) {
    if (!strcmp( object_name,(*k)->get_out_object()->get_name().data() )) {
      if (src_kernel!=NULL) {
	throw(std::string("found two sources for object")); }
      src_kernel = (*k);
    }
    if (!strcmp( object_name,(*k)->last_dependency()->get_in_object()->get_name().data() )) {
      targets->push_back( (*k) );
    }
  }
  *src = src_kernel;
  *tar = targets;
};

/*!
  The base method for executing a queue is only a timer around 
  a method that executes the tasks. The timer calls are virtual
  because they require different treatment between MPI/OpenMP.
 */
void algorithm::execute( int(*tasktest)(std::shared_ptr<task> t) ) {
  if (tasktest==nullptr) throw(fmt::format("Missing task test in queue::execute"));
  if (is_circular()) throw(fmt::format("Can not execute queue with circular dependencies\n"));
  execution_event.begin();
  try {
    execute_tasks(tasktest);
  } catch (std::string c) { fmt::print("Error <<{}>> in queue <<{}>> execute\n",c,get_name());
    throw(fmt::format("Task execute failed"));
  }
  execution_event.end();
  //  if (coordinate_from_linear(0)==processor_coordinate_zero(get_dimensionality()))
  fmt::print("Algorithm runtime: {}\n",execution_event.get_duration());
};

/*!
  Execute all tasks in a queue. Since they may not be in the right order,
  we take each as root and go down their predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.

  This is the base method; it suffices for MPI, but OpenMP overrides it by
  setting some directives before calling the base method.

  \todo move the first two lines to algorithm::execute, argument to this should be vector; check OMP first!
*/
void algorithm::execute_tasks( int(*tasktest)(std::shared_ptr<task> t) ) {
  if (tasktest==nullptr)
    throw(fmt::format("Missing task test in queue::execute_tasks"));
  for ( auto t : get_exit_tasks() ) {
    try {
      if ((*tasktest)(t))
	t->execute_as_root();
    } catch ( std::string c ) {
      fmt::print("Error <<{}>> for task <<{}>> execute\n",c,t->as_string());
      throw(std::string("Task queue execute failed"));
    }
  }
};

/*!
  Execute a task by taking it as root and go down the predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.
*/
void task::execute_as_root() {
  if (get_has_been_executed()) {
    return;
  }
  for ( auto t : get_predecessors() )
    t->execute_as_root();
  execute();
};

/*!
  We try to embed vectors in the halo built on them.
  For now, this works in MPI only, called from \ref mpi_algorithm::mode_analyze_dependencies. 
  In OpenMP we have to think much harder.

  \todo how does this relate to multiple domains?
*/
void algorithm::inherit_data_from_betas() {
  for ( auto t : get_tasks() ) {
    auto deps = t->get_dependencies(); auto p = t->get_domain();
    for ( auto d : deps ) {
      auto in = d->get_in_object(),
	halo = d->get_beta_object();
      if (!in->has_data_status_allocated()) {
	auto embeddable = in->get_processor_structure(p),
	  embedder = halo->get_processor_structure(p);
	if (embedder->contains(embeddable)) {
	  index_int offset = embeddable->linear_location_in(embedder);
	  in->inherit_data(halo,offset,p);
	  in->set_data_parent(halo->get_object_number());
	} else  {
	  in->allocate();
	}
      }
    }
  }
};

//! \todo find a way to make the #tasks a global statistic
std::string algorithm::header_as_string() {
  return fmt::format("Queue <<{}>> protocol: {}; #tasks={}",
		     get_name(),protocol_as_string(),global_ntasks());
};

std::string algorithm::kernels_as_string() {
  fmt::MemoryWriter w;
  w.write("Kernels:");
  auto kernels = get_kernels();
  for (auto k=kernels->begin(); k!=kernels->end(); ++k)
    w.write("<<{}>>\n",(*k)->as_string());
  w.write("]");
  return w.str();
};

std::string algorithm::contents_as_string() {
  fmt::MemoryWriter w;
  auto tsks = get_tasks();
  w.write("Tasks [ ");
  for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    w.write("<<{}>> ",(*t)->header_as_string());
  }
  w.write("]");
  for (auto t=tsks.begin(); t!=tsks.end(); ++t) {
    w.write("\n");
    w.write("{}: predecessors ",(*t)->header_as_string());
    auto preds = (*t)->get_predecessor_coordinates();
    for ( auto p : preds ) //->begin(); p!=preds->end(); ++p)
      w.write("{}, ",p->as_string());
  }
  return w.str();
};
