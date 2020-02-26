// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** imp_base.h: Header file for the base classes
 ****
 ****************************************************************/

#ifndef IMP_BASE_H
#define IMP_BASE_H 1

/*! \mainpage

  The IMP code implements the Integrative Model for Parallelism. 

  For a descsription of the object hierarchy, see \subpage objects.

  The files imp_base.h and imp_base.cxx contain the mathematical logic of IMP entities;
  these are then specialized for various parallelism models. Currently available:
  - MPI : message passing, defined by mpi_base.h and mpi_base.cxx
  - OpenMP : task model, defined by omp_base.h and omp_base.cxx
  - hybrid : in progress

  Further detailed discussions:

  - \subpage ops
  - \subpage embed
 */

/*! \page apps Applications

  Sample applications are provided that do the following:

  - \subpage cg
  - \subpage gropp
  - \subpage heat
  - \subpage kmeans
  - \subpage nbody
  - \subpage power
  - \subpage threepoint
  - \subpage lulesh
  - \subpage laplace9
  - \subpage prktranspose
  - \subpage sstep
*/

/*! \page embed Object embedding

  We have a basic decision to make regarding the relationship between 
  a vector and a halo. We define a halo object as In(beta), meaning
  all the elements of the input object we need for a local computation.
  Note that the halo is more than the usual halo/ghost region, which is only
  a surrounding band of the local domain.

  \todo rewrite this page

  Embedding is not trivial. Here follow some thoughts.

  If the local domain is simply connected, it can be embedded in the halo.
  However, if the local domain is a contiguous allocation of two disjoint 
  index structures, it can not be so embedded.

  Embedding the local domain in the halo means that self-messages are not
  needed, not even as copy. Thus we could split the local execution
  part of \ref task::execute into two parts, perform one while we wait
  for the true halo to be filled, and then perform the rest, thus effecting
  overlap of computation and communication.

  Question: are tasks executed in the right order, both MPI and OpenMP,
  so that we can always locally start on the next kernel? Do we need a 
  simple test that the local part of the halo is finished?

  Halo embedding is certainly also possible with OpenMP (though watch for
  the synchronization problem of the previous paragraph), but maybe
  not desirable in the case where we have scratchpad memory: then the halo
  is actually strictly local, rather than part of a global object.

  As a result of embedding, an object can own its own storage, or borrow
  from another object. We keep track of that with the \ref object_data_status enum class,
  stored in \ref object::data_status. 
  Routines: \ref object::allocate and \ref object::inherit_data. The allocate routine
  is called in several places, so we can postpone actual allocation pretty far.

  Example: in \ref origin_kernel::origin_kernel we immediately allocate the output
  because this object will never be embedded. 
  Also, in \ref dependency::allocate_halo_vector we immediately allocate.

  In \ref algorithm::inherit_data_from_halos we call \ref object::inherit_data
  on all input objects that have not been allocated yet.

  Embedding is controlled by \ref architecture::can_embed_in_beta, which is
  currently set depending on a commandline argument. I think we should just set that.

*/

#include <stdlib.h>
#include <cstdio>
// #include <stdio.h>
#include <string.h>

#include "fmt/format.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <vector>
#include <string>

#include "utils.h"
#include "imp_static_vars.h"
#include "imp_functions.h"
#include "indexstruct.hpp"

//! \todo write a CHK macro that does not use MPI
#define CHK(x) if (x) {                                          \
  char errtxt[200]; int len=200;                                 \
  MPI_Error_string(x,errtxt,&len);                               \
  printf("p=%d, line=%d, err=%d, %s\n",mytid,__LINE__,x,errtxt); \
  return ;}
#define CHK1(x) if (x) {                                          \
  char errtxt[200]; int len=200;                                 \
  MPI_Error_string(x,errtxt,&len);                               \
  printf("p=%d, line=%d, err=%d, %s\n",mytid,__LINE__,x,errtxt); \
  return NULL;}

#define __ROUTINE__ "Unknown Function"

/****
 **** Basics
 ****/

enum class memoization_status { UNSET, SET, LOCKED };

/*!
  A simple class for containing an entity name; 
  right now only architecture inherits from this; 
  everything else inherits from \ref entity.
  \todo roll this into \ref entity
 */
class entity_name {
private:
  std::string name;
protected:
  static int entity_number;
public:
  entity_name() { name = std::string( fmt::format("entity-{}",entity_number++) ); };
  entity_name( const char *n ) { name = std::string(n); };
  entity_name( std::string n ) { name = std::string(n); };
  entity_name( int n ) : entity_name( fmt::format("entity-{}",n) ) {};
  virtual void set_name( const char *n ) { name = std::string(n); };
  virtual void set_name( std::string n ) { name = std::string(n); };
  const std::string &get_name() const { return name; };
};

/****
 **** Entity
 ****/

/*!
  We use this to track globally what kind of entities we have.
  
  SHELLKERNEL : kernel, except that we do not want it to show up in
          dot files. Stuff like reduction.
 */
enum class entity_cookie { UNKNOWN,
    ARCHITECTURE, COMMUNICATOR, DECOMPOSITION, DISTRIBUTION, MASK,
    SHELLKERNEL, SIGNATURE,
    KERNEL, TASK, MESSAGE, OBJECT, OPERATOR, QUEUE };

enum class trace_level {
  NONE=0, CREATE=1, PROGRESS=2, MESSAGE=4, REDUCT=8
    };

class environment;
/*!
  We define a basic entity class that everyone inherits from.
  This is mostly to be able to keep track of everything:
  there is a static environment member which has a list of entities;
  every newly created entity goes on this list.
*/
class entity : public entity_name {
protected:
  static environment *env;
public:
  // General constructor
  entity( entity_cookie c );
  entity( entity *e,entity_cookie c );
  static void set_env( environment *e );

  /*
   * What kind of entity is this?
   */
protected:
  entity_cookie typecookie{entity_cookie::UNKNOWN};
public:
  void set_cookie( entity_cookie c ) { typecookie = c; };
  entity_cookie get_cookie() const { return typecookie; };
  std::string cookie_as_string() const;

public:
  static trace_level tracing;
  //static bool trace_progress;
public:
  static void add_trace_level( trace_level lvl ) {
    tracing = (trace_level) ( (int)tracing | (int)lvl );
  };
  bool has_trace_level( trace_level lvl ) {
    return (int)tracing & (int)lvl;
  };
  trace_level get_trace_level() { return tracing; };
  bool tracing_progress() { return ((int)tracing & (int)trace_level::PROGRESS)>0; };
  
  /*
   * Statistics: allocation and timing
   */
protected:
  float allocated_space{0.};
public:
  void register_allocated_space( float s ) { allocated_space += s; };
  float get_allocated_space() { return allocated_space; };

  /*
   * Output
   */
  virtual std::string as_string() { //!< Base class method for rendering as string
    return fmt::format("type: {}",cookie_as_string()); }
};

/*!
  An event class. This mostly has begin/end methods and records the duration between.
  To accomodate multiple experiments we accumulate the durations.
*/
class timed_event {
protected:
  double milsec{0};
  std::chrono::system_clock::time_point point;
public:
  timed_event() {};
  void begin() {
    point = std::chrono::system_clock::now();
  };
  void end() {
    auto duration = std::chrono::system_clock::now()-point;
    auto millisec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    milsec += .001 * millisec_duration.count();
  };
  double get_duration() {
    return milsec;
  };
};

/****
 **** Architecture
 ****/

/*!
  Architecture types. We need to edit this list (and edit the definition of
  \ref parallel_indexstruct::parallel_indexstruct) every time we add one.

  - SHARED : OpenMP
  - SPMD : MPI
  - ISLANDS : Product?
  - ??? : Hybrid?
  - ??? : Accelerator?
 */
enum class architecture_type { UNDEFINED,SHARED,SPMD,ISLANDS };
std::string architecture_type_as_string( architecture_type type );

/*!
  Definition of the synchronization protocol. 
  Overlap with \ref architecture_type needs to be figured out.
  Observe partial ordering: MPIplusCrap >= MPI.
 */
enum class protocol_type { UNDEFINED,OPENMP,OPENMPinMPI,MPI,MPIRMA,MPIplusOMP,MPItimesOMP };

/*!
  Collective strategy

  - undefined : set at architecture creation
  - all_ptp : naive by all sends and received
  - group : two level grouping
  - tree : recursive grouping by two
  - MPI : (not implemented) using MPI collectives
 */
enum class collective_strategy{ UNDEFINED,ALL_PTP,GROUP,RECURSIVE,MPI };

class decomposition;

//! Processor coordinates on a pretend grid. We order them by rows.
//! \todo write method to range over this
class processor_coordinate {
private:
  std::vector<int> coordinates;
public:
  processor_coordinate() {}; //!< Default constructor. I don't like it.
  // Create an empty processor_coordinate object of given dimension
  processor_coordinate(int dim);
  // Create coordinate from linearized number, against decomposition
  processor_coordinate(int p,decomposition &dec);
  // from explicit dimension sizes
  processor_coordinate( std::vector<int> dims );
  processor_coordinate( std::vector<index_int> dims );
  // Copy constructor
  processor_coordinate( processor_coordinate *other );

  // basic stats and manipulation
  int get_dimensionality() const; int get_same_dimensionality( int d ) const;
  int &at(int i) { return coordinates.at(i); };
  void set(int d,int v); int coord(int d) const ; int volume() const;

  // operators
  processor_coordinate operator+( index_int i);
  processor_coordinate operator+(processor_coordinate &i);
  processor_coordinate operator-(index_int i);
  processor_coordinate operator-(processor_coordinate &i);
  processor_coordinate operator%(processor_coordinate m);
  bool operator>(index_int i);
  bool operator>(processor_coordinate i);
  bool operator==( processor_coordinate &&other );
  bool operator==( processor_coordinate &other );
  //  bool equals( processor_coordinate* );
  bool equals( processor_coordinate& ); bool equals( processor_coordinate&& );
  int operator[](int d) const { return coord(d); };
  processor_coordinate rotate( std::vector<int> v,processor_coordinate m);
  const std::vector<int> &data() { return coordinates; };
  // linearization
  int linearize( const processor_coordinate&) const; // linear number wrt cube layout
  //  int linearize(processor_coordinate*); // linear number wrt cube layout
  int linearize( const decomposition* ) const; // linear number wrt cube
  // equality operation
  processor_coordinate negate();
  bool is_zero();
  // processor_coordinate operate( const ioperator &op );
  // processor_coordinate operate( const ioperator &&op );
  domain_coordinate operate( const ioperator &op );
  domain_coordinate operate( const ioperator &&op );
  std::string as_string() const;

  bool is_on_left_face(decomposition*) const; bool is_on_right_face(decomposition*) const;
  bool is_on_face(decomposition*) const; bool is_on_face(std::shared_ptr<object>) const;
  bool is_null() const;
  processor_coordinate left_face_proc(int d,processor_coordinate &&farcorner);
  processor_coordinate right_face_proc(int d,processor_coordinate &&farcorner);
  processor_coordinate left_face_proc(int d,processor_coordinate &farcorner);
  processor_coordinate right_face_proc(int d,processor_coordinate &farcorner);

  // operators
  //  domain_coordinate operator*(index_int i);

  // iterating
protected:
  int iterator{-1};
public:
  processor_coordinate& begin() { iterator = 0; return *this; };
  processor_coordinate& end() { return *this; };
  bool operator!=( processor_coordinate ps ) { return iterator<coordinates.size()-1; };
  void operator++() { iterator++; };
  int operator*() const {
    if (iterator<0)
      throw(fmt::format("deref negative iterator {} in {}",iterator,as_string()));
    int v = coordinates[iterator];
    //printf("deref coord @%d to %d\n",iterator,v);
    return v; };
};

class processor_coordinate1d : public processor_coordinate {
public:
  processor_coordinate1d(int p) : processor_coordinate(1) { set(0,p); };
};

//! Make a zero coordinate, in case you need to refer to the origin.
class processor_coordinate_zero : public processor_coordinate {
public:
  processor_coordinate_zero(int d) :  processor_coordinate(d) {
    for (int id=0; id<d; id++) set(id,0); };
  processor_coordinate_zero(decomposition &d);
};

class message;
class distribution;
/*!  
  A base class for architecture data. All accessor routines throw
  an exception by default; they are overridden in the mode-specific
  classes. Interestingly, there is no data in this class; we leave
  that again to the modes.

  We probably need to add access functions here if we introduce new modes.

  \todo reinstate disabled collective strategy handling
  \todo install lambda for getting environment
 */
class architecture : virtual public entity {
protected:
public: // data
  int beta_has_local_addres_space{-1}; //!< \todo why is this public data?
public: // methods
  //! architecture constructor will mostly be called through derived architectures.
  architecture(int m,int n) : architecture(n) { arch_procid = m; };
  architecture(int n) : entity(entity_cookie::ARCHITECTURE) {
    arch_nprocs = n; set_name("some architecture"); };
  //! Copy constructor
  architecture( architecture *a )
    : entity( dynamic_cast<entity*>(a),entity_cookie::ARCHITECTURE) {
    arch_procid = a->arch_procid; arch_nprocs = a->arch_nprocs;
    type = a->type; over_factor = a->over_factor; protocol = a->protocol;
    message_as_buffer = a->message_as_buffer; mytid = a->mytid;
    beta_has_local_addres_space = a->beta_has_local_addres_space;
    can_embed_in_beta = a->can_embed_in_beta; strategy = a->strategy;
  }
  //! A mode-dependent switch to turn on tricky optimizations
  virtual void set_power_mode() {};

  /*
   * Type handling
   */
protected:
  architecture_type type{architecture_type::UNDEFINED};
  architecture *embedded_architecture{nullptr};
public:
  architecture_type get_architecture_type() { return type; };
  int has_type(architecture_type t) { return get_architecture_type()==t; };
  architecture *get_embedded_architecture() {
    if (embedded_architecture==nullptr)
      throw(std::string("embedded architecture is not set"));
    return embedded_architecture; };

  /*
   * Over decomposition
   */
protected:
  int over_factor{1};
public:
  void set_over_factor(int o) { over_factor = o; };
  int get_over_factor() { return over_factor; };

  /*
   * Protocols
   */
protected:
  protocol_type protocol{protocol_type::UNDEFINED};
public:
  protocol_type get_protocol() { return protocol; };
  int get_is_mpi_based() { return protocol>=protocol_type::MPI; };
  void set_protocol_is_embedded() { protocol = protocol_type::OPENMPinMPI; };
  int get_protocol_is_embedded() { return protocol==protocol_type::OPENMPinMPI; };
  std::string protocol_as_string() {
    if (protocol==protocol_type::UNDEFINED) return std::string("UNDEFINED");
    else if (protocol==protocol_type::OPENMP) return std::string("OPENMP");
    else if (protocol==protocol_type::MPI) return std::string("MPI");
    else if (protocol==protocol_type::MPIplusOMP) return std::string("MPIplusOMP");
    else if (protocol==protocol_type::MPItimesOMP) return std::string("MPItimesOMP");
  };
protected:
  int split_execution{0};
public:
  void set_split_execution() { split_execution = 1; };
  int get_split_execution() { return split_execution; };

  /*
   * Processor handling
   */
protected:
  int arch_procid{-1},arch_nprocs{-1};
public:
  int nprocs() {
    if (arch_nprocs<0) throw(std::string("Number of processors uninitialized"));
    return arch_nprocs; };
  //! Processor count of the embedded architecture
  int embedded_nprocs() { return get_embedded_architecture()->nprocs(); };
  //! Total number of processors over top level plus embedded
  int product_nprocs() { return nprocs()*embedded_nprocs(); };
  //! MPI has local identification, OMP doesn't
  std::function< int(void) > mytid{
    [] (void) -> int { throw(std::string("no mytid in general")); } };
  virtual int is_first_proc(int p) { return p==0; };
  virtual int is_last_proc(int p)  { return p==arch_nprocs-1; };
protected:
  processor_coordinate *coordinates{nullptr};
public:
  processor_coordinate *get_proc_origin(int d); // far corner of the processor grid
  processor_coordinate *get_proc_endpoint(int d); // far corner of the processor grid
  processor_coordinate *get_proc_layout(int d); // far corner of the processor grid
  processor_coordinate *get_processor_coordinates() {
    if (coordinates==nullptr) throw(std::string("No coordinates translation set"));
    return coordinates; };
protected:
  int random_sourcing{0}; //!< Random resolution of MPI redundancy.
  int rma{0}; //!< Use MPI one-sided.
public:
  void set_random_sourcing() { random_sourcing = 1; };
  int has_random_sourcing() { return random_sourcing; };
  void set_use_rma() { rma = 1; };
  int get_use_rma() { return rma; };

  /*
   * Various properties
   */
protected:
  int can_embed_in_beta{0},can_message_overlap{0};
public:
  //! Declare that betas are in global space.
  void set_can_embed_in_beta() { set_can_embed_in_beta(1); };
  //! Set the embedding explicitly. Good for the unittests.
  void set_can_embed_in_beta(int can) { can_embed_in_beta = can; };
  int get_can_embed_in_beta() { return can_embed_in_beta; };
  void set_can_message_overlap() { can_message_overlap = 1; };
  int get_can_message_overlap() { return can_message_overlap; };

  /*
   * Collectives
   */
protected:
  collective_strategy strategy{collective_strategy::GROUP};
public:
  collective_strategy get_collective_strategy() { return strategy; };
  int has_collective_strategy(collective_strategy s) { return strategy==s; };
  void set_collective_strategy(int s);
  void set_collective_strategy(collective_strategy s);
  void set_collective_strategy_ptp() { strategy = collective_strategy::ALL_PTP; };
  void set_collective_strategy_group() { strategy = collective_strategy::GROUP; };
  void set_collective_strategy_recursive() { strategy = collective_strategy::RECURSIVE; };
  std::string strategy_as_string() {
    switch (strategy) {
    case collective_strategy::ALL_PTP : return std::string("point-to-point"); break;
    case collective_strategy::GROUP : return std::string("grouped"); break;
    case collective_strategy::RECURSIVE : return std::string("treewise"); break;
    case collective_strategy::MPI : return std::string("using mpi"); break;
    default : return std::string("undefined"); }
  };

  /*
   * Timing
   */
  std::chrono::system_clock::time_point unsynchronized_timer() {
    return std::chrono::system_clock::now(); };
  //! \todo reinstall the barrier
  std::chrono::system_clock::time_point synchronized_timer() {
    //barrier();
    return unsynchronized_timer(); };
  
  /*
   * Tracing and output
   */
protected:
public:
  std::function< void(std::string) > print_trace{ [] (std::string c) -> void { return; } };
  void set_stdio_print() { print_trace = [] (std::string c) { fmt::print("{}\n",c); }; };
  virtual std::string summary();

  /*
   * Distribution factory
   */
  virtual distribution *new_scalar_distribution() {
    throw(std::string("Error: can not call base case")); };

  std::function< void(architecture*,message*,std::shared_ptr<std::string>) > message_as_buffer {
    [] (architecture *a,message *m,std::shared_ptr<std::string> b) -> void {
      throw(std::string("message_as_buffer not set")); }
  };

  // I/O
  virtual std::string as_string() {
    return architecture_type_as_string(type);
  };
};

enum class communicator_mode{ UNKNOWN, MPI, OMP };

//! The communicator class holds the collectives
//! \todo make this inherit from entity? right now it conflicts in class distribution
class communicator { //: public entity {
public:
  communicator() {};
  //!\todo without this it tries destruct the std::function objects...
  ~communicator() {};

public:
  void *communicator_context{nullptr};
  communicator_mode the_communicator_mode{communicator_mode::UNKNOWN};
  std::string communicator_mode_as_string() {
    if (the_communicator_mode==communicator_mode::UNKNOWN)
      return std::string("unknown");
    if (the_communicator_mode==communicator_mode::MPI)
      return std::string("mpi");
    if (the_communicator_mode==communicator_mode::OMP)
      return std::string("omp");
    throw(std::string("Weird communicator mode"));
  };
  //! MPI has local identification, OMP doesn't
  std::function< int(void) > procid {
    [] (void) -> int { throw(std::string("no procid in general")); } };
  std::function< processor_coordinate(decomposition&) > proc_coord {
    [] (decomposition &d) -> processor_coordinate {
      throw(std::string("no proc_coord in general")); } };
  std::function< processor_coordinate(decomposition&&) > proc_coord_rv {
    [] (decomposition &&d) -> processor_coordinate {
      throw(std::string("no proc_coord_rv in general")); } };
  std::function< int(void) > nprocs {
    [] (void) -> int { throw(std::string("no nprocs in general")); } };

  std::function< index_int(index_int) > allreduce {
    [] (index_int i) -> index_int { throw(std::string("No default allreduce")); } };
  std::function< double(double) > allreduce_d {
    [] (double i) -> double { throw(std::string("No default allreduce_d")); } };
  std::function< int(int) > allreduce_and {
    [] (double i) -> double { throw(std::string("No default allreduce_and")); } };
  std::function< std::vector<index_int>(std::vector<index_int>) > reduce_max{
    [] (std::vector<index_int> v) -> std::vector<index_int> {
      throw(std::string("No default max")); } };
  std::function< std::vector<index_int>(std::vector<index_int>) > reduce_min{
    [] (std::vector<index_int> v) -> std::vector<index_int> {
      throw(std::string("No default min")); } };


  std::function< void(int contrib,std::vector<int>&) > gather32{
    [] (int contrib,std::vector<int>&) -> void {
      throw(std::string("No default gather32")); } };
  //! \todo needs to be shared_ptr to stdvector?
  std::function< void(index_int contrib,std::vector<index_int>&) > gather64{
    [] (index_int contrib,std::vector<index_int>&) -> void {
      throw(std::string("No default gather64")); } };
  std::function < std::vector<index_int>*(index_int,int) > overgather{
    [] (index_int contrib,int n) -> std::vector<index_int>* {
      throw(std::string("No default overgather")); } };
  std::function< int(int *senders,int root) > reduce_scatter{
    [] (int *senders,int root) -> int { throw(std::string("No default reducescatter")); } };
  std::function< int(int) > scan_my_first {
    [] (int mine) -> int { throw(std::string("No default scan")); } };

  //! This is called in the copy constructor of \ref distribution
  void copy_communicator( communicator* c ) {
    procid = c->procid; nprocs = c->nprocs; proc_coord = c->proc_coord;
    allreduce = c->allreduce; allreduce_d = c->allreduce_d; allreduce_and = c->allreduce_and;
    gather32 = c->gather32; gather64 = c->gather64; overgather = c->overgather;
    reduce_scatter = c->reduce_scatter; scan_my_first = c->scan_my_first;
    reduce_max = c->reduce_max; reduce_min = c->reduce_min;

    the_communicator_mode = c->the_communicator_mode;
    communicator_context = c->communicator_context;
  };
  void *get_communicator_context() { return communicator_context; };

  /*
   * embedded
   */
protected:
  communicator *embedded_communicator{nullptr};
public:
  communicator *get_embedded_communicator() {
    if (embedded_communicator==nullptr)
      throw(fmt::format("Communicator has no embedded"));
    return embedded_communicator; };
  /*
   * Tracing
   */
  // //! \todo make this a std::function too, so that MPI/OMP can do better reporting.
  // virtual std::string as_string() {
  //   return std::string("communicator");
  // };
};

/****
 **** Decomposition & Domainset
 ****/

class processor_set {
private:
  std::vector< std::shared_ptr<processor_coordinate> > set;
public:
  bool contains( std::shared_ptr<processor_coordinate> p ) {
    for ( auto pp : set )
      if (p->equals(pp.get()))
	return true;
    return false;
  };
  void add( std::shared_ptr<processor_coordinate> p ) {
    if (set.size()>0 &&
	p->get_dimensionality()!=set.at(0)->get_dimensionality())
      throw(fmt::format("Can not add vector of dim {}: previous {}",
			p->get_dimensionality(),set.at(0)->get_dimensionality()));
    if (!contains(p))
      set.push_back(p);
  };
  int size() { return set.size(); };
private:
  int cur{0};
public:
  processor_set &begin() { cur = 0; return *this; };
  processor_set &end() { return *this; };
  bool operator!=( processor_set & s ) { return cur<set.size(); };
  void operator++() { cur += 1; };
  std::shared_ptr<processor_coordinate> &operator*() { return set.at(cur); };
};

class parallel_structure;
/*!
  A decomposition describes the set of domains that live on one numa space.
  For MPI that is just one, for OMP that is all.
  It also provides the translation from multi-dimensionality to simple list.

  \todo we need a class linear_decomposition fo the one-d case; then let parallel_indexstruct inherit from it. but that causes trouble with parallel_structure. Hm.
  \todo can we move that communicator to the distribution class?
*/
class decomposition : public architecture,virtual public entity {
public:
  // Default is use all procs of the architecture in one-d manner.
  decomposition( architecture *arch );
  // with explicit layout
  decomposition( architecture *arch,processor_coordinate &nd );
  decomposition( architecture *arch,processor_coordinate &&nd );
  // copy constructor
  decomposition(decomposition *d);
  ~decomposition() {};

  /*
   * Global info
   */
private:
  //! A vector of the sizes in all the dimensions
  processor_coordinate domain_layout;
  processor_coordinate closecorner,farcorner;
public:
  int get_dimensionality(); virtual int get_same_dimensionality( int d );
  const processor_coordinate &get_domain_layout() const { return domain_layout; };
  processor_coordinate &get_origin_processor();
  processor_coordinate &get_farpoint_processor();
  int get_size_of_dimension(int nd) { return domain_layout.coord(nd); };
  //! \todo do we really need this?
  std::vector<int> get_global_domain_descriptor() { return domain_layout.data(); };

  /*
   * Domain handling
   */
protected:
  std::vector< processor_coordinate > mdomains;
  processor_set known_domains;
public:
  int domains_volume(); //void add_domain(int);
  void add_domain(processor_coordinate); void add_domains(indexstruct*);
  //! Return the domains object, for 1d only
  std::vector< processor_coordinate > &get_domains() { return mdomains; };
  //! Get a domain by local number; see \ref get_local_domain_number for global for translation
  int get_local_domain( int dom ) { return mdomains[dom].coord(0); };
  //! The local number of domains.
  int local_ndomains() { return mdomains.size(); };
  //! Get the local number where this domain is stored. Domains are multi-dimensionally numbered.
  int get_domain_local_number( processor_coordinate &d ) {
    for ( int i=0; i<mdomains.size(); i++)
      if (mdomains[i]==d) return i;
    throw(fmt::format("Domain has no localization"));
  };

  int linearize( const processor_coordinate &p ) const {
    return p.linearize(this);
  };
  processor_coordinate coordinate_from_linear(int p);

protected:
  decomposition *embedded_decomposition{nullptr};
  void copy_embedded_decomposition(decomposition *other) {
    if (other->embedded_decomposition!=nullptr)
      embedded_decomposition = new decomposition(other->embedded_decomposition);
  };
public:
  decomposition *get_embedded_decomposition() {
    if (embedded_decomposition==nullptr)
      throw(std::string("Null embedded decomposition"));
    return embedded_decomposition; };

  /*
   * Factory routines
   */
  std::function< distribution*(index_int) > new_block_distribution{
    [] (index_int i) -> distribution* {
      throw(std::string("base new_block_distr")); } };
  //! Copy the factory routines; this is called in the copy constructor.
  void copy_decomp_factory(decomposition *d) {
    new_block_distribution = d->new_block_distribution;
  };

  /*
   * Ranging
   */
protected:
  int cur_linear{0};
  processor_coordinate cur_coord;
public:
  decomposition &begin();
  decomposition &end();
  void operator++();
  bool operator!=( decomposition& );
  bool operator==( decomposition& );
  processor_coordinate &operator*();
};

enum class environment_type {
  BASE,MPI,OMP,PRODUCT,HYBRID,IR};
#define result_tuple std::tuple<\
				/* 0: object count */ int,		\
				/* 1 : kernel count */ int,		\
				/* 2 : task count */ int,               \
				/* 3 : distribution count */ int,       \
				/* 4 : allocated space */ index_int,    \
				/* 5 : run duration */ double,          \
				/* 6 : analysis time */ double,         \
				/* 7 : message count */ int,            \
				/* 8 : message volume */ double,        \
				/* 9 : flop count */ double \
				>
#define RESULT_OBJECT 0
#define RESULT_KERNEL 1
#define RESULT_TASK 2
#define RESULT_DISTRIBUTION 3
#define RESULT_ALLOCATED 4
#define RESULT_DURATION 5
#define RESULT_ANALYSIS 6
#define RESULT_MESSAGE 7
#define RESULT_WORDSENT 8
#define RESULT_FLOPS 9

/*!
  An environment describes the processor structure. 

  There is a feeble attempt 
  to include profiling information in this object.

  \todo make \ref set_command_line virtual with default, override in mpi case
*/
class environment : public entity_name {
private:
protected:
  int debug_level{0};
  // tracing
  std::vector<double> execution_times;
  double flops{0.};
  int ntasks_executed;
public:
  // Store the commandline and look for the debug parameter
  environment(int argc,char **argv);
  // Reporting and cleanup
  ~environment();
  environment( environment& other ) { //! Copy constructor
    arch = other.arch; type = other.type; ir_outputfile = other.ir_outputfile;
    strategy = other.strategy; delete_environment = other.delete_environment;
    allreduce = other.allreduce; allreduce_d = other.allreduce_d; allreduce_and = other.allreduce_and;
    gather32 = other.gather32; gather64 = other.gather64;
  };
  //! In case we want to delete things in reverse order: MPI_Finalize needs to be called last.
  std::function< void(void) > delete_environment{
    [] () -> void { printf("default delete\n"); return; } };

protected:
  int strategy{0};
public:
  int get_collective_strategy() { return strategy; };

  /*
   * Type
   */
protected:
  environment_type type{environment_type::BASE};
public:
  int check_type_is( environment_type t ) { return type==t; };
  int is_type_mpi() { return type==environment_type::MPI; };
  int is_type_omp() { return type==environment_type::OMP; };

  /*
   * Collectives in the environment
   */
  std::function< index_int(index_int) > allreduce { [] (index_int i) -> index_int { return i; } };
  std::function< index_int(index_int) > allreduce_d { [] (double i) -> double { return i; } };
  std::function< int(int) > allreduce_and { [] (double i) -> double { return i; } };
    
  std::function< void(int contrib,std::vector<int>&) > gather32 {
    [] (int c,std::vector<int>&) -> void { throw(std::string("No default gather32")); } };
  std::function< void(index_int contrib,std::vector<index_int>&) > gather64 {
    [] (index_int c,std::vector<index_int>&) -> void { throw(std::string("No default gather64")); } };
  std::function < std::vector<index_int>*(index_int,int) > overgather {
    [] (index_int c,int o) -> std::vector<index_int>* { throw(std::string("No default gather")); } };
  std::function< int(int *senders,int root) > reduce_scatter{nullptr};
  
  /*
   * Architecture
   */
protected:
  //! The actual architecture is created in the mode-specific environments
  architecture *arch{nullptr};
public:
  virtual architecture *make_architecture() = 0;
  architecture *get_architecture() { return arch; };
  architecture *get_embedded_architecture() { return arch->get_embedded_architecture(); };
  virtual void get_comm(void*) {
    throw(std::string("Get Comm not implemented")); };

  /*
   * Commandline
   */
protected:
  int nargs{0}; char **the_args{nullptr};
public:
  void set_command_line(int argc,char **argv) { nargs = argc; the_args = argv; };
  bool has_argument(const char*);
  //! Return an integer commandline argument with default value
  virtual int iargument(const char*,int);
  static std::function< void(void) > print_application_options;
  virtual void print_options();
protected:
  std::vector< std::string > internal_args; // for now unused.
public:
  int hasarg_from_internal(std::string a) {
    for (auto arg : internal_args)
      if (arg==a) return 1;
    return 0;
  };

  // profiling
  void register_execution_time(double);
  void register_flops(double);
  void record_task_executed();
  virtual void print_stats() {}; // by default no-op
#define DEBUG_STATS 1
#define DEBUG_PROGRESS 2
#define DEBUG_VECTORS 4
#define DEBUG_MESSAGES 8
  int get_debug_level() { return debug_level; };

  /*
   * Printable output
   */
protected:
  int do_printing{1};
public:
  //! Test whether this environment does printing; this gets disabled on MPI for mytid>0.
  void set_is_printing_environment( int p=1 ) { do_printing = p; };
  //! Are we a printing environment?
  int get_is_printing_environment() { return do_printing; };
protected:
  std::string ir_outputfilename{"dag.ir"};
  FILE *ir_outputfile{nullptr};
  std::string *indentation{new std::string};
public:
  virtual void set_ir_outputfile( const char *n );
  void close_ir_outputfile() {
    if (ir_outputfile!=nullptr) { fclose(ir_outputfile); ir_outputfile = nullptr; };
  };
  //! By default, return a string counting the global number of processors
  virtual std::string as_string();
  //! By default print to standard out
  void increase_indent() { indentation->push_back(' '); indentation->push_back(' '); };
  void decrease_indent() { indentation->resize( indentation->size()-2 ); };
    //indentation->pop_back(); indentation->pop_back(); };
  void print_line( std::string );
  void open_bracket(); void close_bracket();
  //void print_object_line( char *s );
  virtual void print_to_file( std::string );
  virtual void print_to_file( const char* );
  virtual void print_to_file( int, std::string );
  virtual void print_to_file( int, const char* );
  virtual void print() { this->print_to_file( this->as_string() ); };
  virtual void print_single(std::string s) { print_to_file(s); };
  virtual void print_all(std::string s) { print_to_file(s); };

  /*
   * Entities
   */
protected:
  static std::vector<entity*> list_of_all_entities;
public:
  void push_entity( entity *e ); int n_entities() const; void list_all_entities();
  // basic summary
  result_tuple *local_summarize_entities();
  // mode-specific summary is by default the basic one, but see MPI!
  std::function< result_tuple*(void) > mode_summarize_entities{
    [this] (void) -> result_tuple* { return local_summarize_entities(); } };
  double get_allocated_space(); int nmessages_sent(result_tuple*);
  std::string summary_as_string( result_tuple *results );
  void print_summary();
  std::string kernels_as_dot_string();
  void kernels_to_dot_file();
  std::string tasks_as_dot_string();
  virtual void tasks_to_dot_file(); // more complicated with MPI
};

/*! \page tracing Tracing

  There is a \ref tracer object, from which \ref kernel (and thereby \ref task) 
  and \ref algorithm inherit.

  There are two enum classes: \ref exec_trace_level and \ref comm_trace_level
  for which the tracer object stores values.

 */

/*!
  Different levels of execution tracing.
  - NONE : none
  - EXEC : report execution
  - VALUES : when executing, print out values
  - VALUESOUTANDINT : also inputs
 */
enum class exec_trace_level { NONE,EXEC,VALUES,VALUESOUTANDIN };
enum class comm_trace_level { NONE,EXEC };

//! An object for storing trace information
class tracer {
protected:
  exec_trace_level execlevel{exec_trace_level::NONE};
  comm_trace_level commlevel{comm_trace_level::NONE};
  int proc_to_trace{-1};
public:
  //! We need an empty constructor for the task(int,int) constructor
  tracer() {};
  //! Set the level of execution tracing
  void set_exec_trace_level( exec_trace_level l=exec_trace_level::EXEC ) { execlevel = l; };
  //! Increase the exec trace level, for instance setting the task level to a queue level
  void increase_exec_trace_level_to( exec_trace_level l ) { if (execlevel<l) execlevel = l; };
  exec_trace_level get_exec_trace_level() const { return execlevel; };
  //! Set the level of communication tracing
  void set_comm_trace_level( comm_trace_level l ) { commlevel = l; };
  //! Increase the comm trace level, for instance setting the task level to a queue level
  void increase_comm_trace_level_to( comm_trace_level l ) { if (commlevel<l) commlevel = l; };
  comm_trace_level get_comm_trace_level() const { return commlevel; };
  //! Limit the tracing to one processor
  // void set_proc_to_trace(int p) { proc_to_trace = p; };
  // int get_proc_has_trace(int p) { return proc_to_trace<0 || proc_to_trace==p; };
  // int get_tracing_output_values() { return get_exec_trace_level()>=exec_trace_level::VALUES; };
  // int get_tracing_input_values() { return get_exec_trace_level()>=exec_trace_level::VALUESOUTANDIN; };
};

/****
 **** Distribution
 ****/

//! The list of allowed distribution types.
enum class distribution_type {
  UNDEFINED, //!< initial
    REPLICATED, //!< identical on every processor
    CONTIGUOUS, //!< contiguous blocks
    BLOCKED, //!< contiguous on each proc, but globally not disjoint
    CYCLIC, //!< cyclic assignment
    GENERAL //!< otherwise, for instance when constructed as beta
    };
std::string distribution_type_as_string(distribution_type t);
distribution_type max_type(distribution_type t1,distribution_type t2);

//! Do we do messages to self?
enum class self_treatment { INCLUDE, EXCLUDE, ONLY };

/*! 
  A parallel_indexstruct is the bookkeeping part of a \ref distribution object.

  At the moment, this contains an array of \ref indexstruct objects, one per
  processor. The case where only the local processor is known is not handled
  terribly elegantly.

  \todo why does this inherit from decomposition? decomp can be multi-d, this can not.
*/
class parallel_indexstruct {
private :
protected:
  std::vector< std::shared_ptr<indexstruct> > processor_structures;
public: // methods
  // Basic constructor on specified number of domains.
  parallel_indexstruct(int nd);
  // Copy constructor
  parallel_indexstruct( parallel_indexstruct *other );

  int pidx_domains_volume() const;
  int size() const;
  int locally_get_nprocs() { return processor_structures.size(); }; // do not use this one!
  void create_from_global_size(index_int gsize);
  //void create_from_indexstruct(indexstruct*);
  void create_from_indexstruct(std::shared_ptr<indexstruct>);
  void create_from_uniform_local_size(index_int lsize);
  //void create_from_unique_local_size(index_int);
  void create_from_local_sizes( std::vector<index_int> );
  void create_from_replicated_local_size(index_int lsize);
  void create_from_replicated_indexstruct( std::shared_ptr<indexstruct>);
  void create_cyclic(index_int lsize,index_int gsize);
  void create_blockcyclic(index_int bs,index_int nb,index_int gsize);
  void create_from_explicit_indices(index_int*,index_int**);
  void create_from_function( index_int(*)(int,index_int),index_int ); // from p,i
  void create_by_binning( object *o, double mn, double mx, int id );

  // messing with the processor structures
  std::shared_ptr<indexstruct> get_processor_structure(int p) const;
  void set_processor_structure(int p,std::shared_ptr<indexstruct> pstruct);
  void set_local_pstruct(std::shared_ptr<indexstruct> pstruct) {
    local_structure = pstruct; };
  void extend_pstruct(int p,indexstruct *i);
protected:
  std::shared_ptr<indexstruct> local_structure; // you know who you are
public:

  // query
  int equals(parallel_indexstruct*) const;
  index_int first_index(int) const; index_int last_index(int) const;
  index_int global_first_index() const; index_int global_last_index() const;
  index_int local_size(int);

protected: index_int computed_global_size{-1};
  void uncompute_internal_quantities() { computed_global_size = -1; };
public:
  index_int global_size(); index_int outer_size() const;
  std::shared_ptr<indexstruct> get_enclosing_structure() const;
  bool is_valid_index(index_int) const; int contains_element(int,index_int) const;
  int find_index(index_int); int find_index(index_int,int);
  std::shared_ptr<parallel_indexstruct> operate( const ioperator &op );
  std::shared_ptr<parallel_indexstruct> operate( const ioperator &&op );
  std::shared_ptr<parallel_indexstruct> operate( ioperator&,std::shared_ptr<indexstruct>);
  std::shared_ptr<parallel_indexstruct> operate( const sigma_operator& ) ;
  //  std::shared_ptr<parallel_indexstruct> operate(std::shared_ptr<sigma_operator>) ;
  std::shared_ptr<parallel_indexstruct> operate_base( const ioperator& );
  std::shared_ptr<parallel_indexstruct> operate_base( const ioperator&& );
  std::shared_ptr<parallel_indexstruct> struct_union( std::shared_ptr<parallel_indexstruct> );
  std::string as_string() const;

  // is this structure known globally?
protected:
  //  bool known_status{true};
public:
  bool is_known_globally();

  /*
   * Type handling
   */
protected:
  distribution_type type{distribution_type::UNDEFINED};
public:
  void set_type( distribution_type t ) {
    if (t==distribution_type::UNDEFINED)
      throw(std::string("Should not set undefined type of pidx"));
    type = t; };
  distribution_type get_type() { return type; };
  int has_type(distribution_type t) { return get_type()==t; };
  bool can_detect_type(distribution_type);
  distribution_type infer_distribution_type();

  int has_type_replicated() { return has_type(distribution_type::REPLICATED); };
  int has_type_contiguous() { return has_type(distribution_type::CONTIGUOUS); };
  int has_type_blocked() { return has_type(distribution_type::BLOCKED); };
  int has_type_locally_contiguous() {
    return ( has_type_contiguous() || has_type_blocked() ); }; 
  int has_type_general() { return has_type(distribution_type::GENERAL); };
  int has_defined_type() { return get_type()!=distribution_type::UNDEFINED; };
  std::string type_as_string() { return distribution_type_as_string(get_type()); };
};

/****
 **** Distribution
 ****/

enum class Fuzz { YES,NO,MAYBE };

/*!
  A processor mask indicates a subset of processors
  out of the totality that is contained in an \ref environment.
  A \ref distribution can optionally contain
  a processor mask to indicate that the distribution is only defined on the
  processors included in the mask.

  \todo Should we have a creator from an \ref indexstruct?
  \todo Can this be used in SPMD mode where each processor only adds itself?
  \todo optimize: freeze when set, then make array of `lives_on' values.
*/
class processor_mask : public decomposition,virtual public entity {
private:
  std::vector<Fuzz> included;
  //  Fuzz *include1d{nullptr}; Fuzz **include2d{nullptr};
public:
  // Create an empty mask
  processor_mask(decomposition *d);
  // create a mask with the first P processors added
  processor_mask( decomposition *d,int P );
  // Create a mask from a list of integers.
  processor_mask( decomposition *d, std::vector<int> procs );
  // Copy constructor.
  processor_mask( processor_mask& other );

  // Add a processor to the mask
  void add(processor_coordinate &p);
  // Remove a processor from the mask; this only makes sense for the constructor from P.
  void remove(int p);
  // test aliveness
  int lives_on(processor_coordinate &p);
  // Render mask as list of integers. This is only used in \ref mpi_distribution::add_mask.
  std::vector<int> get_includes();
};

/*!
  A parallel_structure object is an array of \ref parallel_indexstruct objects,
  one for each dimension. There are shortcuts for the common case of one-d.
  \todo memoize the enclosing struct
  \todo the virtual inheritance is not needed. remove
  \todo the default destructor tries to destruct the lambdas. why?
*/
class parallel_structure : public decomposition {
public:
  parallel_structure(decomposition *d); // constructor from the bottom up
  parallel_structure(decomposition *d,std::shared_ptr<parallel_indexstruct> pidx);
  parallel_structure(parallel_structure *d); // copy-ish constructor
  ~parallel_structure() { /* fmt::print("destructing parallel structure\n"); */ };
  std::vector<std::shared_ptr<parallel_indexstruct>> get_dimension_structures() {
    return dimension_structures; };

  // is this structure known globally?
protected:
  bool known_globally{false};
public:
  void set_is_known_globally( bool v=true ) { known_globally = v; };
  bool is_known_globally() { return known_globally; };

protected:
  bool is_converted{false}, is_orthogonal{true};
public:
  bool get_is_orthogonal() { return is_orthogonal; }
  void set_is_orthogonal() { set_is_orthogonal(true); };
  void set_is_orthogonal(bool set) { is_orthogonal = set; };
  bool get_is_converted() { return is_converted; }
  void set_is_converted() { set_is_converted(true); };
  void set_is_converted(bool set) { is_converted = set; };

  void allocate_structure(); // insert actual pidx objects
protected:
  std::vector<std::shared_ptr<parallel_indexstruct>> dimension_structures;
  const std::vector<std::shared_ptr<parallel_indexstruct>> &get_dimension_structures() const {
    return dimension_structures; };
  std::vector<std::shared_ptr<multi_indexstruct>> multi_structures;
public:
  void push_dimension_structure(std::shared_ptr<parallel_indexstruct> pidx);
  void set_dimension_structure(int d,std::shared_ptr<parallel_indexstruct> pidx);
  std::shared_ptr<parallel_indexstruct> get_dimension_structure(int d);
  void convert_to_multi_structures();
  
protected:
  std::shared_ptr<multi_indexstruct> enclosing_structure{nullptr},global_structure{nullptr};
  memoization_status enclosing_structure_is_computed{memoization_status::UNSET},
    global_structure_is_computed{memoization_status::UNSET};
public:
  std::shared_ptr<multi_indexstruct> compute_global_structure();
  void set_enclosing_structure(std::shared_ptr<multi_indexstruct>);
  void set_global_structure(std::shared_ptr<multi_indexstruct>);
  void deduce_global_structure() {}; // VLE hm?
  std::shared_ptr<multi_indexstruct> get_enclosing_structure();
  std::shared_ptr<multi_indexstruct> get_global_structure();

  void lock_global_structure(std::shared_ptr<multi_indexstruct>);
  bool global_structure_is_locked() {
    return global_structure_is_computed==memoization_status::LOCKED; }

  std::shared_ptr<multi_indexstruct> get_processor_structure( processor_coordinate* );
  std::shared_ptr<multi_indexstruct> get_processor_structure( processor_coordinate& );
  void set_processor_structure(int p,std::shared_ptr<indexstruct> pstruct);
  void set_processor_structure( processor_coordinate *p,std::shared_ptr<multi_indexstruct> );
  void set_processor_structure( processor_coordinate &p,std::shared_ptr<multi_indexstruct> );

protected:
  domain_coordinate stored_global_first_index,stored_global_last_index;
public:
  const domain_coordinate &global_first_index();
  index_int global_first_index(int id) { return global_first_index()[id]; };
  const domain_coordinate &global_last_index();
  index_int global_last_index(int id) { return global_last_index()[id]; };

  // Support functions
  index_int linearize( domain_coordinate *coord ) { return linearize(*coord); }
  index_int linearize( domain_coordinate &coord );
  /*
   * Access
   */
  // domain_coordinate &first_index_r(processor_coordinate *p);
  // domain_coordinate &last_index_r(processor_coordinate *p);
  domain_coordinate &first_index_r(processor_coordinate &p);
  domain_coordinate &last_index_r(processor_coordinate &p);
  domain_coordinate &first_index_r(processor_coordinate &&p);
  domain_coordinate &last_index_r(processor_coordinate &&p);

  domain_coordinate last_index(processor_coordinate *p);
  domain_coordinate &global_size();
  domain_coordinate &local_size_r(processor_coordinate p);
  //  index_int volume(processor_coordinate*);
  index_int volume(processor_coordinate&&);
  index_int volume(processor_coordinate&);
  index_int global_volume();
  bool is_valid_index(domain_coordinate&);
  bool is_valid_index(domain_coordinate&&);

  /*
   * Type handling
   */
protected:
  distribution_type type{distribution_type::UNDEFINED};
public:
  void set_type( distribution_type t ) {
    if (t==distribution_type::UNDEFINED)
      throw(fmt::format("Should not set undefined type in <<{}>>",get_name()));
    type = t; };
  distribution_type get_type() { return type; };
  int has_type(distribution_type t) { return get_type()==t; };
  void require_type(distribution_type);
  int has_type_replicated() { return has_type(distribution_type::REPLICATED); };
  void require_type_replicated() { require_type(distribution_type::REPLICATED); };
  int has_type_contiguous() { return has_type(distribution_type::CONTIGUOUS); };
  int has_type_blocked() { return has_type(distribution_type::BLOCKED); };
  int has_type_locally_contiguous() {
    return ( has_type_contiguous() || has_type_blocked() ); }; 
  int has_type_general() { return has_type(distribution_type::GENERAL); };
  int has_defined_type() { return get_type()!=distribution_type::UNDEFINED; };
  distribution_type infer_distribution_type();
  int can_detect_type(distribution_type);
  //! \todo make this a multi-dimensional report
  std::string type_as_string() { return distribution_type_as_string(get_type()); };

  /*
   * Creation
   */
  void create_from_indexstruct( std::shared_ptr<multi_indexstruct> idx);
  void create_from_indexstruct(multi_indexstruct *idx);
  void create_from_replicated_indexstruct(multi_indexstruct *idx);
  // one-d shortcuts:
  void create_from_global_size(index_int gsize);
  void create_from_global_size(std::vector<index_int> gsizes);
  void create_from_indexstruct(std::shared_ptr<indexstruct>);
  void create_from_uniform_local_size(index_int lsize);
  void create_from_local_sizes( std::vector<index_int> szs );
  void create_from_replicated_local_size(index_int lsize);
  void create_from_replicated_indexstruct( std::shared_ptr<indexstruct> );
  void create_cyclic(index_int lsize,index_int gsize);
  void create_blockcyclic(index_int bs,index_int nb,index_int gsize);
  void create_from_explicit_indices(index_int *nidx,index_int **idx);
  void create_from_function( index_int(*f)(int,index_int),index_int n);
  void create_by_binning( object *o );

  /*
   * Operations
   */
  parallel_structure *operate( const ioperator& );
  parallel_structure *operate( const ioperator&& );
  parallel_structure *operate(ioperator&,std::shared_ptr<multi_indexstruct>);
  parallel_structure *operate( multi_ioperator *op );
  parallel_structure *operate(multi_sigma_operator*);
  parallel_structure *operate_base( const ioperator &op );
  parallel_structure *operate_base( const ioperator &&op );
  parallel_structure *struct_union( parallel_structure *merge );

  std::string header_as_string();
  std::string as_string();
};

class object;
class message;
class kernel;
class distribution_sigma_operator;
/*!
  A distribution has much the same functionality as a \ref parallel_indexstruct.
  Additionally, it has functions for analyzing dependencies and deriving messages.

  \todo I have commented out the first/last index, so there is no mask checking going on. do we do that higher up?
*/
class distribution
  : public parallel_structure,virtual public communicator, virtual public entity {
protected:
  int orthogonal_dimension{1};
private:
public:
  distribution(decomposition *d);
  distribution(parallel_structure *ps);
  distribution(decomposition *d,std::shared_ptr<parallel_indexstruct>);
  distribution( distribution *d ); // copy-ish constructor
  ~distribution() { /* fmt::print("destructing distribution\n"); */ };
  
  void create_from_unique_local(std::shared_ptr<multi_indexstruct>);
  
protected:
  std::shared_ptr<multi_indexstruct> the_numa_structure{nullptr};
  domain_coordinate the_numa_first_index = domain_coordinate(1);
  index_int the_numa_local_size{-1};
public:
  bool numa_structure_is_computed{false};
  std::function< void(distribution *d) > compute_numa_structure{
    [] (distribution *d) -> void {
      throw(std::string("No compute numa structure defined")); } };
  void set_numa_structure( std::shared_ptr<multi_indexstruct> n );
  void set_numa_structure
      ( std::shared_ptr<multi_indexstruct> n,const domain_coordinate &f,index_int s);
  std::shared_ptr<multi_indexstruct> get_numa_structure();
  indexstruct *get_numa_structure(int id);
  const domain_coordinate offset_vector();
  const domain_coordinate &numa_size_r();
  
protected:
  // these functions store in the parallel_structure
  std::function< domain_coordinate(distribution*) > compute_global_first_index{nullptr};
  std::function< domain_coordinate(distribution*) > compute_global_last_index{nullptr};
public:
  void memoize();

  /*
   * Factories
   */
  void set_dist_factory(); // for the initial constructor
  void copy_dist_factory(distribution *d); // for the copy constructor
  //! Distribution factory. \todo there is a routine like this in decomposition....
  std::function < distribution*(parallel_structure*) > new_distribution_from_structure {
    [] (parallel_structure *strc) -> distribution* {
      throw(std::string("Unimplemented new_distribution_from_structure")); } };
  //! Distribution factory. \todo there is a routine like this in decomposition....
  std::function < distribution*(std::shared_ptr<multi_indexstruct>) >
  new_distribution_from_unique_local {
    [] (std::shared_ptr<multi_indexstruct> strc) -> distribution* {
      throw(std::string("Unimplemented new_distribution_from_unique_local")); } };
  //! Factory method for making mode-dependent objects \todo can this go when we have new_dist?
  std::function < distribution*() > new_scalar_distribution {
    [] (void) -> distribution* {
      throw(std::string("No new_scalar_distribution iplemented")); } };
  //! Factory method for making mode-dependent objects \todo can this go when we have new_dist?
  std::function < std::shared_ptr<object>(distribution*) > new_object;
  //! Factory method for making mode-dependently reusing objects
  std::function < std::shared_ptr<object>(double*) > new_object_from_data;
  //! Factory for kernels. We need this in the imp_ops.h file.
  std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object> ) > kernel_from_objects{
    [] (std::shared_ptr<object> in,std::shared_ptr<object> out) -> kernel* {
      throw(std::string("No kernel_from_objects defined")); } };
  //! Factory for making kernels. \todo this is only for standalone task testing
  std::function < kernel*(std::shared_ptr<object>) > new_kernel_from_object{nullptr};
  //! Message factory
  std::function< message*(processor_coordinate &snd,processor_coordinate &rcv,
			  std::shared_ptr<multi_indexstruct> g) > new_message;
  //! Message factory
  std::function< message*(processor_coordinate &snd,processor_coordinate &rcv,
			  std::shared_ptr<multi_indexstruct> e,std::shared_ptr<multi_indexstruct> g) > new_embed_message;
public:
  //! Used in \ref omp_object::mask_shift
  index_int internal_local_size(int p) { throw(std::string("Wrong for non-ortho"));
    return get_dimension_structure(0)->local_size(p); };
  //! \todo lose this, replace by global_volume
  index_int outer_size() { return global_volume(); };
  
  // numa stuff is sorta-virtual
  std::function < const domain_coordinate(void) > numa_first_index;
  std::function < index_int(distribution &d,processor_coordinate &p) > location_of_first_index {
    [] (distribution &d,processor_coordinate &p) -> index_int {
      throw(std::string("no loc of first")); } };
  std::function < index_int(distribution &d,processor_coordinate &p) > location_of_last_index {
    [] (distribution &d,processor_coordinate &p) -> index_int {
      throw(std::string("no loc of last")); } };
  std::function < index_int(void) > numa_local_size;
  
protected:
  // these things come from gather operations, so write those to write into pass-by-ref vectors.
  std::vector<int> linear_sizes, linear_starts, linear_offsets;
  bool has_linear_data{false};
  std::vector<bool> processor_skip;
public:
  void compute_linear_sizes(); void compute_linear_offsets();
  std::vector<int> &get_linear_sizes(); std::vector<int> &get_linear_offsets();
  int get_processor_skip(int p) { return processor_skip.at(p); };
  
  /*
   * Allocation matters
   */
  index_int local_allocation_p( processor_coordinate &p); //allocation amount multi-d
  //! Local allocation independent of processor is mode-dependent
  std::function < index_int(void) > local_allocation {
    [] (void) -> index_int { throw(std::string("imp_base.h local allocation")); } };
  //! Global allocation is needed for OMP; for MPI it's just a statistic. \todo wrong w/ masks
  index_int global_allocation() {
    return global_volume()*get_orthogonal_dimension();
  };
  // \todo this can go. it's basically get_numa_structure
  std::function< std::shared_ptr<multi_indexstruct>(processor_coordinate&) > get_visibility {
    [this] (processor_coordinate &p) -> std::shared_ptr<multi_indexstruct> {
      throw(fmt::format("no get_visibility defined")); } };
  
  //! Set the non-distributed dimension of the distribution.
  void set_orthogonal_dimension(index_int o) {
    if (o<=0) throw(std::string("setting nonpositive orthogonal"));
    orthogonal_dimension = o; };
  //! Return the non-distributed dimension of the distribution.
  index_int get_orthogonal_dimension() {
    if (orthogonal_dimension<1) throw(std::string("Strange ortho dimension"));
    return orthogonal_dimension; };
  
  int equals(distribution *d);    
  
  bool contains_element(processor_coordinate *p,domain_coordinate &i) {
    return contains_element(*p,i); }
  bool contains_element(processor_coordinate &p,domain_coordinate &i);
  bool contains_element(processor_coordinate &p,domain_coordinate &&i);
  //  bool contains_element(int p,index_int i);
  int find_index(index_int i)        { return get_dimension_structure(0)->find_index(i); };
  int find_index(index_int i,int p)  { return get_dimension_structure(0)->find_index(i,p); };
  //! \todo can we get rid of these lambdas and use new_distribution_from_structure
  void copy_operate_routines( distribution *d ) {
  }
  distribution *operate(multi_sigma_operator *op);
  distribution *operate(distribution_sigma_operator*);
  distribution *operate( const ioperator& ); distribution *operate( const ioperator&& );
  distribution *operate(multi_ioperator*); 
  distribution *operate_trunc( ioperator &op ,std::shared_ptr<multi_indexstruct> trunc);
  distribution *operate_base( const ioperator &op );
  distribution *operate_base( const ioperator &&op );
  distribution *distr_union( distribution *other );
  // couple of operate functions that are polymorphic
  // std::function< distribution*( std::shared_ptr<ioperator> op ) > operate_base {
  //   [] (std::shared_ptr<ioperator> op) ->distribution* {
  //     throw(std::string("no method op_base")); } };
  // std::function< distribution*( distribution *other ) > distr_union {
  //[] (distribution *other) -> distribution* { throw(std::string("no method distr_union")); } };
  distribution *extend( processor_coordinate*,std::shared_ptr<multi_indexstruct>);
  distribution *extend( processor_coordinate,std::shared_ptr<multi_indexstruct>);
  
  /*
   * Mask
   */
protected:
  processor_mask *mask{nullptr};
public:
  int has_mask() { return mask!=nullptr; };
  //! Add a mask; MPI will do some reconciliation first
  virtual void add_mask( processor_mask *m ) { mask = m; };
  //! Test if this distrubution is present on a certain processor
  int lives_on( processor_coordinate &p ) { return !has_mask() || mask->lives_on(p); };
  //! Throw an error if does not live on p;
  //! this is used in \ref distribution::first_index and such.
  void force_lives_on(processor_coordinate &p) {
    if (!lives_on(p))
      throw(fmt::format("Distribution does not live on {}",p.as_string())); };
  processor_mask *get_mask() {
    if (!has_mask()) throw(std::string("No mask to be got"));
    return mask; };
  //! Count how many processors are actually present in this distribution.
  int n_live_procs() { int n = 0;
    for (int p=0; p<domains_volume(); p++) {
      auto pn = processor_coordinate( std::vector<int>{p} );
      if (lives_on(pn)) n++;
    }
    return n; };

  /*
   * Analysis
   */
  std::vector<message*> *analyze_one_dependence
          (int,int,ioperator*,distribution*,indexstruct*);
  std::vector<message*> *analyze_one_dependence(int mytid,distribution *beta);
  std::vector<message*> messages_for_segment
      (processor_coordinate&,self_treatment,
       std::shared_ptr<multi_indexstruct>,std::shared_ptr<multi_indexstruct>);
  virtual std::string as_string();
};

/*!
  A block distribution has a contiguous block on each processor.
  We define this from a global size or local size. 
  At the moment there is not yet a mechanism for indicatig the thread-local size.
 */
class block_distribution : virtual public distribution, virtual public communicator {
protected:
  //! Pre-basic constructor
  block_distribution(decomposition *d)
    : distribution(d),entity(entity_cookie::DISTRIBUTION) {
    fmt::print("prebasic block\n");
  };
public:
  //! Constructor from multiple explicit parameters
  block_distribution(decomposition *d,int ortho,index_int lsize,index_int gsize)
    : distribution(d),entity(entity_cookie::DISTRIBUTION) {
    //fmt::print("create from sizes {} {} {}\n",ortho,lsize,gsize);
    if (gsize>0) {
      try {
	create_from_global_size(gsize);
      } catch (std::string c) {
	throw(fmt::format("Block distribution from g={} failed: {}",gsize,c));
      }
    } else {
      if (lsize<0)
	throw(std::string("Need nonnegative local size if global==0"));
      create_from_local_sizes( *(overgather(lsize,get_over_factor())) );
    }
    set_orthogonal_dimension(ortho);
  };
  //! Constructor from implicit ortho
  block_distribution(decomposition *d,index_int lsize,index_int gsize)
    : block_distribution(d,1,lsize,gsize) {};
  //! Constructor from implicit ortho and local
  block_distribution(decomposition *d,index_int gsize)
    : block_distribution(d,-1,gsize) {};
  //! Constructor for multi-d: we supply a vector of global sizes
  block_distribution( decomposition *d,std::vector<index_int> gsizes )
    : distribution(d),entity(entity_cookie::DISTRIBUTION) {
    //fmt::print("create from vector of sizes\n");
    d->get_same_dimensionality(gsizes.size());
    try {
      create_from_global_size(gsizes);
    } catch (std::string c) {
      throw(fmt::format("Block distribution from vector of global, create failed with: {}",c));
    }
  };
  /*! Constructor one-d with explicit blocksizes. The gsize argument
    is ignored, but is needed to distinguish from the multi-d constructor from 
    a vector or gsizes
  */
  block_distribution( decomposition *d,std::vector<index_int> lsizes,index_int gsize )
    : distribution(d),entity(entity_cookie::DISTRIBUTION) {
    if (get_dimensionality()>1)
      throw(fmt::format("Can not create from local sizes in d={}",get_dimensionality()));
    if (lsizes.size()!=d->domains_volume())
      throw(fmt::format("Number of local sizes {} mismatch processors {}",
			lsizes.size(),d->domains_volume()));
    create_from_local_sizes(lsizes);
  };
  block_distribution( decomposition *d,std::shared_ptr<multi_indexstruct> idx )
    : distribution(d),entity(entity_cookie::DISTRIBUTION) {
    if (get_dimensionality()>1)
      throw(fmt::format("Can not create from local sizes in d={}",get_dimensionality()));
    create_from_indexstruct(idx);
  };
};

/*! A scalar distribution is the special case of a block distribution
  with block size 1.
*/
class scalar_distribution : public block_distribution {
public:
  scalar_distribution(decomposition *d)
    : distribution(d),block_distribution(d,1,-1),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {};
};

/*!
  Cyclic distributions are very much under construction.
 */
class cyclic_distribution : virtual public distribution {
public:
  cyclic_distribution(decomposition *d,index_int lsize,index_int gsize)
    : distribution(d),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    create_cyclic(lsize,gsize); };
};

/*!
  Blockcyclic distributions are very much under construction.
 */
class blockcyclic_distribution : virtual public distribution {
public:
  blockcyclic_distribution(decomposition *d,index_int bs,index_int nb,index_int gsize)
    : distribution(d),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    create_blockcyclic(bs,nb,gsize); };
};

/*!
  A replicated distribution has the same block of indices on each processor.
  \todo n>1 elements & k=1 is the same as n=1, k>1. disable get_ortho?
 */
class replicated_distribution : virtual public distribution {
public:
  replicated_distribution(decomposition *d,int ortho,index_int lsize)
    : distribution(d),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    create_from_replicated_local_size(lsize);
    set_orthogonal_dimension(ortho);
    set_name("replicated-scalar");
  };
  replicated_distribution(decomposition *d,index_int lsize)
    : replicated_distribution(d,1,lsize) {};
};

/*!
  A gathered distribution is the result of gathering identically sized blocks
  from all processors. It is equivalent to a replicated distribution
  with P times the per-processor local size.
 */
class gathered_distribution : virtual public distribution {
public:
  gathered_distribution(decomposition *d,int k,index_int lsize)
    : distribution(d),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    create_from_replicated_local_size(lsize*domains_volume());
    set_name("gathered-scalar"); set_orthogonal_dimension(k);
  };
};

class binned_distribution : virtual public distribution {
public:
  binned_distribution(decomposition *d,object *o)
    : distribution(d),//decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    create_by_binning(o);
    set_name("binned");
  };
};

/*!
  A \ref sigma_operator that has access to a whole distribution, rather
  than just its input multi_indexstruct.
 */
class distribution_sigma_operator {
protected:
  sigma_operator sigop;
  bool sigma_based{false};
  multi_sigma_operator *msigop{nullptr};
  bool multi_based{false};
  std::function< std::shared_ptr<multi_indexstruct>(distribution*,processor_coordinate&) >
      dist_sigma_f;
  bool lambda_based{false};
  std::function< distribution*(distribution*) > dist_global_f{nullptr};
  bool global_based{false};
public:
  distribution_sigma_operator(const sigma_operator &s) { sigop = s; sigma_based = true; };
  distribution_sigma_operator(multi_sigma_operator *s) { msigop = s; multi_based = true; };
  distribution_sigma_operator
  ( std::function< std::shared_ptr<multi_indexstruct>(distribution*,processor_coordinate&) > f )
  { dist_sigma_f = f; lambda_based = true; };
  distribution_sigma_operator( std::function< distribution*(distribution*) > f )
  { dist_global_f = f; global_based = true; };

  std::shared_ptr<indexstruct> operate(int dim,distribution *d,processor_coordinate &p) const;
  std::shared_ptr<indexstruct> operate(int dim,distribution *d,processor_coordinate p) const;
  std::shared_ptr<multi_indexstruct> operate(distribution *d,processor_coordinate &p) const;
  distribution *operate(distribution *d) const;
  bool is_global_based() const { return global_based; };
};

class distribution_abut_operator : public distribution_sigma_operator {
public:
  distribution_abut_operator( processor_coordinate me )
    : distribution_sigma_operator
      ( [me] (distribution *d) -> distribution* {
	if (d->get_same_dimensionality(me.get_dimensionality())!=1)
	  throw(fmt::format("Can only abut in 1d"));
	// get the local block
	index_int localsize;
	try {
	  processor_coordinate whyme = me; // something with qualifiers
	  auto block = d->get_processor_structure(whyme);
	  if (!block->is_contiguous())
	    throw(fmt::format("Can only abut contiguous"));
	  localsize = block->volume();
	} catch (...) {
	  throw(fmt::format("Error getting block for local {}",me.as_string()));
	}
	// gather all block sizes
	try {
	  int nprocs = d->domains_volume();
	  std::vector<index_int> sizes(nprocs);
	  d->gather64(localsize,sizes);
	  // assemble all blocks
	  index_int shift = 0;
	  auto rstruct = new parallel_structure( dynamic_cast<decomposition*>(d) );
	  for (int p=0; p<nprocs; p++) {
	    localsize = sizes.at(p);
	    auto new_block = std::shared_ptr<multi_indexstruct>
	      ( new multi_indexstruct
		( std::shared_ptr<indexstruct>
		  ( new contiguous_indexstruct(shift,shift+localsize-1) ) ) );
	    auto proc_coord = d->coordinate_from_linear(p);
	    rstruct->set_processor_structure(proc_coord,new_block);
	    // fmt::print("{}: abut block @{}, s={}: {}\n",
	    // 	       proc_coord->as_string(),shift,localsize,new_block->as_string());
	    shift += localsize;
	  }
	  rstruct->set_is_known_globally();
	  auto abuted = d->new_distribution_from_structure(rstruct);
	  //fmt::print("abuted distro: {}\n",rstruct->as_string());
	  return abuted;
	} catch (std::string c) { throw(fmt::format("Trouble constructing abuted distro {}",c));
	} catch (...) { throw(fmt::format("Trouble constructing abuted distro"));
	}
      } ) {}
};

class distribution_stretch_operator : public distribution_sigma_operator {
public:
  distribution_stretch_operator(domain_coordinate big) // also a ref variant?
    : distribution_sigma_operator
      ( [big] (distribution *d) -> distribution* {
	if (d->get_dimensionality()!=1)
	  throw(fmt::format("Can only stretch in 1d, not {}",d->as_string()));
	index_int src_size = 0, tar_size = big[0];
	auto decomp = dynamic_cast<decomposition*>(d);
	if (decomp==nullptr)
	  throw(fmt::format("Could not upcast to decomp: {}",d->as_string()));
	if (!d->is_known_globally())
	  throw(fmt::format("Can only stretch globally known, not {}",d->as_string()));
	// what are the block sizes?
	int nprocs = d->domains_volume(); std::vector<index_int> sizes(nprocs);
	for (int p=0; p<nprocs; p++) {
	  auto proc_coord = d->coordinate_from_linear(p);
	  sizes.at(p) = d->get_processor_structure(proc_coord)->volume();
	  src_size += sizes.at(p);
	}
	// stretch sizes, prevent zeros.
	index_int mid_size=0; float ratio = (1.*tar_size)/src_size; int pmax=0,imax=0;
	for (int p=0; p<nprocs; p++) {
	  sizes.at(p) = static_cast<index_int>( sizes.at(p) * ratio );
	  if (sizes[p]==0) sizes[p] = 1;
	  if (sizes[p]>sizes[pmax]) { imax = sizes[p]; pmax = p; };
	  mid_size += sizes.at(p);
	  //if (sizes.at(p)==0) zeros++;
	}
	// make sure that we sum up to exactly the right size
	index_int excess = tar_size-mid_size;
	if (excess<=0) {
	  if (sizes[pmax]+excess<=0)
	    throw(fmt::format("Weird case: shortfall of {} more than max proc",-excess));
	  sizes[pmax] += excess; excess = 0;
	  // std::stringstream result;
	  // std::copy(sizes.begin(), sizes.end(), std::ostream_iterator<int>(result, " "));
	  // throw(fmt::format("Could not naive stretch to {}:\ninput {}\nstretched sizes {}",
	  // 		    tar_size,d->as_string(),result.str()));
	} else {
	  // add in the remaining difference
	  for (int p=0; p<nprocs; p++)
	    sizes.at(p) += ((p+1)*excess)/nprocs - (p*excess)/nprocs;
	}
	// now make the new structures
	auto rstruct = new parallel_structure(decomp); index_int shift=0;
	for (int p=0; p<nprocs; p++) {
	  index_int localsize = sizes.at(p);
	  auto new_block = std::shared_ptr<multi_indexstruct>
	    ( new multi_indexstruct
	      ( std::shared_ptr<indexstruct>
		( new contiguous_indexstruct(shift,shift+localsize-1) ) ) );
	  auto proc_coord = d->coordinate_from_linear(p);
	  rstruct->set_processor_structure(proc_coord,new_block);
	  shift += localsize;
	}
	rstruct->set_is_known_globally();
	auto stretched = d->new_distribution_from_structure(rstruct);
	return stretched;
      } ) { };
	  // if (excess>=zeros && zeros>0) {
	  //   for (int p=0; p<nprocs; p++) {
	  //     if (sizes.at(p)==0) { sizes.at(p)++; excess--; }
	  //     if (excess==0) break;
	  //   }
	  // }
};

/****
 **** Utility: sparse stuff
 ****/

/*! A sparse element is a floating point value plus an index.
  We are using these to make \ref sparse_row and then \ref sparse_matrix object.
  Since we like to keep elements ordered by index, we define an ordering.
*/
class sparse_element {
protected:
  std::tuple<index_int,double> e;
public:
  sparse_element() {}; // default constructor
  sparse_element( index_int i,double v ) {
    std::get<0>(e) = i; std::get<1>(e) = v; };
  index_int get_index() { return std::get<0>(e); };
  double get_value() { return std::get<1>(e); };
  int operator<( sparse_element other ) { return get_index()<other.get_index(); };
  int operator>( sparse_element other ) { return get_index()>other.get_index(); };
  std::string as_string();
};

/*!
  A sparse row is a vector of \ref sparse_element objects.
  Adding a sparse element preserves the ordering by element index.
 */
class sparse_row {
protected:
  std::vector<sparse_element> r;
public:
  sparse_row() { r.reserve(10); };
  void reserve(int n) { r.reserve(n); };
  void add_element( sparse_element e ) {
    auto pos = r.begin();
    for ( ; pos!=r.end(); ++pos) {
      if (e<*pos) break;
    }
    r.insert( pos, e );
  };
  void add_element(index_int i,double v) { add_element( sparse_element(i,v) ); };
  bool has_element(index_int i) { 
    auto pos = r.begin();
    for ( ; pos!=r.end(); ++pos) { index_int testi = (*pos).get_index();
      if (i==testi) return true; if (i<testi) return false; }
    return false; };
  std::shared_ptr<indexstruct> all_indices();
  int size() { return r.size(); };
  sparse_element operator[](int i) { return r[i]; };
  sparse_element at(int i) {
    if (i>=r.size()) throw(fmt::format("Index {} out of bound {}",i,r.size()));
    return r[i]; };
  double inprod( std::shared_ptr<object> o,processor_coordinate &p );
  double row_sum() { double s=0.; for (int i=0; i<r.size(); i++) s += r[i].get_value();
    return s; };
  std::string as_string();
};

class sparse_rowi {
protected:
  std::tuple<index_int,std::shared_ptr<sparse_row>> r;
public:
  sparse_rowi(index_int i) {
    std::get<0>(r) = i; std::get<1>(r) = std::shared_ptr<sparse_row>( new sparse_row() );
  };
  index_int get_row_number() const { return std::get<0>(r); }; //!< Get the index component
  std::shared_ptr<sparse_row> get_row() const { //!< Get the actual sparse row
    return std::get<1>(r); };
  int size() { return std::get<1>(r)->size(); };
  void add_element(index_int j,double v) { std::get<1>(r)->add_element(j,v); };
  int has_element(index_int j) { return std::get<1>(r)->has_element(j); };
  std::shared_ptr<indexstruct> all_indices() { return std::get<1>(r)->all_indices(); };
  double row_sum() { return std::get<1>(r)->row_sum(); };
  double inprod( std::shared_ptr<object> o,processor_coordinate &p ) {
    return std::get<1>(r)->inprod(o,p); };
  sparse_element at(int i) { return std::get<1>(r)->at(i); };
  std::string as_string();
};

class sparse_matrix : public entity {
protected:
  std::vector< std::shared_ptr<sparse_rowi> > m;
  index_int globalsize{-1}; //!< Test for column overflow
public:
  sparse_matrix()
    : entity(entity_cookie::OPERATOR) { m.reserve(100); set_name("sparse-mat"); };
  //! Create from list of rows; after this the rows are locked.
  sparse_matrix( indexstruct* );
  //! \todo make the shared ptr the default
  sparse_matrix( std::shared_ptr<indexstruct> idx ) : sparse_matrix(idx.get()) {}
  //! Creation in a parallel context //! \todo use shared ptr
  sparse_matrix( parallel_indexstruct *struc,int mytid )
    : sparse_matrix( struc->get_processor_structure(mytid).get()->make_clone() ) {};

protected:
  int row_creation_locked{0};
public:
  void set_row_creation_locked() { row_creation_locked = 1; };
  int get_row_creation_locked() { return row_creation_locked; };
  virtual void add_element( index_int i,index_int j,double v );
  virtual void add_element( index_int i,index_int j ) {
    add_element(i,j,1.); }; //!< Adjacency matrix
  int get_row_index_by_number(index_int i);
  void insert_row( std::shared_ptr<sparse_rowi> row );
  index_int first_row_index() { return m.at(0)->get_row_number(); };
  index_int last_row_index() { return m.at(m.size()-1)->get_row_number(); };
  std::shared_ptr<indexstruct> row_indices();
  bool has_element( index_int i,index_int j );
  virtual std::shared_ptr<indexstruct> all_columns(); // there seems to be an MPI version
  std::shared_ptr<indexstruct> all_local_columns() { return sparse_matrix::all_columns(); }; //!< \todo needless synonym?
  std::shared_ptr<indexstruct> all_columns_from( std::shared_ptr<multi_indexstruct> i );
  index_int local_size() { return m.size(); };
  index_int nnzeros() {
    index_int r=0;
    for (auto row=m.begin(); row!=m.end(); ++row) {
      r += (*row)->size();
    }
    return r;
  };
  virtual double row_sum( index_int i ) { auto row = get_row_index_by_number(i);
    if (row<0) throw(fmt::format("Row sum: row {} not in matrix",i));
    return m.at(row)->row_sum(); };
  virtual void multiply( std::shared_ptr<object> in,std::shared_ptr<object> out,processor_coordinate &p);

  virtual sparse_matrix *transpose() const;

public: // should be protected
  static int sparse_matrix_trace;
public:
  void set_trace() { sparse_matrix_trace = 1; };
  int get_trace() { return sparse_matrix_trace; };
  std::string as_string();
  std::string contents_as_string();
};

/****
 **** Data
 ****/

/*!
  We need to keep track of an object's data status.
  This value is set in the \ref object::allocate routine, which is virtual,
  so make sure you check the OpenMP backend when you make changes.

  - UNALLOCATED : there is no data
  - ALLOCATED : data is allocated internally
  - INHERITED : data belongs to another object that may also be in use
  - REUSED : data belongs to another object that may not be in use anymore
  - USER : data is suppplied by the user

  We use the ordering: >UNALLOCATED means there is data.
 */
enum class object_data_status{ UNALLOCATED,ALLOCATED,INHERITED,REUSED,USER };

/*!
  We like to manage data independent of the distribution type
  \todo use shared pointer for data
*/
class object_data {
protected:
public:
  //! Create data object; create space for data pointers.
  object_data(int ndom) { create_data_pointers(ndom); };

protected:
  object_data_status data_status{object_data_status::UNALLOCATED};
public:
  object_data_status get_data_status() { return data_status; };
  //! Check that the object is still unallocated
  int has_data_status_unallocated() { return data_status==object_data_status::UNALLOCATED; };
  //! Check that the object has data, no matter where it comes from
  int has_data_status_allocated() { return data_status>=object_data_status::ALLOCATED; };
  //! Check that the object has private data
  int has_data_status_private() { return data_status==object_data_status::ALLOCATED; };
  //! Check that the object shared its data with another object
  int has_data_status_inherited() { return data_status==object_data_status::INHERITED; };
  //! Check that the object shared its data with another object
  int has_data_status_reused() { return data_status==object_data_status::REUSED; };
  std::string data_status_as_string();

private:
  //! Pointers to the data for the local domains.
  std::shared_ptr<double> numa_data_pointer;
  index_int numa_data_size;
public:
  double *get_raw_data() const; index_int get_raw_size() const;
  std::shared_ptr<double> get_numa_data_pointer() const;

private:
  std::vector<double*> domain_data_pointers;
  std::vector<index_int> data_sizes;
  std::vector<index_int> data_offsets;
  static double create_data_count;
  static bool trace_create_data; //! \todo can we initialize static variables in the class?
  void delete_data() {}; // if (data_pointer!=nullptr) delete data_pointer; };
public:
  void create_data_pointers(int ndom);
  void set_domain_data_pointer( int,double*,index_int,index_int=0 );
  double *get_data_pointer( int n ); index_int get_data_size( int n );
  void create_data(index_int s); void create_data(index_int s, std::string c);
  static void set_trace_create_data() { trace_create_data = true; };
  static void set_trace_create_data( bool trace ) { trace_create_data = trace; };
  bool get_trace_create_data() { return trace_create_data; };
  void inherit_data(std::shared_ptr<object>,processor_coordinate&);
  void inherit_data(std::shared_ptr<object>,index_int,processor_coordinate&);
  void set_value( std::shared_ptr<double> x ); void set_value(double x);
  double get_max_value(); double get_min_value();

  //! Register the data amount; this gets set in the object to the entity routine
  std::function< void(index_int) > register_data{
    [] (index_int s) -> void { throw(std::string("basic register data")); } };
  //! \todo there is no reason for two levels of get_data
  double *get_data(processor_coordinate &p) const {
    return get_data_p(p); };
  double *get_data(processor_coordinate &&p) const {
    return get_data_p(p); };
public:
  //! Return a pointer to the processor specific data. Global on this address space.
  std::function< double*(processor_coordinate &p) > get_data_p{
    [] (processor_coordinate &p) -> double* {
      throw(std::string("no basic get_data_p")); } };
  std::function< double*(processor_coordinate &&p) > get_data_pp{
    [] (processor_coordinate &&p) -> double* {
      throw(std::string("no basic get_data_pp")); } };

public:
  //! Allocation. This is done in the specific object::allocate routine.
  //! \todo should not be here: make object method
  std::function< void(void) > allocate{
    [] (void) -> void { throw(std::string("basic allocate")); } };

protected:
  int data_parent{-1}; //!< Object number from which data is inherited; see \ref algorithm::inherit_data_from_betas.
public:
  int get_data_parent() { return data_parent; };
  void set_data_parent(int p) { data_parent = p; };
};


/****
 **** Object
 ****/

/*!
  An object is based on a \ref distribution object and contains the actual data.
  The allocation of data is done in the derived classes \ref mpi_object 
  and \ref omp_object.
*/
class object : public distribution,public object_data,virtual public entity {
private:
  static int count;
public:
  /*!
    Create an object with locally allocated data. The actual allocation is done in the
    derived classes, since it depends on details of the architecture.
   */
  object( distribution *d )
    : distribution(d),object_data(d->local_ndomains()),
      entity(dynamic_cast<entity*>(d),entity_cookie::OBJECT) {
    object_number = count++; data_is_filled = new processor_mask(d);
    this->set_name(fmt::format("object-{}",object_number));
  };

protected:
  int object_number{-1}; //!< Unique number for each object \todo do this in entity?
public:
  int get_object_number() { return object_number; };
  // this is just to make static casts possible
  virtual void poly_object() = 0;
  //! \todo write this factory

  distribution *get_distribution() {
    auto d = dynamic_cast<distribution*>(this);
    if (d==nullptr)
      throw(fmt::format("Could not cast object->distribution: <<{}>>",as_string()));
    return d;
  };
  decomposition *get_decomposition() {
    auto d = dynamic_cast<decomposition*>(this);
    if (d==nullptr)
      throw(fmt::format("Could not cast object->decomposition: <<{}>>",as_string()));
    return d;
  };
  std::shared_ptr<object> object_with_same_distribution() {
    //    new product_object( dynamic_cast<distribution*>(this.get()) );
  };

protected:
  bool has_successor{false};
public:
  // Mark objects that are used by subsequent tasks. \todo should be task method?
  void set_has_successor() { has_successor = 1; };
  bool get_has_successor() { return has_successor ; };

  /*
   * Data and status
   */
public:
  //! Store a data pointer in the right location, multi-d domain number
  void register_data_on_domain
      ( processor_coordinate &dom,double *dat,index_int s ) {
    register_data_on_domain_number(get_domain_local_number(dom),dat,s); }
  //! Store a data pointer by local index; see \ref register_data_on_domain for global
  void register_data_on_domain_number( int loc,double *dat,index_int s ) {
    set_domain_data_pointer(loc,dat,s); register_allocated_space(s);
  };
  double get_element_by_index(index_int i,processor_coordinate &p); // sparse matrix support
  std::shared_ptr<object> reuse() {
    return new_object_from_data(get_numa_data_pointer().get()); };
  double get_ith_element(index_int i) { throw(std::string("Unimplemented get ith from object")); };

protected:
  processor_mask *data_is_filled{nullptr};  
public:
  //! Set in \ref mpi_task::acceptReadyToSend 
  void set_data_is_filled( processor_coordinate &p ) { data_is_filled->add(p); };
  //! If this is not set, we do not copy betas \todo should not ignore domain value
  int get_data_is_filled( processor_coordinate &p ) { return data_is_filled->lives_on(p); };

  virtual void copy_data_from( std::shared_ptr<object> in,message *smsg,message *rmsg ) {
    throw(std::string("copy_data_from: unimpl")); };

  /*
   * Mask stuff
   */
  //! Indexing shift; see \ref omp_distribution::mask_shift for a good example.
  virtual domain_coordinate mask_shift( processor_coordinate &p ) = 0;
  //  virtual index_int mask_shift(int p) = 0;

  virtual std::string as_string();
  std::string values_as_string(processor_coordinate &p);
  std::string values_as_string(processor_coordinate &&p);
};

/****
 **** Message
 ****/

class message_tag {
private:
  int contents[4]; // sender,receiver,kernel_number,domains,
public:
  //! Construct tag from sender, receiver, kernel number, ndomains
  message_tag(int s,int r,int k,int d) {
    contents[0] = s; contents[1] = r; contents[2] = k; contents[3] = d;
    int t = s*d + r*d*d; if (k>=0) t+=k; };
  message_tag( int *c ) : message_tag( c[0],c[1],c[2],c[3] ) {};
  int get_tag_value() { return get_tag_kernel()+message_tag_admin_threshold; };
  int get_tag_sender() { return contents[0]; };
  int get_tag_kernel() { return contents[2]; };
  int get_length() { return 4; };
  int *get_contents() { return contents; }; // not to be abused
};

/*!
  What is the status of this message?
  - VIRGIN : message was created
  - SKIPPED : message was not sent when sending was done
  - SENT : actively sent
*/
enum class message_status { VIRGIN,SKIPPED,POSTED,COMPLETED };

/*!
  Is this a send or receive message?
  - NONE : not assigned
  - SEND : send message
  - RECEIVE : receive message
*/
enum class message_type { NONE,SEND,RECEIVE };

/*!
  A message is the correspondence of an \ref indexstruct index set and two processors:
  the #sender and #receiver. The index set is stored in global terms as #global_struct,
  on the sender and receiver is stored in local terms as #local_struct, 
  derived by #relativize_to().

  Messsages are independent of the protocol used. Their send/recv/wait actions are
  taken in the specific definition of task::execute().
  \todo we should really let this inherit from decomposition
*/
class message : public decomposition,virtual public entity {
public:
  message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	  std::shared_ptr<multi_indexstruct> &g) : message(d,snd,rcv,g,g) {};
  message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	  std::shared_ptr<multi_indexstruct> &e,std::shared_ptr<multi_indexstruct> &g);
  //  ~message() { delete_subarray(); };

  /*
   * Sender/receiver
   */
protected:
  processor_coordinate sender,receiver;
public:
  processor_coordinate &get_sender(); processor_coordinate &get_receiver();

protected:
  message_type sendrecv_type{message_type::NONE};
public:
  void set_send_type() ; void set_receive_type() ;
  message_type get_sendrecv_type() const;  std::string sendrecv_type_as_string() const;


  /*
   * Local/global struct
   */
protected:
  std::shared_ptr<multi_indexstruct> global_struct{nullptr},local_struct{nullptr},
    embed_struct{nullptr},halo_struct{nullptr};
public: // in the end we need access functions for this
  int *numa_sizes{nullptr},*struct_sizes{nullptr},*struct_starts{nullptr};
public:
  std::shared_ptr<multi_indexstruct> get_global_struct();
  std::shared_ptr<multi_indexstruct> get_local_struct();
  std::shared_ptr<multi_indexstruct> get_embed_struct();
  //! \todo is this actually needed?
  void set_halo_struct(std::shared_ptr<multi_indexstruct> h) { halo_struct = h; };
  std::shared_ptr<multi_indexstruct> get_halo_struct() {
    if (halo_struct==nullptr) throw(std::string("No halo struct to get from message"));
    return halo_struct;
  };
  void compute_subarray
      (std::shared_ptr<multi_indexstruct>,std::shared_ptr<multi_indexstruct>,int);
  void delete_subarray() { if (numa_sizes) delete numa_sizes;
    if (struct_sizes) delete struct_sizes; if (struct_starts) delete struct_starts; };
    
  /*
   * Collective
   */
protected:
  bool is_collective{false};
public:
  void set_is_collective( bool is=true ) { is_collective = is; };
  bool get_is_collective() { return is_collective; };

  /*
   * In/out object
   */
protected:
  std::shared_ptr<object> in_object{nullptr}, out_object{nullptr};
  index_int src_index{-1},tar_index{-1};
public:
  virtual void compute_src_index(); virtual void compute_tar_index();
  void set_in_object( std::shared_ptr<object> in );
  void set_out_object( std::shared_ptr<object> out );
  std::shared_ptr<object> get_in_object( ); std::shared_ptr<object> get_out_object( );
  index_int get_src_index(); index_int get_tar_index();
  
protected:
   //!< \todo these are only used int he snd message to find the objects. Ugly.
  int in_object_number{-1},out_object_number{-1}, dependency_number{-1};
public:
  void set_in_object_number(int in) { in_object_number = in; };
  int get_in_object_number() {
    if (in_object==nullptr) return in_object_number;
    else return in_object->get_object_number(); };
  int get_out_object_number() {
    if (out_object==nullptr) return out_object_number;
    else return out_object->get_object_number(); };
  void set_out_object_number(int out) { out_object_number = out; };
  void set_dependency_number(int n) { dependency_number = n; };
  int get_dependency_number() {
    if (dependency_number<0)
      throw(std::string("No dependency number was set for this message"));
    return dependency_number;
  };

  /*
   * tag and other identifiers
   */
protected:
  message_tag *tag{nullptr};
  void set_tag_by_content(processor_coordinate&,processor_coordinate&,int step,int np);
public:
  int how_many_times{0};
  void set_tag( message_tag *t ) { tag = t; };
  void set_tag_from_kernel_step( int step,int nprocs ) {
    set_tag_by_content( sender,receiver,step,nprocs); };
  message_tag *get_tag() { return tag; };
  int get_tag_value() { return tag->get_tag_value(); };
  int get_tag_kernel() { return tag->get_tag_kernel(); };
protected:
  message_status status{message_status::VIRGIN};
public:
  message_status get_status() const { return status; };
  void set_status( message_status s ) { status = s; };
  //  void set_was_skipped() { status = message_status::SKIPPED; }
  void clear_was_sent() { status = message_status::VIRGIN; }

  void relativize_to(std::shared_ptr<multi_indexstruct> beta);
  index_int volume() { return global_struct->volume(); };
  void as_char_buffer(char *buf,int *len);
  virtual std::string as_string();
public:
  fmt::MemoryWriter annotation;
};

/****
 **** Signature function
 ****/

/*!
  An signature function determines the \f$ \beta \f$ distribution.
  Since \f$ \beta=\sigma_f\gamma \f$,
  a big part of the work is in describing the \f$ \sigma_f\f$ function. This can be
  - operator based (\ref signature_function::signature_type::OPERATORS); this is mostly useful for stencils. Each `leg' of the stencil is declared with \ref add_sigma_operator.
  - pattern based (\ref signature_function::signature_type::PATTERN); this covers the sparse matrix case by passing an \ref index_pattern in \ref set_index_pattern.
  - function based (\ref signature_function::signature_type::FUNCTION); this is the most general case, declared with \ref set_signature_function.


  These three descriptions are explicit: they need to be translated to
  an actual \ref distribution object.  This is done in the routine
  \ref derive_beta_structure, which calls \ref explicit beta
  distribution.

  Beta objects are not handled directly: they are included in the \ref kernel object,
  and their methods are called from kernel methods.

  \todo make a class signature_operator which contains the vector of operators.
 */
class signature_function : virtual public entity {
public:
  //! The constructor does not set or allocate anything.
  signature_function()
    : entity(entity_cookie::SIGNATURE) {};
  void copy_from(signature_function *f) { type = f->type; func = f->func;
    operators = f->operators; pattern = f->pattern; 
    explicit_beta_distribution = f->explicit_beta_distribution;
    fmt::MemoryWriter w; w.write("embedded-{}",f->get_name()); set_name(w.str());
  };

  // type
protected:
  enum class signature_type { UNINITIALIZED,
      OPERATORS, PATTERN, EXPLICIT, LOCAL, FUNCTION };
  signature_type type{signature_type::UNINITIALIZED};
public:
  int has_type_uninitialized() { return type==signature_type::UNINITIALIZED; };
  int has_type_initialized( signature_type t ) {
    if (has_type_uninitialized()) throw(std::string("Beta needs type"));
    return get_type()==t; };
  //! Set type if uninitialized or already of this type; otherwise complain
  void set_type( signature_type t ) {
    if (has_type_uninitialized()) type = t;
    else if (has_type_initialized(t)) return;
    else { throw(fmt::format("Can not change signature type from {} to {}\n",
			     type_to_string(type),type_to_string(t))); }
  };
  signature_type get_type() { return type; };
  void set_type_explicit_beta()  { set_type(signature_type::EXPLICIT); };
  void set_type_operator_based() { set_type(signature_type::OPERATORS); };
  void set_type_local()          { add_sigma_operator( ioperator("none") ); };
  void set_type_function()       { set_type(signature_type::FUNCTION); };
  int has_type_explicit_beta()   { return has_type_initialized(signature_type::EXPLICIT); };
  int has_type_operator_based()  { return has_type_initialized(signature_type::OPERATORS); };
  int has_type_local()           { throw(std::string("no local anymore")); };
  int has_type_function()        { return has_type_initialized(signature_type::FUNCTION); };
  std::string type_to_string(signature_type t) const {
    if (t==signature_type::UNINITIALIZED) return std::string("uninitialized");
    if (t==signature_type::OPERATORS) return std::string("operators");
    if (t==signature_type::PATTERN) return std::string("pattern");
    if (t==signature_type::EXPLICIT) return std::string("explicit");
    if (t==signature_type::FUNCTION) return std::string("function");
    throw(std::string("unknown type")); }
  std::string type_as_string() { return type_to_string(get_type()); };

  // Signature from operators
protected:
  std::vector<multi_ioperator*> operators;
public:
  void add_sigma_operator( multi_ioperator *op );
  void add_sigma_operator( ioperator &op ) { add_sigma_operator( new multi_ioperator(op) ); };
  void add_sigma_operator( ioperator &&op ) { add_sigma_operator( new multi_ioperator(op) ); };
  std::vector<multi_ioperator*> &get_operators() { return operators; };
  void add_sigma_stencil( stencil_operator *stencil ) {
    auto ops = stencil->get_operators();
    for (int iop=0; iop<ops.size(); iop++)
      add_sigma_operator(ops.at(iop));
  };
  void add_sigma_stencil( stencil_operator &stencil ) {
    auto ops = stencil.get_operators();
    for (int iop=0; iop<ops.size(); iop++)
      add_sigma_operator(ops.at(iop));
  };

  // Signature from sparsity pattern
protected:
  sparse_matrix *pattern{nullptr}; //index_pattern *pattern{nullptr};
public:
  void set_type_pattern() { set_type(signature_type::PATTERN); };
  int has_type_pattern() { return has_type_initialized(signature_type::PATTERN); };
  void set_index_pattern( sparse_matrix *patt ) { set_type_pattern(); pattern = patt; };

  // Signature from function: derive point-by-point
protected:
  multi_sigma_operator *func{nullptr};
public:
  void set_signature_function_function( multi_sigma_operator *op );
  void set_signature_function_function( const sigma_operator &op );
  void set_signature_function_function
      ( std::function< std::shared_ptr<indexstruct>(index_int) > f );

  // Signature from explicit beta
protected:
  distribution *explicit_beta_distribution{nullptr};
public:
  virtual void set_explicit_beta_distribution( distribution *d );
  distribution *get_explicit_beta_distribution();

  //! Derive without truncation
  parallel_structure *derive_beta_structure(distribution *d) {
    derive_beta_structure(d,nullptr); };
  parallel_structure *derive_beta_structure(distribution*,std::shared_ptr<multi_indexstruct>);
  parallel_structure *derive_beta_structure_operator_based(distribution*,std::shared_ptr<multi_indexstruct>);
  parallel_structure *derive_beta_structure_pattern_based(distribution*);
  std::shared_ptr<multi_indexstruct> make_beta_struct_from_ops 
  ( processor_coordinate&,std::shared_ptr<multi_indexstruct>,
    std::vector<multi_ioperator*>&,std::shared_ptr<multi_indexstruct>);

protected:
  bool tracing{false};
public:
  void set_tracing() { tracing = true; };
};

/****
 **** Requests
 ****/

enum class request_protocol { UNKNOWN, MPI, OPENMP };
enum class request_type { UNKNOWN, INCOMING, OUTGOING };

/*!
  Inspired by MPI requests, a request object is something that can be
  tested to see if a communication, whether sending or receiving,
  has concluded. 

  The mode-dependent derived versions have their own data, such as the MPI_Request.
*/
class request {
public:
  message *msg;
  request_protocol protocol{request_protocol::UNKNOWN};
  request_type type{request_type::UNKNOWN};
public:
  request( message *m, request_protocol p ) {
    protocol = p; msg = m;
  };
  virtual void poly() = 0; //! Just to make it dynamic castable
  message *get_message() { return msg; };
  void set_completed() { msg->set_status( message_status::COMPLETED ); };
  bool has_type( request_type t ) const { return type==t; };
  bool is_completed() const { return msg->get_status() == message_status::COMPLETED; };
};

class request_vector {
protected:
  std::vector<request*> requests;
public:
  request_vector() { requests.reserve(10); };
  virtual void add_request(request *r) { requests.push_back(r); };
  virtual void wait() = 0;
  int size() const { return requests.size(); };
  bool completed() {
    for ( auto r : requests )
      if (!r->is_completed()) return false;
    return true;
  };
  void set_completed() {
    for ( auto r : requests )
      r->set_completed();
  };
};

/*!
  A task is identified by its step, which is the same as the object id
  of its outvector, and the domain which is the part of the distribution
  that it covers.
*/
class task_id {
protected:
  int step{-1};
  processor_coordinate domain;
public:
  //! \todo pass coordinate by reference
  task_id(int s,processor_coordinate &d) { step = s; domain = d; };
  int get_step() { return step; };
  processor_coordinate &get_domain() { return domain; };
  std::string as_string() { return fmt::format("<{},{}>",step,domain.as_string()); };
};

/****
 **** Kernel
 ****/

/*!
  A dependency is a signature function, applied to a distribution.
  Thus we inherit from \ref signature_function.
*/
class dependency : public signature_function,virtual public entity {
private:
  std::shared_ptr<object> in_object{nullptr};
public:
  dependency(std::shared_ptr<object> in);
  std::shared_ptr<object> get_in_object();

  /*
   * Beta
   */
protected:
  distribution *beta_distribution{nullptr};
public:
  distribution *get_beta_distribution(); int has_beta_distribution();
  void set_beta_distribution( distribution *b);
  void derive_beta_distribution();
  void ensure_beta_distribution(std::shared_ptr<object> out);

  /*
   * Beta
   */
protected:
  std::shared_ptr<object> beta_object{nullptr};
public:
  void set_beta_object( std::shared_ptr<object> ); std::shared_ptr<object> get_beta_object();
  int has_beta_object() { return beta_object!=nullptr; };
  void create_beta_vector(std::shared_ptr<object>); // ensure beta dist and create beta obj

  /*
   * Collective
   */
protected:
  bool is_collective{false};
public:
  void set_is_collective() { set_is_collective(true); };
  void set_is_collective( bool is ) { is_collective = is; };
  bool get_is_collective() { return is_collective; };

};

/*! 
  A class containing a bunch of dependency objects. A kernel has one of these 
  things, to make reasoning simpler. I think.

 \todo should this inherit from entity, as opposed to kernel doing so?
 */
class dependencies {
protected:
  std::vector<dependency*> the_dependencies;
public:
  //! Create dependencies for origin kernel
  dependencies() {};
  //! Create first dependency for compute kernel
  dependencies( std::shared_ptr<object> in ) : dependencies() { add_in_object(in); };
  std::vector<dependency*> &get_dependencies() { return the_dependencies; };
  dependency *get_dependency(int d) {
    if (d<0 || d>=the_dependencies.size())
      throw(std::string("Can not get that in dependency"));
    return the_dependencies.at(d); };
  //! Push a new object on to the dependencies list. This is overridden in the kernel class
  virtual void add_in_object( std::shared_ptr<object> in ) {
    dependency *d = new dependency(in);
    the_dependencies.push_back(d); };
  std::vector<std::shared_ptr<object>> get_in_objects() {
    throw(std::string("who wants the in objects?")); };
  int get_n_in_objects() { return the_dependencies.size(); };
  std::shared_ptr<object> get_in_object(int b) { return get_dependency(b)->get_in_object(); };
  distribution *get_beta_distribution(int n) {
    return get_dependency(n)->get_beta_distribution(); };
  void set_beta_distribution( int n,distribution *d ) {
    if (n<0 || n>=the_dependencies.size())
      throw(std::string("Setting invalid dependency beta"));
    the_dependencies.at(n)->set_beta_distribution(d); };
  void ensure_beta_distribution( std::shared_ptr<object> out ) {
    for (auto d : the_dependencies )
      d->ensure_beta_distribution(out);
  };
  void allocate_beta_vector(int d,std::shared_ptr<object> out) {
    get_dependency(d)->create_beta_vector(out); };
  std::shared_ptr<object> get_beta_object( int d ) {
    return get_dependency(d)->get_beta_object(); };  
  std::vector<std::shared_ptr<object>> get_beta_objects();
  //! Test that all betas are defined on this domain
  int all_betas_live_on(processor_coordinate &d) { int live = 1;
    for (auto dep : the_dependencies) //.begin(); dep!=the_dependencies.end(); ++dep)
      live = live && dep->get_beta_object()->lives_on(d);
    return live; };
  //! Test that all betas are filled on this domain
  int all_betas_filled_on(processor_coordinate &d) { int live = 1;
    for (auto dep : the_dependencies)
      live = live && dep->get_beta_object()->get_data_is_filled(d);
    return live; };
  // Shortcuts for the last dependency
  dependency *last_dependency() {
    if (the_dependencies.size()==0) throw(std::string("No last dependency to get"));
    return the_dependencies.at( the_dependencies.size()-1 ); };
  void add_sigma_operator(ioperator op) {
    last_dependency()->add_sigma_operator(op); };
  void add_sigma_operator(multi_shift_operator *op) {
    last_dependency()->add_sigma_operator(op); };
  void add_sigma_stencil( stencil_operator *stencil ) {
    last_dependency()->add_sigma_stencil(stencil); };
  void set_explicit_beta_distribution( distribution *d ) {
    last_dependency()->set_explicit_beta_distribution(d); };
  void create_beta_vector( std::shared_ptr<object> o ) {
    last_dependency()->create_beta_vector(o); };
};

class task;
enum class kernel_type { UNDEFINED,ORIGIN,COMPUTE,TRACE };

/*!
  A kernel is the application of a function between two \ref object objects.
  Since we don't have a compiler to derive the \f$ I_f\f$ data dependency function
  from the function definition, we have a number of calls to derive the 
  #beta_definition object.

  A kernel is somewhat abstract:
  both analysis and execution are delegated to the \ref task objects into which
  it is split.

  \todo kernel should inherit from object.....
*/
class kernel : public dependencies,public tracer,virtual public entity {
  //public std::enable_shared_from_this<kernel> {
private:
protected:
  //! Pre-basic constructor
  kernel()
    : dependencies(),
      entity(entity_cookie::KERNEL) {};
  //! Origin kernel constructor. \todo can we delegate to set the cookie?
public:
  //! Origin kernel only has an output object
  kernel( std::shared_ptr<object> out ) : kernel() {
    out_object = out;
    type = kernel_type::ORIGIN; set_name("origin kernel"); };
  //! Compute kernel constructor. \todo can we delegate this one too?
  kernel( std::shared_ptr<object> in,std::shared_ptr<object> out) : kernel(out) {
    add_in_object(in);
    type = kernel_type::COMPUTE; set_name("compute kernel"); };
  // copy constructor
  kernel( kernel *old );

  /*
   * Type
   */
protected:
  kernel_type type{kernel_type::UNDEFINED};
public:
  void set_type( kernel_type t ) { type = t; };
  kernel_type get_type() { return type; };
  int has_type_origin() { return type==kernel_type::ORIGIN; };
  int has_type_compute() { return type==kernel_type::COMPUTE; };
  int has_type_trace() { return type==kernel_type::TRACE; };

protected:
  int step_counter{0};
public:
  int get_step_counter() { return step_counter; };
  void set_step_counter( int s ) { step_counter = s; };
  //! \todo lose this one.
  int get_step() { return get_out_object()->get_object_number(); };

  /*
   * Output
   */
protected:
  std::shared_ptr<object> out_object{nullptr};
public:
  std::shared_ptr<object> get_out_object() {
    if (out_object==nullptr)
      throw(std::string("Kernel does not have out object"));
    return out_object;};
  //! We can get the gamma distribution from casting the output object.
  //! \todo find uses for this. there is a bunch in mpi_ops.h
  distribution *get_gamma_distribution() {
    return dynamic_cast<distribution*>(get_out_object().get()); };

  /*
   * Local execution
   */
public: //protected:
  std::function
  < kernel_function_proto > localexecutefn;
  void *localexecutectx{nullptr};
public:
  //! Set the task-local function; in the product mode this is overriden.
  virtual void set_localexecutefn
      ( std::function< kernel_function_proto > f ) { localexecutefn = f; };

  //! Set the task-local context; in the product mode this is overriden.
  virtual void set_localexecutectx( void *ctx ) {
    localexecutectx = ctx; };

  /*
   * Input dependencies
   */
public:
  //! If we add objects to an origin kernel, it turns into a compute
  void add_in_object( std::shared_ptr<object> in ) override {
    dependencies::add_in_object(in); set_type(kernel_type::COMPUTE); };
  //! \todo this needs to move to the dependencies class
  void set_type_local() { last_dependency()->set_type_local(); };
  // function
  void set_signature_function_function( const sigma_operator &f ) {
    last_dependency()->set_signature_function_function(f); };
  void set_signature_function_function( multi_sigma_operator *f ) {
    last_dependency()->set_signature_function_function(f); };
  void set_signature_function_function
  ( std::function< std::shared_ptr<indexstruct>(index_int) > f ) {
    last_dependency()->set_signature_function_function(f); };
    
  // explicit
  void set_explicit_beta_distribution( distribution *d) {
    last_dependency()->set_explicit_beta_distribution(d); };
  void set_explicit_beta_distribution( int n,distribution *d) {
    get_dependencies().at(n)->set_explicit_beta_distribution(d); };

  int has_beta_distribution() { return last_dependency()->has_beta_distribution(); };
  distribution *get_beta_distribution() {
    return last_dependency()->get_beta_distribution(); };
  distribution *get_beta_distribution(int n) {
    return get_dependencies().at(n)->get_beta_distribution(); };

  /*
   * task & queue stuff
   */
protected:
  std::vector< std::shared_ptr<task> > kernel_tasks;
  int was_split_to_tasks{0};
public:
  int kernel_has_tasks() const { return was_split_to_tasks; };
  void set_kernel_tasks(std::vector< std::shared_ptr<task> >);
  void addto_kernel_tasks(std::vector< std::shared_ptr<task> >);
  const std::vector< std::shared_ptr<task> > &get_tasks() const;
  // Split a kernel to tasks; we will override this in composite kernels.
  virtual void split_to_tasks();

  /*
   * Analysis stuff
   */
protected:
  int has_been_analyzed{0};
public:
  void set_has_been_analyzed() { has_been_analyzed = 1; };
  int get_has_been_analyzed() { return has_been_analyzed; };
protected:
public:
  //! For most purposes the default works; not if we compose kernels into a new kernel
  virtual void analyze_dependencies();
  void analyze_contained_kernels( kernel*,kernel* );
  void analyze_contained_kernels( kernel*,kernel*,kernel* );
  void split_contained_kernels( kernel*,kernel* );
  void split_contained_kernels( kernel*,kernel*,kernel* );

  //! For most purposes the default works; not if we compose kernels into a new kernel
  virtual void execute();

  //! This is a factory routine for origin tasks: each mode produces its own type
  virtual std::shared_ptr<task> make_task_for_domain(processor_coordinate&) = 0;

  //! A factory for reduction routines: MPI is just so different from OpenMP
  static std::function< kernel*(std::shared_ptr<object>,std::shared_ptr<object>) > make_reduction_kernel;

  // statistics
  virtual std::string as_string(); // is overridden for composite kernels
  int local_nmessages();
};

/*!
  An origin kernel takes whatever data it finds in its allocation.
*/
class origin_kernel : virtual public kernel {
public:
  origin_kernel(std::shared_ptr<object> out,std::string name=std::string("origined object"))
    : kernel(out),entity(entity_cookie::KERNEL) {
    out->allocate();
    set_localexecutefn( &vecnoset );
    set_name(name);
  };
};

/****
 **** Task
 ****/

/*!
  Local executability is related to not having synchronization points.
  However, it is a recursively defined quantity, related to a task predecessors.
  Thus, its meaning is clear for omp-embedded-in-product; not so much in other cases.

  We use the relative ordering that YES/NO > UNKNOWN
*/
enum class task_local_executability { INVALID,UNKNOWN,NO,YES };

class algorithm;
/*!
  A task is the execution of a \ref kernel on one specific processor.
  The task object is mostly created from a kernel: one should never create them
  explicitly. Data members such as #localexecutefn and #localexecutectx
  are inherited (in an informal sense) from the kernel. Right now we don't keep
  track of the kernel from which a task comes, other than through its #step
  value.
*/
class task : virtual public kernel,virtual public entity,
	     public std::enable_shared_from_this<task> {
protected:
  processor_coordinate domain; //!< the processor number of this task in the kernel
  static int count;
  int task_number{-1};
public:
  std::function< std::shared_ptr<task>(int,processor_coordinate&) > find_other_task_by_coordinates{nullptr};
  std::function<
    std::shared_ptr<task>(processor_coordinate&) > find_kernel_task_by_domain{
    nullptr};
  virtual void make_requests_vector() = 0;
  virtual void delete_requests_vector() = 0;
  //! Create a task that belongs to an already existing kernel
  task(processor_coordinate &d,kernel *k)
    : kernel(k),
      entity(dynamic_cast<entity*>(k),entity_cookie::TASK) {
    domain = d; task_number = count++; set_name("unnamed-task"); };
  //! Create an origin task for standalone testing: create a kernel that is normally inherited
  task(processor_coordinate &d,std::shared_ptr<object> out)
    : kernel( *out->new_kernel_from_object(out) ),
      entity(dynamic_cast<entity*>(out.get()),entity_cookie::TASK) {
    domain = d; task_number = count++;
    set_type( kernel_type::ORIGIN ); set_name("unnamed-origin-task"); };
  //! Create a compute task for standalone testing: create a kernel that is normally inherited
  task(processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out)
    : kernel( *out->new_kernel_from_object(out) ),
      entity(dynamic_cast<entity*>(in.get()),entity_cookie::TASK) {
    if (in==NULL || in==nullptr) throw(std::string("using wrong task constructor"));
    domain = d; task_number = count++;
    add_in_object(in);
    set_type(kernel_type::COMPUTE); set_name("unnamed-compute-task"); };
  //! \todo can we make this const?
  processor_coordinate &get_domain() { return domain; };

  /*
   * Relation management
   */
protected:
  //! The step/domain coordinates of the predecessors; this can be locally determined.
  std::vector<task_id*> predecessor_coordinates;
  //! The actual predecessors in the (local) task queue
  std::vector<std::shared_ptr<task>> predecessors;
  //! Message from buffer. This is purely for MPI, but too hard to put it in the derived class
  std::function< message*(int,std::string&) > message_from_buffer {
    [] (int p,std::string &buf) -> message* {
      throw(std::string("basic msg from buf")); } };
public:
  //! Declare a local dependence relation, that is, on the same address space even if the dependence is elsewhere
  virtual void declare_dependence_on_task( task_id* ) = 0;
  std::vector<task_id*> &get_predecessor_coordinates() { return predecessor_coordinates; };
  std::vector<std::shared_ptr<task>> &get_predecessors() { return predecessors; };
  void add_predecessor( std::shared_ptr<task> t ) { predecessors.push_back(t); };

  /*
   * Requests
   */
protected:
  request_vector *requests{nullptr},*pre_requests{nullptr};
public:
  request_vector *get_requests() {
    if (requests==nullptr) throw(std::string("task has no requests"));
    return requests; };
  int get_n_outstanding_requests() { return get_requests()->size(); }
  void set_pre_requests( request_vector *v ) { pre_requests = v; };
  bool communication_completed() { return requests->completed(); };

protected:
  std::shared_ptr<algorithm> node_queue{nullptr}; // only used for OMP
public:
  //! Get the embedded queue.
  std::shared_ptr<algorithm> get_node_queue() { return node_queue; };

  /*
   * Split execution
   */
protected:
  int (*unsynctest)( std::shared_ptr<task> t) {nullptr};
  int (*synctest)( std::shared_ptr<task> t) { & task::task_test_true };
public:
  //! Replace the default tests for task selection. Usually called by surrounding queue.
  void set_sync_tests
      ( int(*t1)(std::shared_ptr<task>), int(*t2)(std::shared_ptr<task>) ) {
    unsynctest = t1; synctest = t2; };
  static int task_test_true( std::shared_ptr<task> t ) { return 1; };
  static int task_test_false( std::shared_ptr<task> t ) { return 0; };

  /*
   * Execution stuff
   */
  void execute();
  virtual void execute_as_root();
  // basic local execute function
  virtual void local_execute
      (std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,void*);
  // conditional execution, overriden for product
  virtual void local_execute
      (std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,void*,
       int(*)(std::shared_ptr<task>));
  //! Missing in object is probably an origin
  void local_execute(std::shared_ptr<object> out) {
    std::vector<std::shared_ptr<object>> nullobjs;
    local_execute(nullobjs,out,nullptr,task::task_test_true); };
protected:
  bool non_exit{false};
public:
  void set_is_non_exit() { non_exit = true; };
  bool is_non_exit() { return non_exit; };
protected:
  int circular{-1};          //!< Has this task been found part of a cycle? -1=untested 0=not-circular 1=circular (unused) \todo lose this var: abort if circular tested
public:
  int check_circularity( std::shared_ptr<task> root );
  //! Declare that this task has been tested to be not in a cycle
  void set_not_circular() { circular = 0; };
  //! Test that this task has been tested to be not in a cycle
  int is_not_circular() { return circular==0; };
protected:
  int done{0};               //!< has this task been executed?
public:
  //! Record task execution; for OMP we also record the thread.
  virtual void set_has_been_executed() { done += 1; };
  void clear_has_been_executed();
  int get_has_been_executed() { return done!=0; };
  int get_number_of_executions() { return done; }; //!< To get whether executed exactly once.

  // synchronization functions
  void notifyReadyToSend( std::vector<message*> &msgs,request_vector* );
  virtual void acceptReadyToSend( std::vector<message*> &msgs,request_vector* ) = 0;
  virtual request *notifyReadyToSendMsg( message* ) = 0;

  /*
   * Synchronization
   */
protected:
  int is_synchronization_point{0};
public:
  void set_is_synchronization_point() { is_synchronization_point = 1; };
  int get_is_synchronization_point() { return is_synchronization_point; };
protected:
  task_local_executability can_execute_locally{task_local_executability::UNKNOWN};
public:
  void set_can_execute_locally( task_local_executability c ) { can_execute_locally = c;};
  void set_can_execute_locally() {  set_can_execute_locally(task_local_executability::YES); };
  task_local_executability get_local_executability() { return can_execute_locally;};
  //! Recursive determination of local executability is only defined for OpenMP.
  virtual void check_local_executability() {};

  /*
   * Analysis
   */
protected:
  int has_been_analyzed{0};  //!< has this task been analyzed?
  int has_been_optimized{0}; //!< has this task been optimized; see algorithm::optimize
public:
  void analyze_dependencies();
  //! Most tasks don't need local analysis, but the product mode definitely does.
  virtual void local_analysis() {};
  dependency *find_dependency_for_object_number(int);
  int get_has_been_analyzed() { return has_been_analyzed; };
  void set_has_been_analyzed() { has_been_analyzed = 1; };

  /*
   * Messages
   */
protected:
  //! \todo just use a vector
  std::vector<message*> send_messages;//{new std::vector<message*>};
  std::vector<message*> recv_messages;
public:
  // derive messages and store internally
  void derive_receive_messages();
  //! By default we don't need send messages; MPI is an exception
  virtual void derive_send_messages(/*int,int*/) {};
  std::vector<message*> &get_send_messages();
  void set_send_messages( std::vector<message*>&);
  void set_receive_messages( std::vector<message*>& );
  std::vector<message*> &get_receive_messages();
  std::vector<message*> lift_recv_messages();
  std::vector<message*> lift_send_messages();
  message *matching_send_message(message *rmsg); int get_nsends();

  // Post & Xpct messages
protected:
  std::vector<message*> post_messages; //{new std::vector<message*>};
  std::vector<message*> xpct_messages; 
public:
  void add_post_messages( std::vector<message*> &msgs );
  std::vector<message*> &get_post_messages();
  void add_xpct_messages( std::vector<message*> &msgs );
  std::vector<message*> &get_xpct_messages();

  int get_has_been_optimized() { return has_been_optimized; };
  void set_has_been_optimized() { has_been_optimized = 1; };
protected:
  int nmessages_sent{0};
public:
  std::function< void(int) > record_nmessages_sent{ 
    [this] (int n) -> void { nmessages_sent+=n; } }; 
public:

  /*
   * Tracing and statistics
   */
public:
  std::function< void(double) > record_flop_count{ [] (double c) -> void { return; } };
protected:
  std::function< void(std::shared_ptr<task>) > result_monitor{
    [] (std::shared_ptr<task> t) -> void { return; } };
public:
  void set_flop_count( double c ) { this->record_flop_count(c); };
  virtual std::string as_string();
  std::string header_as_string() {
    return fmt::format("<{}-{}>:{}",get_step_counter(),get_domain().as_string(),get_name()); };
  void set_result_monitor( std::function< void(std::shared_ptr<task> t) > m ) {
    result_monitor = m; };
};

/****
 **** Queue
 ****/

enum class algorithm_type { UNDEFINED,MPI,OMP };

/*!
  A task queue contains both a vector of kernels and a vector of tasks.

  \todo write a unittest for kernel graphs with no overlap
  \todo define overlap in a kernel graph and measure
 */
class algorithm : public decomposition,public communicator,public tracer {
private:
protected:
  algorithm_type type{algorithm_type::UNDEFINED};
  int circular{0}; //!< does this queue contain circularities?
public: // routines
  algorithm(decomposition *d)
    : decomposition(*d),entity(entity_cookie::QUEUE) {
    set_name("unnamed-queue"); };
  ~algorithm() { delete all_kernels; };
  void add_kernel(kernel *k);
  int has_type(algorithm_type t) { return type==t; };

  /*
   * Data access
   */
protected:
  std::vector<kernel*> *all_kernels{new std::vector<kernel*>};
  std::vector<std::shared_ptr<task>> tasks;
public:
  std::vector<kernel*> *get_kernels() { return all_kernels; };
  std::vector<std::shared_ptr<task>> &get_tasks() { return tasks; };
  int global_ntasks() { return allreduce( tasks.size() ); };

  /*
   * Analysis
   */
protected:
  int has_been_analyzed{0}; //!< Has this queue been analyzed?
  int has_been_split_to_tasks{0}; //!< A queue needs to be split only once
public:
  void set_has_been_split_to_tasks() { has_been_split_to_tasks = 1; };
  int get_has_been_split_to_tasks() { return has_been_split_to_tasks; };
  void split_to_tasks();
  // analyze a queue; this is remarkably uniform over the modes,
  // but we have a hook for mode-dependent
  void analyze_dependencies();
  //! Hook for mode-dependent algorithm analysis
  virtual void mode_analyze_dependencies() {};
  // Kernel-only (that is, not queue-level) analysis
  void analyze_kernel_dependencies();
  // synchronization points so far only for OMP
  virtual void find_synchronization_points() {};
  int get_has_been_analyzed() { return has_been_analyzed; };
  void set_has_been_analyzed() { has_been_analyzed = 1; };
  void get_data_relations(const char*,kernel **src,std::vector<kernel*> **tar );
  void optimize();
  void set_circular() { circular=1; };
  int is_circular() { return circular==1; };

  /*
   * Execution
   */
   //! Execute with null selection test
  //  virtual void execute() { execute( &task::task_test_true); };
  virtual void execute( int(*)(std::shared_ptr<task>)=&task::task_test_true );
  void execute_tasks() { execute_tasks(nullptr); };
  // execute tasks, with a task selection test
  virtual void execute_tasks( int(*)(std::shared_ptr<task>) );
  void execute_task( int n ) { tasks.at(n)->execute(); };;
  int get_all_tasks_executed();
  void inherit_data_from_betas();
  void clear_has_been_executed();

  /*
   * Split execution
   */
protected:
  int (*unsynctest)(std::shared_ptr<task> t) {  &task::task_test_false };
  int (*synctest)(std::shared_ptr<task> t) {  &task::task_test_true };
public:
  //! Replace the default tests for task selection.
  void set_sync_tests( int(*t1)(std::shared_ptr<task>), int(*t2)(std::shared_ptr<task>) ) {
    unsynctest = t1; synctest = t2; };

  // graph management
  void add_kernel_tasks_to_queue( kernel *k );
  void add_task( std::shared_ptr<task>& );
  std::shared_ptr<task> &get_task(int n);

protected:
  //! Each kernel has a unique id in the queue; see \ref add_kernel
  int kernel_counter{0}; 
public:
  //int get_kernel_count() { return kernel_counter; };
  void set_kernel_zero( int z ) {
    if (all_kernels->size()>0)
      throw(fmt::format("Should not set kernel zero for non-empty algorithm"));
    kernel_counter = z;
  };
  int get_kernel_counter_pp() { return kernel_counter++; };

  int size() { return tasks.size(); };
  std::shared_ptr<task> find_task_by_id(int); int find_domain_from_id(int);
  int find_task_number_by_coordinates( int s,processor_coordinate &d );
  std::shared_ptr<task> find_task_by_coordinates( int s,processor_coordinate &d );
  void find_predecessors();
protected:
  std::vector<std::shared_ptr<task>> exit_tasks;
public:
  void find_exit_tasks();
  std::vector<std::shared_ptr<task>> &get_exit_tasks() { return exit_tasks; };

  /*
   * Statistics and output
   */
  //! Record flops performed, typically from \ref task::execute
protected:
  double flopcount{0.};
public:
  timed_event analysis_event; timed_event execution_event;
  virtual void record_flop_count( double c ) { flopcount += c; };
  double get_flop_count() { return flopcount; };
  
  //  virtual void gather_statistics();
  virtual std::string kernels_as_string();
  std::string header_as_string();
  std::string contents_as_string();
  //  void print(environment *e);
public: // should be protected
  static int queue_trace_summary;
  static bool do_optimize;
public:
  void set_trace_summary() { queue_trace_summary = 1; }; // how do you call this?
  int get_trace_summary() { return queue_trace_summary; };

  /*
   * Synchronization tasks
   */
protected:
  int has_synchronization_points{-1};
public:
  //! Set the number of detected synchronization tasks
  void set_has_synchronization_tasks( int snc ) {
    if (snc<0) throw(fmt::format("Suspicious sync#: {}",snc));
    has_synchronization_points = snc; };
  int get_has_synchronization_tasks();
  // we split the embedded execute_tasks in local and nonlocal
  void determine_locally_executable_tasks();
  // fake routine for identifying synchronization tasks
  void set_outer_as_synchronization_points();

};

#endif
