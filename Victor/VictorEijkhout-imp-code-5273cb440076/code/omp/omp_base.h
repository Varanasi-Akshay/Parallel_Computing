// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** omp_base.h: Headers for the OpenMP classes
 ****
 ****************************************************************/
#ifndef OMP_BASE_H
#define OMP_BASE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>
#include "imp_base.h"
#include "imp_functions.h"

/****
 **** Basics
 ****/
#include "omp_environment.h"
  
/****
 **** Architecture
 ****/

// index_int omp_allreduce(index_int contrib);
// double omp_allreduce_d(double contrib);
//int omp_allreduce_and(int contrib);
std::vector<index_int> *omp_gather(index_int contrib);

/*!
  In an OpenMP architecture we can only ask for \ref nprocs; 
  we do not override the exception when asking for 
  \ref architecture_data::mytid or
  \ref architecture_data::nthreads_per_node
 */
class omp_architecture : public architecture {
protected:
public :
  omp_architecture( int n )
    : architecture(n),entity(entity_cookie::ARCHITECTURE) {
    type = architecture_type::SHARED; protocol = protocol_type::OPENMP;
    beta_has_local_addres_space = 0;
    set_name(fmt::format("omp-architecture-on-{}",n));
  };
  // omp_architecture( omp_architecture &a )
  //   : architecture(a),entity(entity_cookie::ARCHITECTURE) {};    

  virtual std::string as_string() override;
  virtual void set_power_mode() override { set_can_embed_in_beta(); };

  /*
   * Collectives
   */
  std::vector<index_int> *omp_gather(index_int contrib);
  int omp_reduce_scatter(int *senders,int root);
};

class omp_decomposition : public decomposition {
public:
  omp_decomposition( architecture *arch,processor_coordinate &grid )
    : decomposition(arch,grid),entity(entity_cookie::DECOMPOSITION) {
    int ntids = arch->nprocs();
    for (int mytid=0; mytid<ntids; mytid++) {
      auto mycoord = this->coordinate_from_linear(mytid);
      add_domain(mycoord);
    }
    set_decomp_factory();
  };
  omp_decomposition( architecture *arch,processor_coordinate &&grid )
    : decomposition(arch,grid),entity(entity_cookie::DECOMPOSITION) {
    int ntids = arch->nprocs();
    for (int mytid=0; mytid<ntids; mytid++) {
      auto mycoord = this->coordinate_from_linear(mytid);
      add_domain(mycoord);
    }
    set_decomp_factory();
  };
  //! Default constructor is one-d.
  omp_decomposition( architecture *arch )
    : omp_decomposition(arch,arch->get_proc_layout(1)) {};
  void set_decomp_factory();
  virtual std::string as_string() override {
    return fmt::format("omp {}",decomposition::as_string());
  };
};

/*! A container for collective routines, specifically in MPI
  \todo should we take architecture as an input, and copy the routines?
*/
// class omp_communicator : virtual public communicator {
// public:
//   omp_communicator() : communicator() {
//     the_communicator_mode = communicator_mode::OMP;
//     make_omp_communicator();
//   };
//   //! \todo this will also be called with ndomains. hm.
//   void make_omp_communicator() {
//     nprocs = [] (void) -> int { int np;
// #pragma omp parallel
// 				np = omp_get_num_threads();
// 				return np; };
//     make_omp_communicator(nprocs());
//   }

void make_omp_communicator(communicator *cator,int P);

/****
 **** Distribution
 ****/

index_int omp_location_of_first_index(distribution &d,processor_coordinate &pcoord);
index_int omp_location_of_last_index(distribution &d,processor_coordinate &pcoord);

//! OpenMP distribution differs from MPI in the use of shared memory.
class omp_distribution : virtual public distribution {
public:
  omp_distribution(decomposition *d);
  omp_distribution( parallel_structure *struc );
  omp_distribution( decomposition *d,index_int(*pf)(int,index_int),index_int nlocal );
  //! Copy constructor from other distribution
  omp_distribution( distribution *other )
    : distribution(other),
      entity(entity_cookie::DISTRIBUTION) {
    set_dist_factory(); set_numa(); set_operate_routines();
    set_type(other->get_type()); }
  void set_dist_factory(); void set_memo_routines();

  void set_operate_routines() { //!< \todo should we maybe not capture this, but add an arg?
    // operate_base =
    //   [this] (std::shared_ptr<ioperator> op) ->distribution* {
    //   return omp_dist_operate_base(op); };
    // distr_union =
    //   [this] (distribution *other) -> distribution* { return omp_dist_distr_union(other); };
  };
  //  distribution *omp_dist_operate_base( std::shared_ptr<ioperator> );
  distribution *omp_dist_distr_union( distribution* );

  // NUMA
  void set_numa();
};

class omp_block_distribution : public omp_distribution,public block_distribution {
public:
  //! OpenMP block distribution from local / global
  omp_block_distribution(decomposition *d,int o,index_int l,index_int g)
    : distribution(d),omp_distribution(d),block_distribution(d,o,l,g),
      entity(entity_cookie::DISTRIBUTION) {
    try { memoize();
    } catch (std::string c) {
      throw(fmt::format("Failed to memoize omp block distr: {}",c));
    }
  };
  omp_block_distribution(decomposition *d,index_int l,index_int g)
    : omp_block_distribution(d,1,l,g) {};
  omp_block_distribution(decomposition *d,index_int g)
    : omp_block_distribution(d,-1,g) {};
  //! Multi-d constructor takes an endpoint vector: array of global sizes
  omp_block_distribution(decomposition *d,std::vector<index_int> endpoint)
    : distribution(d),omp_distribution(d),block_distribution(d,endpoint),
      entity(entity_cookie::DISTRIBUTION) {
    d->get_same_dimensionality(endpoint.size());
    try { memoize();
    } catch (std::string c) {
      throw(fmt::format("Failed to memoize omp block distr: {}",c));
    }
  };
  //! Constructor from vector of explicit block sizes. The gsize is ignored.
  omp_block_distribution(decomposition *d,std::vector<index_int> lsizes,index_int gsize)
    : distribution(d),omp_distribution(d),block_distribution(d,lsizes,gsize),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  //! Constructor from multi_indexstruct
  omp_block_distribution( decomposition *d,std::shared_ptr<multi_indexstruct> idx )
    : distribution(d),omp_distribution(d),block_distribution(d,idx),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
};

class omp_scalar_distribution : public omp_distribution,public scalar_distribution {
public:
  omp_scalar_distribution(decomposition *d)
    : distribution(d),omp_distribution(d),scalar_distribution(d),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
};

class omp_cyclic_distribution : public omp_distribution,public cyclic_distribution {
public:
  omp_cyclic_distribution(decomposition *d,int l,int g)
    : distribution(d),omp_distribution(d),cyclic_distribution(d,l,g),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  omp_cyclic_distribution(decomposition *d,int l) : omp_cyclic_distribution(d,l,-1) {};
};

class omp_replicated_distribution : public omp_distribution,public replicated_distribution {
public:
  omp_replicated_distribution(decomposition *d,int ortho,index_int l)
    : distribution(d),omp_distribution(d),replicated_distribution(d,ortho,l),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  omp_replicated_distribution(decomposition *d,index_int l)
    : omp_replicated_distribution(d,1,l) {};
  omp_replicated_distribution(decomposition *d)
    : omp_replicated_distribution(d,1) {};
  // //! Each domain has its own block of data
  // omp_replicated_distribution(decomposition *d,int l)
  //   : distribution(d),omp_distribution(d),replicated_distribution(d,l),
  //     entity(entity_cookie::DISTRIBUTION) { memoize(); };
  // omp_replicated_distribution(decomposition *d) : omp_replicated_distribution(d,1) {};
};

class omp_gathered_distribution : public omp_distribution,public gathered_distribution {
public:
  omp_gathered_distribution(decomposition *d,int l)
    : distribution(d),omp_distribution(d),gathered_distribution(d,1,l) ,
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  omp_gathered_distribution(decomposition *d) : omp_gathered_distribution(d,1) {};
};

/****
 **** Sparse matrix / index pattern
 ****/

class omp_sparse_matrix : public sparse_matrix {
 public:
  omp_sparse_matrix( distribution *d )
    : sparse_matrix(d->get_dimension_structure(0)->get_enclosing_structure()) {
    globalsize = d->global_volume();
  };
  omp_sparse_matrix( distribution *d,index_int g ) : omp_sparse_matrix(d) { globalsize = g; };
};

//! Make upper bidiagonal toeplitz matrix.
class omp_upperbidiagonal_matrix : public omp_sparse_matrix {
public:
  omp_upperbidiagonal_matrix( distribution *dist, double d,double r )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row+1; if (col<g) add_element(row,col,r);
    }
  }
};

//! Make lower bidiagonal toeplitz matrix.
class omp_lowerbidiagonal_matrix : public omp_sparse_matrix {
public:
  omp_lowerbidiagonal_matrix( distribution *dist, double l,double d )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
    }
  }
};

class omp_toeplitz3_matrix : public omp_sparse_matrix {
public:
  omp_toeplitz3_matrix( distribution *dist, double l,double d,double r )
    : omp_sparse_matrix(dist) {
    index_int g = dist->global_volume();
    for (index_int row=0; row<g; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
      col = row+1; if (col<g) add_element(row,col,r);
    }
  }
};

/****
 **** Beta object
 ****/

/****
 **** Object
 ****/

/*!
  Data in an omp_object is allocated only once. The \ref get_data method
  returns this pointer, regardless on what `processor' is it requested.
  This is the right strategy, since each processor has its own first/last
  index.

  However, with a replicated object, each processor has the same first/last
  index, yet we still need to accomodate multiple instances of the data.
  Therefore, the constructor overallocated the data in this case,
  and \ref get_data(int) returns a unique pointer into the (single, overdimensioned)
  array.
 */
class omp_object : public object {
private:
protected:
public:
  //! Create an object from a distribution, locally allocating the data
  omp_object( distribution *d )
    : object(d),
      entity(entity_cookie::OBJECT) {
    set_data_handling();
    // if objects can embed in a halo, we'll allocate later; otherwise now.
    if (!d->get_can_embed_in_beta()
	|| d->get_type()==distribution_type::REPLICATED)
      omp_allocate();
  };
  /*! Create an object from user data. This is dangerous for replicated data;
    see constructor with extra integer parameter 
    \todo can we delegate this? \todo the registration is wrong
  */
  omp_object( distribution *d, double *dat )
    : object(d),
      entity(entity_cookie::OBJECT) {
    set_data_handling();
    if (get_type()==distribution_type::REPLICATED)
      throw(std::string("too dangerous to create omp object from data"));
    for ( auto dom : get_domains() ) {
      index_int domain_offset = location_of_first_index(*d,dom);
      register_data_on_domain(dom,dat+domain_offset,0);
    }
    data_status = object_data_status::INHERITED;
  };
  //! Create from user data; this requires explicit size, for our own protection.
  omp_object( distribution *d, double *dat,index_int s )
    : object(d),
      entity(entity_cookie::OBJECT) {
    int nd = get_domains().size();
    if (get_type()==distribution_type::REPLICATED)
      if (s!=d->local_allocation()) throw("supplied data size mismatch");
    for ( auto dom : get_domains() ) {
      index_int domain_offset = location_of_first_index(*d,dom);
      register_data_on_domain(dom,dat+domain_offset,0);
    }
    data_status = object_data_status::INHERITED;
  };
  //! Create an object from data of another object. \todo revisit registration
  omp_object( distribution *d, std::shared_ptr<object> x )
    : object(d),
      entity(entity_cookie::OBJECT) { set_data_handling();
    int nd = get_domains().size();
    for ( auto dom : get_domains() ) {
      int nd = get_domain_local_number(dom);
      register_data_on_domain_number(nd,x->get_data_p(dom),0);
    }
  };
  // this is just to make static casts possible
  void poly_object() { printf("this serves no purpose\n"); };

  /*
   * Data
   */
protected:
  double *omp_get_data(processor_coordinate &p);
  //! The OpenMP object allocation has a special case for replicated objects
  void omp_allocate();
  //! Install all omp-specific data routines
  void set_data_handling() {
    get_data_p = [this] (processor_coordinate &p) -> double*  {
      return omp_get_data(p); };
    get_data_pp = [this] (processor_coordinate &&p) -> double*  {
      return omp_get_data(p); };
    allocate = [this] (void) -> void { omp_allocate(); };
  };
    
public:
  std::string values_as_string();
  //! Mask shift compensates for the location of \ref object::first_index.
  virtual domain_coordinate mask_shift( processor_coordinate &p ) {
  //virtual index_int mask_shift(int p) {
    if (!lives_on(p))
      throw(fmt::format("Should not ask mask shift for {} on {}",
			p.as_string(),get_name()));
    throw(fmt::format("mask shift not implemented"));
    // index_int s = 0;
    // for (int q=0; q<p; q++) if (!lives_on(q)) s += internal_local_size(q);
    // return s;
  };
  virtual void copy_data_from( std::shared_ptr<object> in,message *smsg,message *rmsg ) override;
};

/****
 **** Message
 ****/

class omp_message : public message,virtual public entity {
public:
  omp_message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> &e,std::shared_ptr<multi_indexstruct> &g)
    : message(d,snd,rcv,e,g),entity(entity_cookie::MESSAGE) {};
  omp_message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> &g)
    : message(d,snd,rcv,g),entity(entity_cookie::MESSAGE) {};
public:
  //! \todo why do we need this? is there a more basic functionality?
  message *send_msg{nullptr};
};

/****
 **** Requests
 ****/

/*! 
  In OpenMP a request is a task on which we depend. 
  Fullfilling the wait involves copying data.
*/
class omp_request : public request{
public:
  std::shared_ptr<task> tsk{nullptr};
  std::shared_ptr<object> obj{nullptr};
  int closed{0};
public:
  omp_request( std::shared_ptr<task> t,message *m,std::shared_ptr<object> o,request_type type=request_type::INCOMING )
    : request(m,request_protocol::OPENMP) { tsk = t; obj = o; this->type = type; };
  void poly() { return; }; //!< Very silly: just to make dynamic casts possible.
};

class omp_request_vector : public request_vector {
protected:
public:
  omp_request_vector() : request_vector() {};
  virtual void wait();
};

/****
 **** Kernel
 ****/

class omp_kernel : virtual public kernel {
private:
public:
  omp_kernel() : kernel(),entity(entity_cookie::KERNEL){};
  omp_kernel( std::shared_ptr<object> out ) : kernel(out),entity(entity_cookie::KERNEL) {};
  omp_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),entity(entity_cookie::KERNEL) {};
  omp_kernel( const kernel& k ) : kernel(k),entity(entity_cookie::KERNEL) {};
  std::shared_ptr<task> make_task_for_domain(processor_coordinate&);
};

class omp_origin_kernel : public omp_kernel,public origin_kernel {
public:
  omp_origin_kernel( std::shared_ptr<object> out )
    : omp_kernel(out),kernel(out),origin_kernel(out),entity(entity_cookie::KERNEL) {};
};

/****
 **** Task
 ****/

class omp_task : public task,public omp_kernel {
private:
protected:
  omp_lock_t dolock; //!< this locks the done variable of the base class
public: // methods
  void make_requests_vector() { requests = new omp_request_vector(); };
  void delete_requests_vector() { delete requests; };
  omp_task(processor_coordinate &d,kernel *k)
    : kernel(*k),task(d,k),omp_kernel(*k),entity(entity_cookie::TASK) {
    make_requests_vector(); omp_init_lock(&dolock); };
  //! Create origin kernel from anonymous kernel
  omp_task(processor_coordinate &d,std::shared_ptr<object> out)
    : task(d,new omp_kernel(out)),entity(entity_cookie::TASK) {
    make_requests_vector(); omp_init_lock(&dolock); };
  //! Create compute kernel from anonymous kernel
  omp_task(processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out)
    : task(d,new omp_kernel(in,out)),entity(entity_cookie::TASK) {
    make_requests_vector(); omp_init_lock(&dolock); };

  virtual void derive_send_messages(/*int,int*/) override ;
  virtual void declare_dependence_on_task( task_id *id ); // pure virtual

  // pure virtual synchronization functions
  //! In OpenMP there is no need to alert the sending party that we are ready to receive.
  virtual void notifyReadyToSend( std::vector<message*>&,request_vector* ) {};
  //! We make a dummy request for outgoing stuff. OpenMP doesn't need to send.  
  request *notifyReadyToSendMsg( message* msg ) {
    return new omp_request(this->shared_from_this(),msg,msg->get_out_object(),request_type::OUTGOING); };
  virtual void acceptReadyToSend( std::vector<message*>&,request_vector* );

  virtual void create_send_structure_for_dependency(int,int,dependency*) {};
  void make_infrastructure_for_sending( int n ) {}; // another pure virtual no-op

  /*
   * Execution stuff
   */
  void execute_as_root() override; // same as base method, but with directives
protected:
  int done_on_thread{-1};
public:
  virtual void set_has_been_executed() override { task::set_has_been_executed();
    done_on_thread = omp_get_thread_num(); };
  int get_done_on_thread() { return done_on_thread; };
  virtual void check_local_executability() override;

};

class omp_origin_task : public omp_task {
public:
  omp_origin_task( processor_coordinate &d,std::shared_ptr<object> out )
    : omp_task(d,out),entity(entity_cookie::TASK) {
    set_type( kernel_type::ORIGIN ); set_name("origin omp task");
  };
};

/****
 **** Queue
 ****/

class omp_algorithm : public algorithm {
private:
protected:
public:
  omp_algorithm(decomposition *d)
    : algorithm(d),entity(entity_cookie::QUEUE) {
    type = algorithm_type::OMP;
    make_omp_communicator(this,this->domains_volume());
  }

  //! \todo why is it not taking the one from the base class?
  //virtual void execute() override { execute( &task::task_test_true ); };
  // the omp task execute has directives to make it parallel
  virtual void execute( int(*)(std::shared_ptr<task> t)=&task::task_test_true ) override;
  // the omp task execute has directives to make it parallel
  virtual void execute_tasks( int(*)(std::shared_ptr<task> t) ) override;

  //! Local analysis: 1. try to embed vectors in a halo. 2. find non-synchronizing tasks
  virtual void mode_analyze_dependencies() override {
    determine_locally_executable_tasks(); inherit_data_from_betas();
  };

  int find_task( int s,processor_coordinate *d );
};

#endif
