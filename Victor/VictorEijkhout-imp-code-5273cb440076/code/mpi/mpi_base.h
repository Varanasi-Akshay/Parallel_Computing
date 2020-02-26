// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** mpi_base.h: Header file for the MPI derived class
 ****
 ****************************************************************/

#ifndef MPI_BASE_H
#define MPI_BASE_H 1

#include <stdlib.h>
#include <cstdio>
#include <string.h>

#include <mpi.h>
#include <imp_base.h>
#include "imp_functions.h"

#ifdef VT
#include "VT.h"
#endif

/*! \page mpi MPI backend

  We realize an MPI backend by letting tasks synchronize through MPI messages.
  While our API still looks like sequential semantics, we now assume
  an SPMD mode of execution: each MPI process executes not just 
  the same executable, but it also never asks about information that distinguishes
  it from the other processors. This story springs a leak when we consider the
  \ref mpi_sparse_matrix class, where the matrix is constructed on each processor from a
  different local part.

  The MPI backend for IMP is characterized by the following
  - task synchronization has an active component of both sending and receiving;
  - redundant distributions are non-trivial, see \ref mpi_replicated_scalar and \ref mpi_gathered_scalar;
  - an \ref mpi_object has disjoint data allocated on each process;
  - we have some routines for packing and unpacking a message object.
 */

/****
 **** Basics
 ****/
#include "mpi_environment.h"

#ifdef VT
void vt_register_kernels();
#endif

/****
 **** Architecture
 ****/

// prototypes of mpi collectives, defined in mpi_base.cxx
index_int mpi_allreduce(index_int contrib,MPI_Comm comm);
double mpi_allreduce_d(double contrib,MPI_Comm comm);
int mpi_allreduce_and(int contrib,MPI_Comm comm);
//! \todo lose that star
void mpi_gather32(int contrib,std::vector<int>&,MPI_Comm comm);
void mpi_gather64(index_int contrib,std::vector<index_int>&,MPI_Comm comm);
//! \todo lose that star
std::vector<index_int> *mpi_overgather(index_int contrib,int over,MPI_Comm comm);
int mpi_reduce_scatter(int *senders,int root,MPI_Comm comm);
std::vector<index_int> mpi_reduce_max(std::vector<index_int> local_values,MPI_Comm comm);
std::vector<index_int> mpi_reduce_min(std::vector<index_int> local_values,MPI_Comm comm);

void mpi_message_as_buffer( architecture *arch,message *msg,std::shared_ptr<std::string>);
message *mpi_message_from_buffer( std::shared_ptr<task> t,int step,const std::string &buf);

/*!
  MPI architecture data consists of the output of MPI_Comm_rank/size,
  as done in the environment setup.
  Based on these, we offer implementations of \ref nprocs and \ref mytid;
  we do not override \ref architecture_data::nthreads_per_node
*/
class mpi_architecture : public architecture {
public :
  //! Constructor.
  mpi_architecture( int tid,int ntids )
    : architecture(tid,ntids),entity(entity_cookie::ARCHITECTURE) {
    type = architecture_type::SPMD; protocol = protocol_type::MPI;
    beta_has_local_addres_space = 1;
    set_name(fmt::format("mpi-architecture-on-proc{}-out-of-{}",tid,ntids));

    message_as_buffer = [] (architecture *a,message *m,std::shared_ptr<std::string> b) -> void {
      mpi_message_as_buffer(a,m,b); };
    // For MPI we can actually report a `mytid'
    mytid = [this] (void) -> int { return arch_procid; };
  };
  mpi_architecture( int mytid,int ntids,int o ) : mpi_architecture(mytid,ntids) {
    set_over_factor(o); };
  //! Copy constructor
  mpi_architecture( mpi_architecture *a )
    : architecture(a),entity(entity_cookie::ARCHITECTURE) {
    comm = a->comm;
  };

protected:
  MPI_Comm comm;
public:
  void set_mpi_comm( MPI_Comm c ) { comm = c; };
  MPI_Comm get_mpi_comm() { return comm; };

  //! Enable all tricky optimizations
  virtual void set_power_mode() override {
    set_can_embed_in_beta(); set_can_message_overlap(); };

  virtual std::string as_string() override;
};

/*!
  An mpi decomposition has one domain per processor by default,
  unless there is a global over-decomposition parameters.

  No one ever inherits from this, but mode-specific distributions are built from this
  because it contains a distribution factory.
*/
class mpi_decomposition : public decomposition {
public:
  //! Multi-d decomposition from explicit processor grid layout
  mpi_decomposition( architecture *arch,processor_coordinate &&grid)
    : decomposition(arch,grid),entity(entity_cookie::DECOMPOSITION) {
    int mytid = arch->mytid(); int over = arch->get_over_factor();
    for ( int local=0; local<over; local++) {
      auto mycoord = this->coordinate_from_linear(over*mytid+local);
      try {
	add_domain(mycoord);
      } catch (...) { fmt::print("trouble adding domain\n"); };
      //fmt::print("Tid {} translates to domain <<{}>>\n",mytid,mycoord->as_string());
    }
    //compute_global_ndomains();
    set_decomp_factory(); 
  };
  mpi_decomposition( architecture *arch,processor_coordinate &grid)
    : decomposition(arch,grid),entity(entity_cookie::DECOMPOSITION) {
    int mytid = arch->mytid(); int over = arch->get_over_factor();
    for ( int local=0; local<over; local++) {
      auto mycoord = this->coordinate_from_linear(over*mytid+local);
      try {
	add_domain(mycoord);
      } catch (...) { fmt::print("trouble adding domain\n"); };
      //fmt::print("Tid {} translates to domain <<{}>>\n",mytid,mycoord->as_string());
    }
    //compute_global_ndomains();
    set_decomp_factory(); 
  };
  //! Default mpi constructor is one-d.
  mpi_decomposition( architecture *arch )
    : mpi_decomposition(arch,arch->get_proc_layout(1)) {};

  void set_decomp_factory();
  virtual std::string as_string() override {
    return fmt::format("MPI decomposition <<{}>>",decomposition::as_string()); };
};

void make_mpi_communicator(communicator *cator);

/****
 **** Distribution
 ****/

class mpi_object;
/*!
  An MPI distribution inherits from distribution, and therefore it also
  inherits the \ref architecture that the base distribution has.
*/
class mpi_distribution : virtual public distribution {
public:
  mpi_distribution( decomposition *d ); // basic constructor
  mpi_distribution( parallel_structure *struc ); // from structure
  //! Constructor from function.
  mpi_distribution(decomposition *d,index_int(*pf)(int,index_int),index_int nlocal)
    : mpi_distribution(d) {
    get_dimension_structure(0)->create_from_function( pf,nlocal );
    memoize();
  };
  //! Copy constructor from other distribution
  mpi_distribution( distribution *other )
    : distribution(other),
      entity(entity_cookie::DISTRIBUTION) {
    set_dist_factory(); set_numa(); set_operate_routines();
    set_type(other->get_type()); };

  void set_dist_factory(); void set_memo_routines();
  void set_operate_routines() {};
  /*
    // operate_base =
    //   [this] (std::shared_ptr<ioperator> op) ->distribution* {
    //   return mpi_dist_operate_base(op); };
    // distr_union =
    //   [this] (distribution *other) -> distribution* { return mpi_dist_distr_union(other); };
  };
  distribution *mpi_dist_operate_base( std::shared_ptr<ioperator> );
  distribution *mpi_dist_distr_union( distribution* );
  */

  // NUMA
  void set_numa();

  // Resolve, then set mask
  virtual void add_mask( processor_mask *m ) override;
};

class mpi_block_distribution : public mpi_distribution,public block_distribution {
public:
  //! Explicit one-d constructor from ortho,local,global.
  mpi_block_distribution(decomposition *d,int o,index_int l,index_int g)
    : distribution(d),mpi_distribution(d),block_distribution(d,o,l,g),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  //! Constructor with implicit ortho=1.
  mpi_block_distribution(decomposition *d,index_int l,index_int g)
    : mpi_block_distribution(d,1,l,g) {};
  //! Constructor with implicit ortho and local is decided.
  mpi_block_distribution(decomposition *d,index_int g)
    : mpi_block_distribution(d,-1,g) {};
  //! Multi-d constructor takes an endpoint vector: array of global sizes
  mpi_block_distribution(decomposition *d,std::vector<index_int> endpoint)
    : distribution(d),mpi_distribution(d),block_distribution(d,endpoint),
      entity(entity_cookie::DISTRIBUTION) {
    d->get_same_dimensionality(endpoint.size()); memoize(); };
  //! Constructor from vector of explicit block sizes. The gsize is ignored.
  mpi_block_distribution(decomposition *d,std::vector<index_int> lsizes,index_int gsize)
    : distribution(d),mpi_distribution(d),block_distribution(d,lsizes,gsize),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  //! Copy constructor.
  mpi_block_distribution( mpi_distribution *other )
    : distribution((decomposition*)other,other->get_dimension_structure(0)),
      mpi_distribution((decomposition*)other),block_distribution((decomposition*)other),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
};

class mpi_scalar_distribution : public mpi_distribution,public scalar_distribution {
public:
  mpi_scalar_distribution(decomposition *d)
    : distribution(d),mpi_distribution(d),scalar_distribution(d),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
};

class mpi_cyclic_distribution : public mpi_distribution,public cyclic_distribution {
public:
  mpi_cyclic_distribution(decomposition *d,index_int l,index_int g)
    : distribution(d),mpi_distribution(d),cyclic_distribution(d,l,g),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  mpi_cyclic_distribution(decomposition *d,index_int l)
    : mpi_cyclic_distribution(d,l,-1) {};
};

class mpi_blockcyclic_distribution : public mpi_distribution,public blockcyclic_distribution {
public:
  mpi_blockcyclic_distribution(decomposition *d,index_int bs,index_int nb,index_int g)
    : distribution(d),mpi_distribution(d),blockcyclic_distribution(d,bs,nb,g),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  mpi_blockcyclic_distribution(decomposition *d,index_int bs,index_int nb)
    : mpi_blockcyclic_distribution(d,bs,nb,-1) {};
};

class mpi_replicated_distribution : public mpi_distribution,public replicated_distribution {
public:
  mpi_replicated_distribution(decomposition *d,int ortho,index_int l)
    : distribution(d),mpi_distribution(d),replicated_distribution(d,ortho,l),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  mpi_replicated_distribution(decomposition *d,index_int l)
    : mpi_replicated_distribution(d,1,l) {};
  mpi_replicated_distribution(decomposition *d)
    : mpi_replicated_distribution(d,1) {};
};

class mpi_gathered_distribution : public mpi_distribution,public gathered_distribution {
public:
  //! Create a gathered distribution with s per processor and k orthogonal.
  mpi_gathered_distribution(decomposition *d,int k,index_int s)
    : distribution(d),mpi_distribution(d),gathered_distribution(d,k,s),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
  //! Creating from a single integer parameter assumes k=1
  mpi_gathered_distribution(decomposition *d,index_int l)
    : mpi_gathered_distribution(d,1,l) {};
  //! Creating without integer parameters corresponds to one element per proc
  mpi_gathered_distribution(decomposition *d) : mpi_gathered_distribution(d,1,1) {};
};

class mpi_binned_distribution : public mpi_distribution,public binned_distribution {
public:
  mpi_binned_distribution(decomposition *d,object *o)
    : distribution(d),mpi_distribution(d),binned_distribution(d,o),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) { memoize(); };
};

/****
 **** Sparse matrix / index pattern
 ****/

class mpi_sparse_matrix : public sparse_matrix {
protected:
  processor_coordinate mycoord; std::shared_ptr<multi_indexstruct> mystruct;
  index_int my_first,my_last,localsize; //!< these need to be domain_coordinate
  index_int maxrowlength{-1};
  mpi_distribution *matrix_distribution{nullptr};
  std::vector< std::vector<sparse_element> > elements;
  MPI_Win create_win;
public:
  //! \todo can we do mycoord without the cast?
  mpi_sparse_matrix( distribution *d )
    : sparse_matrix(d->get_dimension_structure(0).get(),d->procid()) {
    matrix_distribution = dynamic_cast<mpi_distribution*>(d);
    if (matrix_distribution==nullptr)
      throw(std::string("Could not upcast to mpi distribution"));
    // identify the current process & structure
    mycoord
      = matrix_distribution->proc_coord( *dynamic_cast<decomposition*>(matrix_distribution) );
    mystruct = matrix_distribution->get_processor_structure(mycoord);
    if (mystruct->get_dimensionality()!=1)
      throw(fmt::format("Matrix dimensionality needs to be 1"));
    //
    my_first = mystruct->first_index_r()[0];
    my_last = mystruct->last_index_r()[0];
    localsize = mystruct->volume();
    globalsize = matrix_distribution->global_volume(); // VLE this presumes square
  };
  mpi_sparse_matrix( distribution *d,index_int g )
    : mpi_sparse_matrix(d) {
    maxrowlength = g;
    auto element_storage = new sparse_element[localsize*maxrowlength];
    // elements = std::vector< std::vector<sparse_element> >(localsize);
    // for (int row=0; row<localsize; row++)
    //   elements.at(row) = std::vector<sparse_element>(ocalsize)
    MPI_Aint winsize = localsize*maxrowlength;
    MPI_Win_create( element_storage,winsize,sizeof(sparse_element),
		    MPI_INFO_NULL,MPI_COMM_WORLD,&create_win);
  };
  virtual void add_element( index_int i,index_int j,double v ) override;
  virtual void add_element( index_int i,index_int j ) override {
    sparse_matrix::add_element(i,j,1.);
  };
  std::shared_ptr<indexstruct> all_columns() override {
    throw("can not ask all columns in MPI\n"); };
  virtual sparse_matrix *transpose() const override {
    throw(std::string("MPI matrix transpose not implemented"));
  };
};

class mpi_upperbidiagonal_matrix : public mpi_sparse_matrix {
public:
  mpi_upperbidiagonal_matrix( distribution *dist, double d,double r )
    : mpi_sparse_matrix(dist) {
    // int mytid = dist->procid();
    // index_int g = dist->global_volume(),
    //   my_first = dist->first_index(mytid), my_last = dist->last_index(mytid);
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row+1; if (col<globalsize) add_element(row,col,r);
    }
  }
};

class mpi_lowerbidiagonal_matrix : public mpi_sparse_matrix {
public:
  mpi_lowerbidiagonal_matrix( distribution *dist, double l,double d )
    : mpi_sparse_matrix(dist) {
    // int mytid = dist->mytid();
    // index_int g = dist->global_volume(),
    //   my_first = dist->first_index(mytid), my_last = dist->last_index(mytid);
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
    }
  }
};

class mpi_toeplitz3_matrix : public mpi_sparse_matrix {
public:
  mpi_toeplitz3_matrix( distribution *dist, double l,double d,double r )
    : mpi_sparse_matrix(dist) {
    // int mytid = dist->mytid();
    // index_int g = dist->global_volume(),
    //   my_first = dist->first_index(mytid), my_last = dist->last_index(mytid);
    for (index_int row=my_first; row<=my_last; row++) {
      index_int col;
      col = row;   add_element(row,col,d);
      col = row-1; if (col>=0) add_element(row,col,l);
      col = row+1; if (col<globalsize) add_element(row,col,r);
    }
  }
};

/****
 **** Object
 ****/

class mpi_object : public object {
private:
public:
  //! Create an object from a distribution, locally allocating the data.
  //! Use this as initializer only if it's the right allocation strategy.f
  mpi_object( distribution *d )
    : object(d),
      entity(entity_cookie::OBJECT) {
    set_data_handling(); 
    // if objects can embed in a halo, we'll allocate later; otherwise now.
    if (!d->get_can_embed_in_beta())
      mpi_allocate();
  };
  //! Create an object from a distribution, using a pointer to external data.
  //! \todo should be double* ?
  mpi_object( distribution *d,double *dat )
    : object(d),
      entity(entity_cookie::OBJECT) {
    set_data_handling(); 
    if (local_ndomains()>1)
      throw(std::string("Can not create from other object on >1 domain"));
    register_data_on_domain_number(0,dat,0);
    data_status = object_data_status::USER;    
  };
  //! Create an object from the data of another object
  mpi_object( distribution *d,std::shared_ptr<object> x)
    : object(d),
      entity(entity_cookie::OBJECT) {
    set_data_handling(); 
    if (local_ndomains()>1)
      throw(std::string("Can not create from other object on >1 domain"));
    if (!x->has_data_status_allocated()) {
      printf("warning: allocating vector so that it can be inherited\n");
      x->allocate(); }
    try { register_data_on_domain_number(0,x->get_raw_data(),0);
    } catch (std::string c) {
      fmt::print("Error <<{}>> in constructor from <<{}>>",c,x->get_name());
      throw(std::string("Could not create object from object"));
    }
    data_status = object_data_status::REUSED;
  };
  // this is just to make static casts possible
  void poly_object() { printf("this serves no purpose\n"); };

  /*
   * Data
   */
protected:
  double *mpi_get_data(processor_coordinate &p);
  void mpi_allocate();
  //! Install all mpi-specific data routines
  void set_data_handling() {
    get_data_p = [this] (processor_coordinate &p) -> double* {
      return mpi_get_data(p); };
    allocate = [this] (void) -> void { mpi_allocate(); };
  };

public:
  //! MPI has no mask shift, because a processor is either there or not
  virtual domain_coordinate mask_shift( processor_coordinate &p ) {
  //  virtual index_int mask_shift(int np) {
    if (!lives_on(p))
      throw(fmt::format("Should not ask mask shift for {} on {}",p.as_string(),get_name()));
    return 0;
  };
};

/****
 **** Message
 ****/

class mpi_message : public message,virtual public entity {
public:
  mpi_message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> e,std::shared_ptr<multi_indexstruct> g)
    : message(d,snd,rcv,e,g),entity(entity_cookie::MESSAGE) {};
  mpi_message(decomposition *d,processor_coordinate &snd,processor_coordinate &rcv,
	      std::shared_ptr<multi_indexstruct> g)
    : message(d,snd,rcv,g),entity(entity_cookie::MESSAGE) {};
public: // figure out a way to access this
  MPI_Datatype embed_type{-1}; index_int lb{-1},extent{-1};
public:
  void compute_src_index() override; void compute_tar_index() override;
};

// utility routines for mpi messages
int mpi_message_buffer_length(int dim);

/****
 **** Kernel
 ****/

/*!
  The MPI kernel class exists to provide a few MPI-specific allocators.

  - the task creator for a domain
  - the beta distribution
 */
class mpi_kernel : virtual public kernel {
protected:
#ifdef VT
  int vt_kernel_class;
#endif
private:
public:
  mpi_kernel() : kernel(),entity(entity_cookie::KERNEL) {};
  mpi_kernel( std::shared_ptr<object> out ) : kernel(out),entity(entity_cookie::KERNEL) {};
  mpi_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out ) : kernel(in,out),entity(entity_cookie::KERNEL) {};
  //! This copy constructur is used in task creation. 
  mpi_kernel( const kernel& k ) : kernel(k),entity(entity_cookie::KERNEL) {};
  std::shared_ptr<task> make_task_for_domain(processor_coordinate&);
};

/*!
  An origin kernel only has output data.
  After task queue optimization its tasks may receive post and xpct messages.

  By default we set the local function to no-op, this means that we can
  set the vector elements externally.
 */
class mpi_origin_kernel : public mpi_kernel, public origin_kernel {
public:
  mpi_origin_kernel( std::shared_ptr<object> out,std::string name=std::string("mpi-origin kernel") )
    : kernel(out),mpi_kernel(out),origin_kernel(out,name),entity(entity_cookie::KERNEL) {};
};

/****
 **** Task
 ****/

class mpi_request : public request {
protected:
  MPI_Request rquest;
public:
  mpi_request(message *m,MPI_Request r) : request(m,request_protocol::MPI) { rquest = r; };
  MPI_Request get_mpi_request() { return rquest; };
  void poly() { return; };
};

//! \todo reinstate that status pushback?
class mpi_request_vector : public request_vector {
protected:
  std::vector<MPI_Status> statuses;
public:
  mpi_request_vector() : request_vector() { statuses.reserve(1000); };
  virtual void add_request(request *r) override {
    request_vector::add_request(r);
    //statuses.push_back( (MPI_Status)NULL );
  };
  void wait() { int s = size(); auto reqs = new MPI_Request[s];    
    for (int i=0; i<s; i++) {
      if (requests[i]->protocol!=request_protocol::MPI)
      	throw(std::string("transmutated request?\n"));
      mpi_request *mpi_req = dynamic_cast<mpi_request*>(requests[i]);
      if (mpi_req==nullptr)
	throw(std::string("Could not upcast to mpi request"));
      reqs[i] = mpi_req->get_mpi_request();
      message *msg = mpi_req->get_message();
      // fmt::print("{} Message {} -> {} indexset: {} == {}\n",
      // 		 msg->sendrecv_type_as_string(),
      // 		 msg->get_sender()->as_string(),msg->get_receiver()->as_string(),
      // 		 msg->get_global_struct()->as_string(),
      // 		 msg->get_local_struct()->as_string());
    }
    MPI_Waitall(s,reqs,MPI_STATUSES_IGNORE); // ,statuses.data());
    set_completed();
  };
};

class mpi_task : public task,public mpi_kernel {
private:
protected:
public:
  void make_requests_vector() { requests = new mpi_request_vector(); };
  void delete_requests_vector() { delete requests; };
  //! Make a task on a surrounding kernel.
  mpi_task(processor_coordinate &d,kernel *k)
    : kernel(*k),task(d,k),mpi_kernel(*k),entity(entity_cookie::TASK) {
    make_requests_vector(); set_factories(); };
  // make an origin task for standalone testing
  mpi_task(processor_coordinate &d,std::shared_ptr<object> out)
    : task(d,new mpi_kernel(out)),entity(entity_cookie::TASK) {
    make_requests_vector(); set_factories(); };
  // make a compute task for standalone testing
  mpi_task(processor_coordinate &d,std::shared_ptr<object> in,std::shared_ptr<object> out)
    : mpi_task(d,out) { add_in_object(in); };

  void set_factories() {
    message_from_buffer = [this] (int step,std::string &buf) -> message* {
      return mpi_message_from_buffer(this->shared_from_this(),step,buf);
    };
  };
  virtual void declare_dependence_on_task( task_id *id ); // pure virtual

  // pure virtual synchronization functions
  void acceptReadyToSend( std::vector<message*> &msgs,request_vector* );
  void notifyReadyToSend( std::vector<message*> &msgs,request_vector* );
  request *notifyReadyToSendMsg( message* msg );

  virtual void derive_send_messages() override ;
  void make_infrastructure_for_sending( int n );

  /* Stuff */
  void print();
};

class mpi_origin_task : public mpi_task {
public:
  mpi_origin_task( processor_coordinate &d,std::shared_ptr<object> out )
    : mpi_task(d,out),entity(entity_cookie::TASK) {
    set_type( kernel_type::ORIGIN ); set_name("origin mpi task");
  };
};

/****
 **** Queue
 ****/

/*!
  An MPI task queue is a processor-specific subset of the global queue,
  which we never assemble. The magic is done in routine
  \ref mpi_task::declare_dependence_on_task to make this local task queue
  conform to the global one.
*/
class mpi_algorithm : public algorithm { //,public mpi_communicator {
private:
public:
  mpi_algorithm(decomposition *d)
    : algorithm(d),
      entity(entity_cookie::QUEUE) {
    type = algorithm_type::MPI; make_mpi_communicator(this);
  };
  //! MPI-only analysis: try to embed vectors in a halo.
  virtual void mode_analyze_dependencies() override { inherit_data_from_betas(); };
  std::string kernels_as_string() override {
    if (mytid()==0)
      return algorithm::kernels_as_string();
    else {
      auto s =  new std::string; return *s;
    }
  };
};

#endif
