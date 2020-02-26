// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** product_base.h: Header file for the product derived class
 ****
 ****************************************************************/
#ifndef PRODUCT_BASE_H
#define PRODUCT_BASE_H 1

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
using namespace std;

#include <omp.h>
#include <mpi.h>

#include <imp_base.h>
#include <omp_base.h>
#include <mpi_base.h>

/*! \page product MPI/OMP product backend

  A product of MPI and OpenMP is the easiest way for hybrid programming.
  At first glance we construct MPI distributions and objects, but when
  we execute a task, we invoke an \ref omp_algorithm. Of course this
  task queue will be flat since it contains the tasks of just one kernel.

  The product backend is thus somewhat uninteresting; the main point
  is the aggregation of MPI messages, and the load balancing through
  possible oversubscription of the OMP distribution over the hardware threads.

  Regarding the implementation, most product classes are the same as MPI classes, only
  - the \ref product_distribution contains an \ref omp_distribution
  - the \ref product_task contains an \ref omp_algorithm for the local execution.

  Currently we have unit tests working for
  - unittest_distribution.cxx : block and replicated distributions,
    operations on distributions;
    test[2] investigates the structure of a product distributions
  - unittest_struct: testing the mpi messaging, structure of embedded algorithm,
    shift left

  The unit tests are broken for 
  - modulo operators. This is because MPI is broken there.

  \todo write a product shift kernel analogous to struct[36]
 */

/****
 **** Basics
 ****/
#include "product_environment.h"

/****
 **** Architecture
 ****/

/*!
  Only one object of this class is created, namely in the environment.
  Each object then has its private \ref product_architecture object,
  which has just a pointer to this unique architecture data object.
 */
class product_architecture : public mpi_architecture {
protected:
  //int mpi_ntids,mpi_mytid,nthreads;
public :
  product_architecture( int mytid,int ntids,int nthreads )
    : mpi_architecture(mytid,ntids),entity(entity_cookie::ARCHITECTURE) {
    type = architecture_type::ISLANDS;
    set_name("product-architecture");
    embedded_architecture = new omp_architecture(nthreads);
    embedded_architecture->set_protocol_is_embedded();
    embedded_architecture->set_name("omp-in-product-architecture");
  };
};

/*!
  A product decomposition is an mpi decomposition with embedded an omp decomposition.
 */
class product_decomposition : public mpi_decomposition {
public:
  //! Multi-d decomposition from explicit processor grid layout
  product_decomposition( architecture *arch,processor_coordinate *grid)
    : mpi_decomposition(arch,grid),//decomposition(arch,grid),
      entity(entity_cookie::DECOMPOSITION) {
    embedded_decomposition = new omp_decomposition( arch->get_embedded_architecture() );
  };
  product_decomposition( architecture *arch )
    : mpi_decomposition(arch),//decomposition(arch),
      entity(entity_cookie::DECOMPOSITION) {
    embedded_decomposition = new omp_decomposition( arch->get_embedded_architecture() );
  };
  virtual std::string as_string() override;
};

/****
 **** Distribution
 ****/

/*!
  A product distribution is an \ref mpi_distribution with an embedded
  \ref omp_distribution. The latter is not created in the base class
  but in the derived classes, so that its type is set correctly.
*/
class product_distribution : public mpi_distribution {
protected:
  omp_distribution *embedded_distribution{nullptr};
public:
  //! basic constructor: create mpi and openmp member distributions
  product_distribution(decomposition *d)
    : mpi_distribution(d),distribution(d),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    if (d->get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
  };
  //! Not technically a copy constructor.
  product_distribution( distribution *other)
    : mpi_distribution( //dynamic_cast<decomposition*>(other),
			dynamic_cast<parallel_structure*>(other) ),
      distribution( dynamic_cast<decomposition*>(other) ),
      //decomposition(other),
      entity(entity_cookie::DISTRIBUTION) {
    if (other->get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
    product_distribution *pother = dynamic_cast<product_distribution*>(other);
    if (pother==nullptr)
      throw(fmt::format("Could not upcast <<{}>> to product",other->as_string()));
    embedded_distribution = new omp_distribution( pother->get_omp_distribution() );
    set_dist_factory();
  };
  //! constructor from parallel_indexstruct
  product_distribution(/*decomposition *d,*/parallel_structure *struc)
    : mpi_distribution(struc),distribution(struc),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    //fmt::print("Product dist has type {}\n",type_as_string());
    if (struc->get_architecture_type()!=architecture_type::ISLANDS)
      throw(std::string("Is not a product architecture"));
    set_dist_factory();
  };

  void set_dist_factory();
  omp_distribution *get_omp_distribution() {
    if (embedded_distribution==nullptr) throw(std::string("null omp distro"));
    return embedded_distribution; };
  //! \todo this can only word if pouter==me; test for that.
  index_int embedded_volume( processor_coordinate *pouter,processor_coordinate *pinner ) {
    auto embedded = get_omp_distribution();
    return embedded->volume(pinner);
  };

  // // Factory for objects
  // virtual std::shared_ptr<object> new_object();
  // virtual std::shared_ptr<object> new_object( double* );
};

class product_block_distribution
  : public product_distribution,public block_distribution {
public:
  //! \todo can the embedded_distribution creation go into the basic constructor?
  product_block_distribution(decomposition *d,int o,index_int l,index_int g)
    : product_distribution(d),block_distribution(d,o,l,g),distribution(d),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    int procno = d->mytid();
    auto pcoord = d->coordinate_from_linear(procno);
    auto domain = get_processor_structure(pcoord);
    try {
      decomposition *omp_decomp = d->get_embedded_decomposition();
      parallel_structure *parallel = new parallel_structure(omp_decomp);
      parallel->create_from_indexstruct(domain); // this sets type to BLOCKED
      embedded_distribution = new omp_distribution(/*omp_decomp,*/parallel);
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> creating embedded for {}\n",c,pcoord.as_string()));
    } catch (...) {
      throw(fmt::format("Unknown error creating embedded for {}\n",pcoord.as_string()));
    }
    embedded_distribution->set_enclosing_structure( this->get_enclosing_structure() );
    /*
    try {
      embedded_distribution->set_enclosing_structure( this->get_enclosing_structure() );
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> setting embedded for {}\n",c,pcoord.as_string()));
    } catch (...) {
      throw(fmt::format("Unknown error setting embedded for {}\n",pcoord.as_string()));
    }
    */
  };
  product_block_distribution(decomposition *d,index_int l,index_int g)
    : product_block_distribution(d,1,l,g) {};
  product_block_distribution(decomposition *d,index_int g)
    : product_block_distribution(d,-1,g) {};
  //! Multi-d constructor takes an endpoint vector: array of global sizes
  product_block_distribution(decomposition *d,std::vector<index_int> endpoint)
    : product_distribution(d),block_distribution(d,endpoint),distribution(d),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    d->get_same_dimensionality(endpoint.size()); };
};

/*!
  A replicated scalar is the output of a all-reduction operation.
  The cleverness here is taken care of by 
  \ref parallel_indexstruct::create_from_replicated_local_size
  (called in the constructor)
  which puts the same indices on each processor.
*/
class product_replicated_distribution
  : public product_distribution,public replicated_distribution {
public:
  product_replicated_distribution(decomposition *d,index_int s)
    : product_distribution(d),replicated_distribution(d,s),distribution(d),
      //decomposition(d),
      entity(entity_cookie::DISTRIBUTION) {
    decomposition *omp_decomp = d->get_embedded_decomposition();
    parallel_structure *parallel = new parallel_structure(omp_decomp);
    parallel->create_from_replicated_local_size(s); // this sets type to REPLICATED
    embedded_distribution = new omp_distribution(/*omp_decomp,*/parallel);
  };
  //! Create without integer arguments corresponds to a replicated single element
  product_replicated_distribution(decomposition *d)
    : product_replicated_distribution(d,1) {};
};

/*!
  Product gathered distributions describe the gathered result of each omp thread
  on each mpi process having s elements. This translates to a \ref mpi_gathered_distribution
  of s*omp_nprocs elements for each mpi process. Each omp process then replicates
  this full storage of size s*product_nprocs.
 */
class product_gathered_distribution : public product_distribution,public gathered_distribution {
public:
  //! Create a gather of s elements, with k ortho
  product_gathered_distribution(decomposition *d,int k,index_int s)
    : product_distribution(d),
      gathered_distribution(d,k,s*d->embedded_nprocs()), // s elements for each omp proc
      //decomposition(d),
      distribution(d),entity(entity_cookie::DISTRIBUTION) {
    int procno = d->mytid();
    auto pcoord = d->coordinate_from_linear(procno);
    auto domain = get_processor_structure(pcoord);
    decomposition *omp_decomp = d->get_embedded_decomposition();
    parallel_structure *parallel = new parallel_structure(omp_decomp);
    parallel->create_from_replicated_local_size(s*d->product_nprocs());
    embedded_distribution =
      new omp_distribution(/*omp_decomp,*/parallel);
    /*
    embedded_distribution = new omp_replicated_distribution
      (d->get_embedded_decomposition(),s*d->product_nprocs());
    */
    embedded_distribution->set_enclosing_structure( this->get_enclosing_structure() );
  };
  //! Create with a single integer arguments corresponds to ortho=1
  product_gathered_distribution(decomposition *d,index_int s)
    : product_gathered_distribution(d,1,s) {};
  //! Create without integer arguments corresponds to one element per processor
  product_gathered_distribution(decomposition *d)
    : product_gathered_distribution(d,1,1) {};
};

/****
 **** Sigma object
 ****/

/****
 **** Object
 ****/

/*!
  We let the data of a product object be allocated by the MPI distribution
  component of the product distribution.
*/
class product_object : public mpi_object {
private:
protected:
public:
  void set_factory() {
  };
  product_object( distribution *d )
    : mpi_object(d),
      entity(entity_cookie::OBJECT) { set_factory(); };
  product_object( distribution *d, double *x )
    : mpi_object(d,x),
      entity(entity_cookie::OBJECT) { set_factory(); };
};

/****
 **** Message
 ****/

/****
 **** Kernel
 ****/

/*!
  A product kernel is largely an mpi kernel, except that the \ref kernel::localexecutefn
  has a fixed value to execute the embedded \ref omp_algorithm.
 */
class product_kernel : public mpi_kernel {
private:
public:
  product_kernel() : kernel(),mpi_kernel(),entity(entity_cookie::KERNEL) {};
  product_kernel( std::shared_ptr<object> in,std::shared_ptr<object> out )
    : mpi_kernel( in,out ),kernel(in,out),entity(entity_cookie::KERNEL) {};
  product_kernel( std::shared_ptr<object> out )
    : mpi_kernel( out ),kernel(out),entity(entity_cookie::KERNEL) {};
  std::shared_ptr<task> make_task_for_domain(processor_coordinate &dom);
};

class product_origin_kernel : public product_kernel, public origin_kernel {
public:
  product_origin_kernel( std::shared_ptr<object> out )
    : kernel(out),product_kernel(out),origin_kernel(out),entity(entity_cookie::KERNEL) {};
};

/****
 **** Task
 ****/

/*!
  A product task is an MPI task, except that it executes a complete \ref omp_algorithm
  as its local function. Other functions, in particular \ref allocate_halo_vector 
  are completely inherited from the MPI version.

  \todo try constructor delegating for the node queue & local stuff
 */
class product_task : public mpi_task {
private:
  std::shared_ptr<object> omp_inobject,omp_outobject;
protected:
public:
  product_task(processor_coordinate &d,kernel *k)
    : mpi_task(d,k),kernel(*k),entity(entity_cookie::TASK) {
    node_queue = std::shared_ptr<algorithm>
      ( new omp_algorithm( k->get_out_object()->get_embedded_decomposition() ) );
    node_queue->set_name(fmt::format("omp-queue-in-task:{}",this->get_name()));
    if (get_out_object()->get_split_execution()) {
      node_queue->set_sync_tests
	(
	 [] (std::shared_ptr<task> t) -> int {
	   return dynamic_cast<omp_task*>(t.get())->get_local_executability()
	     ==task_local_executability::YES; },
	 [] (std::shared_ptr<task> t) -> int {
	   return dynamic_cast<omp_task*>(t.get())->get_local_executability()
	     !=task_local_executability::YES; }
	 );
    }
    //localexecutectx = node_queue;
  };

  /*
   * Embedded OpenMP queue
   */
public:
  std::vector<std::shared_ptr<task>> &get_omp_tasks() {
    if (!node_queue->get_has_been_analyzed())
      throw("Can not get embedded tasks until analyzed\n");
    return node_queue->get_tasks(); };
  // local execute calls \ref algorithm::execute on the embedded queue
  virtual void local_execute
      (std::vector<std::shared_ptr<object>>&,std::shared_ptr<object>,void*,
       int(*)(std::shared_ptr<task>)) override;
  // local execute calls \ref algorithm::execute on the embedded queue
  virtual void local_execute
      (std::vector<std::shared_ptr<object>> &ins,std::shared_ptr<object> out,void *ctx)
    override {
    local_execute(ins,out,ctx,&task::task_test_true);
  };

  // See product_task::local_analysis
  std::shared_ptr<object> omp_object_from_mpi( std::shared_ptr<object> mpi_obj );
  // Used in omp_object_from_mpi
  distribution *omp_distribution_from_mpi( distribution *mpi_distr,processor_coordinate &mytid );

  //! Non-trivial override of the (no-op) virtual function
  virtual void local_analysis() override;

  /*
   * Contexts for embedded tasks
   */
protected:
  std::function
  < void(kernel_function_types) > tasklocalexecutefn;
  void *tasklocalexecutectx{nullptr};
public:
  /*! Override the standard behaviour, because we already have a local function;
    instead we store this function to give as local function to the omp tasks */
  virtual void set_localexecutefn
  ( std::function< void(kernel_function_types) > f )
    override { tasklocalexecutefn = f; };
  /*! Similarly we store this context to give to the omp tasks */
  virtual void set_localexecutectx( void *ctx ) override { tasklocalexecutectx = ctx; };
};

/****
 **** Queue
 ****/

class product_algorithm : public mpi_algorithm {
private:
public:
  product_algorithm(decomposition *d) : mpi_algorithm(d),entity(entity_cookie::QUEUE) {};
};

#endif
