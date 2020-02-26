/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** omp_base.cxx: Implementations of the OpenMP classes
 ****
 ****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <iostream>
using namespace std;

#include <omp.h>
#include "omp_base.h"
#include "imp_base.h"

/****
 **** Basics
 ****/

/*!
  Our OpenMP implementation does not have thread-local data yet, 
  so gathering will just be replication.
 */
std::vector<index_int> *omp_architecture::omp_gather(index_int contrib) {
  if (omp_in_parallel()) throw(std::string("Unsupported use case for gather"));
  int P = nprocs();
  auto gathered = new std::vector<index_int>; gathered->reserve(P);
  for (int i=0; i<P; i++)
    gathered->push_back(contrib);
  return gathered;
};

/*!
  OMP implementation of the mode-independent reduce-scatter.
  \todo this gets called somewhere deep down where the arch is wrong ????
*/
int omp_architecture::omp_reduce_scatter(int *senders,int root) {
  printf("omp reduce scatter is wrong\n");
  return senders[root];
};

std::string omp_architecture::as_string() {
  fmt::MemoryWriter w; w.write("OpenMP architecture on {} threads",nprocs());
  return w.str();
};

/*!
  Initialize environment and process commandline options.
  \todo think about these omp allreduce things.
 */
omp_environment::omp_environment(int argc,char **argv) : environment(argc,argv) {
  type = environment_type::OMP;
  arch = make_architecture();
  set_is_printing_environment();

  if (has_argument("help")) {
    print_options();
    abort();
  }
  arch->set_collective_strategy( iargument("collective",0) );

  if (has_argument("embed"))
    arch->set_can_embed_in_beta();

  ntasks_executed = 0; // lose this

  // default collectives in the environment
  allreduce = [] (index_int i) -> index_int { return i; };
  allreduce_d = [] (double i) -> double { return i; };
  allreduce_and = [] (int i) -> int { return i; };
};

architecture *omp_environment::make_architecture() {
  architecture *arch; int nt; int over = iargument("over",1);
#pragma omp parallel shared(nt)
#pragma omp master
  {
    // nt becomes the nprocs value
    nt = omp_get_num_threads(); nt *= over;
    arch = new omp_architecture(nt);
  }
  // for (int p=0; p<arch->nprocs(); p++)
  //   arch->add_domain(p);
  //printf("created omp environment with %d threads\n",nt);
  return arch;
};

//! This is largely identical to the MPI code.
omp_environment::~omp_environment() {
  if (has_argument("dot")) {
    kernels_to_dot_file();
    tasks_to_dot_file();
  }
};

void omp_environment::record_task_executed() {
  ntasks_executed++;
};

/*!
  Document omp-specific options.
*/
void omp_environment::print_options() {
  printf("OpenMP-specific options:\n");
  printf("  -embed : try embedding objects in halos\n");
  environment::print_options();
}

void omp_environment::print_stats() {
  double t_x=0.,tmax; int n_x=0;

  // find average execution time over multiple runs
  for (std::vector<double>::iterator t=execution_times.begin();
       t!=execution_times.end(); ++t) {
    t_x = t_x+ (*t); n_x++;
  } t_x /= n_x;

  tmax = t_x;
};

/****
 **** Decomposition
 ****/

//! A factory for making new distributions from this decomposition
void omp_decomposition::set_decomp_factory() {
  new_block_distribution = [this] (index_int g) -> distribution* {
    return new omp_block_distribution(this,g); };
  // distribution_from_structure = [this] (parallel_structure *s) {
  //   return new omp_distribution(this,s);
  // };
};

/****
 **** Distribution
 ****/

void make_omp_communicator(communicator *cator,int P) {
  cator->the_communicator_mode = communicator_mode::OMP;
  
  cator->nprocs =        [P] (void) -> int { return P; };
  cator->allreduce =     [] (index_int contrib) -> index_int { return contrib; };
  cator->allreduce_d =   [] (double contrib) -> double { return contrib; };
  cator->allreduce_and = [] (int contrib) -> int { return contrib; };
  cator->overgather =    [P] (index_int contrib,int over) -> std::vector<index_int>* {
    auto v = new std::vector<index_int>; v->reserve(over*P);
    for (int i=0; i<over*P; i++) v->push_back(contrib);
    return v; };
};

//! Basic constructor
omp_distribution::omp_distribution(decomposition *d) 
  : distribution(d),entity(entity_cookie::DISTRIBUTION) {
  make_omp_communicator(this,d->domains_volume());
  set_dist_factory(); set_numa(); set_operate_routines(); set_memo_routines();
  numa_structure_is_computed = 0;
  set_name("omp-distribution");
};

//! Constructor from parallel structure
omp_distribution::omp_distribution( parallel_structure *struc )
  : distribution(struc),entity(entity_cookie::DISTRIBUTION) {
  make_omp_communicator(this,this->domains_volume());
  set_dist_factory(); set_numa(); set_operate_routines(); set_memo_routines();
  try {
    memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in omp distribution from struct {}",
		      struc->as_string()));
  }
  // VLE this code does not appear in MPI. do we need it here?
  // for ( int is=0; is<struc->get_structures().size(); is++ ) {
  //   parallel_indexstruct *oldstruct = struc->get_dimension_structure(is);
  //   set_dimension_structure(is,oldstruct);
  // }
  // set_type( struc->get_type() );
};

//! Constructor from function
omp_distribution::omp_distribution( decomposition *d,index_int(*pf)(int,index_int),index_int nlocal )
  : omp_distribution(d) {
  get_dimension_structure(0)->create_from_function( pf,nlocal );
  memoize();
};

/*
 * NUMA
 */
//! Set a number of routines that specify the NUMA structure of an OpenMP distribution
void omp_distribution::set_numa() {
  //snippet ompnuma
  compute_numa_structure = [this] (distribution *d) -> void {
    try {
      auto numa_struct = d->get_enclosing_structure();
      d->set_numa_structure
      ( numa_struct,numa_struct->first_index_r(),numa_struct->volume() );
    } catch (std::string c) {
      throw(fmt::format("Trouble computing numa structure: {}",c));
    }
  };
  //snippet end
  //! The first index that this processor can see.
  numa_first_index = [this] (void) -> const domain_coordinate {
    return first_index_r( proc_coord(*this) );
  };
  //! Total data that this processor can see.
  numa_local_size = [this] (void) -> index_int {
    try {
      return global_volume();
    } catch (std::string c) {
      throw(fmt::format("Trouble omp numa local size: {}",c));
    }
  };
  location_of_first_index = [] (distribution &d,processor_coordinate &p) -> index_int {
    return omp_location_of_first_index(d,p); };
  location_of_last_index = [] (distribution &d,processor_coordinate &p) -> index_int {
    return omp_location_of_last_index(d,p); };
  //snippet end
  local_allocation = [this] (void) -> index_int { return global_allocation(); };
  //snippet ompvisibility
  //! A processor can see all of the address space
  get_visibility = [this] (processor_coordinate &p) -> std::shared_ptr<multi_indexstruct> {
    return this->get_enclosing_structure(); };
  //snippet end
};

void omp_distribution::set_memo_routines() {
  compute_global_first_index = [this] (distribution *dist) -> domain_coordinate* {
    try {
      auto pstruct = dist->get_processor_structure( dist->get_origin_processor() );
      auto first = pstruct->first_index_r();
      //fmt::print("Computed global first as {}\n",first->as_string());
      return new domain_coordinate( first );
    } catch (std::string c) {
      fmt::print("Error: {}\n",c);
      throw(std::string("Could not compute omp global first index"));
    }
  };
  compute_global_last_index = [this] (distribution *dist) -> domain_coordinate* {
    auto last_struct = dist->get_processor_structure( dist->get_farpoint_processor() ); 
    return new domain_coordinate( last_struct->last_index_r() );
  };
};

// //! \todo this can be made a base method if we have a factory for the omp_distribution
// distribution *omp_distribution::omp_dist_operate_base( std::shared_ptr<ioperator> op ) {
//   distribution *d = new omp_distribution(dynamic_cast<decomposition*>(this));
//   for (int id=0; id<get_dimensionality(); id++)    
//     d->set_dimension_structure( id,get_dimension_structure(id)->operate(op) );
//   d->set_orthogonal_dimension( get_orthogonal_dimension() );
//   return d; };

//! \todo this can be made a base method if we have a factory for the omp_distribution
distribution *omp_distribution::omp_dist_distr_union( distribution *other ) {
  distribution *d = new omp_distribution(dynamic_cast<decomposition*>(this));
  for (int id=0; id<get_dimensionality(); id++)
    d->set_dimension_structure
      ( id,get_dimension_structure(id)->struct_union(other->get_dimension_structure(id)) );
  d->set_type(distribution_type::GENERAL);
  d->set_orthogonal_dimension( get_orthogonal_dimension() );
  try {
    d->memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in unioned distr {}",d->get_name()));
  }
  return d; };

//! Set the factory routines for creating a new object from this distribution.
void omp_distribution::set_dist_factory() {
  // Factory for new distributions
  new_distribution_from_structure = [this] (parallel_structure *strct) -> distribution* {
    return new omp_distribution( /*dynamic_cast<decomposition*>(this),*/strct ); };
  // Factory for new scalar distributions
  new_scalar_distribution = [this] (void) -> distribution* {
    return new omp_scalar_distribution( dynamic_cast<decomposition*>(this) ); };
  new_object = [this] (distribution *d) -> std::shared_ptr<object>
    { return std::shared_ptr<object>( new omp_object(d) ); };
  new_object_from_data = [this] (double *d) -> std::shared_ptr<object>
    { return std::shared_ptr<object>( new omp_object(this,d) ); };
  // Factory for making mode-dependent kernels (this seems only for testing?)
  new_kernel_from_object = [] ( std::shared_ptr<object> out ) -> kernel*
    { return new omp_kernel(out); };
  // Factory for making mode-dependent kernels, for the imp_ops kernels.
  kernel_from_objects = [] ( std::shared_ptr<object> in,std::shared_ptr<object> out ) -> kernel*
    { return new omp_kernel(in,out); };
  // Set the message factory.
  new_message =
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	std::shared_ptr<multi_indexstruct> g) -> message* {
    return new omp_message(this,snd,rcv,g);
  };
  new_embed_message =
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	std::shared_ptr<multi_indexstruct> e,std::shared_ptr<multi_indexstruct> g) -> message* {
    return new omp_message(this,snd,rcv,e,g);
  };
};

/*! 
  Find the location of a processor structure in the allocated data.
  \todo not correct for masks
  \todo this could use a unit test. next one too
*/
index_int omp_location_of_first_index(distribution &d,processor_coordinate &pcoord) {
  int dim = d.get_same_dimensionality(pcoord.get_dimensionality());
  auto enc = d.get_enclosing_structure();
  domain_coordinate
    d_first = d.first_index_r(pcoord),
    enc_first = enc->first_index_r(),
    enc_last = enc->last_index_r();
  index_int loc = d_first.linear_location_in(enc); //(enc_first,enc_last);
  //fmt::print("{} first={}, location in {} is {}\n",pcoord->as_string(),d_first.as_string(),enc->as_string(),loc);
  return loc;
};

//! \todo this should be done through linear location in numa struct
index_int omp_location_of_last_index(distribution &d,processor_coordinate &pcoord) {
  return d.last_index_r(pcoord).linear_location_in( d.global_last_index() );
};

/****
 **** Object
 ****/

/*! Getting OMP data is handled by the data pointers.
  \todo get_domain_local_number should have a non-star variant
 */
double *omp_object::omp_get_data(processor_coordinate &p) {
  //  if (!lives_on(p)) throw(fmt::format("object does not live on {}",p->as_string()));
  //  int locdom = get_domain_local_number(p);
  for ( auto dom : get_domains() )
    if (p.equals(dom) /*->equals(p)*/ ) {
      int locdom = get_domain_local_number(dom);
      return get_data_pointer(locdom);
    }
  throw(fmt::format("Could not find data for <<{}>>",p.as_string()));
}

void omp_object::omp_allocate() {
  if (has_data_status_allocated()) return;
  auto domains = get_domains(); processor_coordinate d0 = domains.at(0);
  index_int
    s = global_allocation(),
    slocal = local_allocation_p(d0); // this incorporates masks
  if (get_type()==distribution_type::REPLICATED) {
    create_data(domains.size()*slocal,get_name());
    double *rawdat = get_numa_data_pointer().get(); bool init{true};
    for (int idom=0; idom<domains.size(); idom++) {
      auto dom = domains.at(idom);
      //for ( auto dom : domains ) {
      auto offset = idom * slocal; //location_of_first_index(*this,dom);
      fmt::print("{}: data offset={}\n",dom.as_string(),offset);
      register_data_on_domain(dom,rawdat+offset,0);
    }
  } else {
    create_data(domains.size(),get_name());
    double *rawdat = get_numa_data_pointer().get(); bool init{true};
    for ( auto dom : domains ) {
      if (init) {
	register_data_on_domain(dom,rawdat /* +location_of_first_index(*this,dom) */ ,s);
	init = false;
      } else
	register_data_on_domain(dom,rawdat /* +location_of_first_index(*this,dom) */ ,0);
    }
  }
};

/*!
  This routine is typically used to copy from an in object to a halo.
  In OpenMP the halo is the output vector; see \ref omp_task::acceptReadyToSend,
  so the indexing.....

  \todo deal with the case where global/local struct are not contiguous
*/
//snippet ompcopydata
void omp_object::copy_data_from( std::shared_ptr<object> in,message *smsg,message *rmsg ) {
  if (has_data_status_unallocated() || in->has_data_status_unallocated())
    throw(std::string("Objects should be allocated by now"));
  //  std::shared_ptr<object> out = this->shared_from_this();
  auto out = this;

  auto p = rmsg->get_receiver(), q = rmsg->get_sender();
  int dim = get_same_dimensionality( in->get_dimensionality() );
  double
    *src_data = in->get_data(q), *tar_data = out->get_data(p);
  auto src_struct = smsg->get_local_struct(), tar_struct = rmsg->get_local_struct(),
    src_gstruct = smsg->get_global_struct(), tar_gstruct = rmsg->get_global_struct();
  auto struct_size = tar_struct->local_size_r();

  int k = in->get_orthogonal_dimension();
  if (k>1)
    throw(std::string("copy data too hard with k>1"));

  domain_coordinate
    pfirst = tar_gstruct->first_index_r(), plast = tar_gstruct->last_index_r(),
    qfirst = src_gstruct->first_index_r(), qlast = src_gstruct->last_index_r();
  // fmt::print("Copy {}:{} -> {}:{}\n",
  // 	     q->as_string(),qfirst.as_string(),p->as_string(),pfirst.as_string());

  auto in_nstruct = in->get_numa_structure(),
    out_nstruct = out->get_numa_structure(),
    in_gstruct = in->get_global_structure(),
    out_gstruct = out->get_global_structure();
  domain_coordinate
    in_nsize = in_nstruct->local_size_r(), out_nsize = out_nstruct->local_size_r(),
    in_offsets = in_nstruct->first_index_r() - in_gstruct->first_index_r(),
    out_offsets = out_nstruct->first_index_r() - out_gstruct->first_index_r();

  if (dim==0) {
  } else if (dim==2) {
    if (src_struct->is_contiguous() && tar_struct->is_contiguous()) {
      int done = 0;
      for (index_int isrc=qfirst[0],itar=pfirst[0]; itar<=plast[0]; isrc++,itar++) {
	for (index_int jsrc=qfirst[1],jtar=pfirst[1]; jtar<=plast[1]; jsrc++,jtar++) {
	  index_int
	    Iout = INDEX2D(itar,jtar,out_offsets,out_nsize),
	    Iin = INDEX2D(isrc,jsrc,in_offsets,in_nsize);
	  if (!done) {
	    fmt::print("{}->{} copy data {} between {}->{}\n",
		       q.as_string(),p.as_string(),src_data[ Iin ],Iin,Iout);
	    done = 1; }
	  tar_data[ Iout ] = src_data[ Iin ];
	}
      }

    } else {
      throw(std::string("omp 2d copy requires cont-cont"));
    }
  } else if (dim==1) {
    auto local = rmsg->get_local_struct(), global = rmsg->get_global_struct();
    if (global->is_contiguous() && local->is_contiguous()) {

      for (index_int i=pfirst[0]; i<=plast[0]; i++) {
	tar_data[ INDEX1D(i,out_offsets,out_nsize) ] = 
	  src_data[ INDEX1D(i,in_offsets,in_nsize) ];
      }

    } else {
      index_int localsize = local->volume();
      index_int len = localsize*k;

      auto
	src_struct = global->get_component(0), tar_struct = local->get_component(0);
      index_int
        src0 = in->get_enclosing_structure()->linear_location_of(local),
        tar0 = this->get_enclosing_structure()->linear_location_of(global);
      if (src_struct->is_contiguous() && tar_struct->is_contiguous()) {
        throw(std::string("This should have bee done above"));
      } else if (tar_struct->is_contiguous()) {
        index_int itar=tar0; 
        for ( auto isrc : *src_struct )
          tar_data[itar++] = src_data[isrc];
      } else {
        for (index_int i=0; i<len; i++)
          tar_data[ tar_struct->get_ith_element(i) ] = src_data[ src_struct->get_ith_element(i) ];
      }
    }
  } else 
    throw(std::string("Can not omp copy in other than 1-d or 2-d"));
};
//snippet end

std::string omp_object::values_as_string() {
  fmt::MemoryWriter w; w.write("{}:",get_name());
  if (get_orthogonal_dimension()>1)
    throw(std::string("Can not handle k>1 for object values_as_string"));
  auto data = this->get_raw_data(); index_int s = global_volume();
  for (index_int i=0; i<s; i++)
    w.write(" {}:{}",i,data[i]);
  return w.str();
};

/****
 **** Request & request vector
 ****/

//snippet ompwait
/*!
  Wait for some requests, presumably for a task. 
  Each incoming requests carries an output object of another task.
  Outgoing requests are ignored. OpenMP does not send.
 */
void omp_request_vector::wait() {
  int outstanding = size();
  for ( ; outstanding>0 ; ) { // loop until all requests fullfilled
    //fmt::print("Requests outstanding: {}\n",outstanding);
    for ( auto r : requests ) {
      if (r->has_type(request_type::UNKNOWN))
	  throw(fmt::format("Can not wait for request of unknown type"));
      omp_request *req = dynamic_cast<omp_request*>(r);
      if (req==nullptr) throw(std::string("could not upcast to omp request"));
      if (req->has_type(request_type::OUTGOING)) {
	//fmt::print("Outgoing request, closing\n");
        req->closed = 1;
        outstanding--;
	continue;
      }
      if (!req->closed) {
        // unclosed requestion: make sure depency task is executed,
        auto t = req->tsk; std::shared_ptr<object> task_object;
	//fmt::print("unclosed request for task <<{}>>\n",t->as_string());
        if (!t->get_has_been_executed())
          try {
            t->execute();
          } catch (const char *c) { fmt::print("Error <<{}>>\n",c); 
            throw(fmt::format("Could not execute {} as dependency",t->get_name()));
	  }
	// then copy data
	omp_message *recv_msg = dynamic_cast<omp_message*>(req->msg);
	if (recv_msg==nullptr)
	  throw(std::string("Could not upcast to omp_message"));
	message *send_msg = recv_msg->send_msg;
	if (send_msg==nullptr)
	  throw(fmt::format("Can not find send message for <<{}>>",req->msg->as_string()));
        try {
          task_object = t->get_out_object();
        } catch (const char *c) { printf("Error <<%s>>\n",c); 
          throw( fmt::format("Could not get out data from dependent task {}",t->get_name())); }
        req->obj->copy_data_from( task_object,send_msg,recv_msg );
        req->closed = 1;
        outstanding--;
      }
    };
  }
};
//snippet end

/****
 **** Task
 ****/

/*!
  Build the send messages;
  \todo how far we unify this with MPI?
*/
void omp_task::derive_send_messages() {
  auto dom = get_domain();
  
  for ( auto msg : get_receive_messages() ) {
    auto sender_domain = msg->get_sender();
    std::shared_ptr<task> sender_task;
    try { sender_task = find_kernel_task_by_domain(sender_domain);
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> finding sender for msg <<{}>>",c,msg->as_string())); }
    dependency *dep = find_dependency_for_object_number(msg->get_in_object_number());
    auto in = dep->get_in_object(), out = dep->get_beta_object();
    auto send_struct = msg->get_global_struct();

    message *send_msg;
    send_msg = new omp_message
      (out.get() /* as decomposition */,sender_domain,dom, send_struct);
    send_msg->set_in_object(in); send_msg->set_out_object(out);

    try {
      send_msg->set_name( fmt::format("send-{}-obj:{}->{}",
	      msg->get_name(),msg->get_in_object_number(),msg->get_out_object_number()) );
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> setting name of msg <<{}>>",c,send_msg->as_string())); }


    try {
      send_msg->compute_src_index();
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> computing src index in <<{}>> from <<{}>>",
			c,send_msg->as_string(),send_msg->get_in_object()->get_name())); }
    try {
      sender_task->get_send_messages().push_back(send_msg);
      // add this send message to the receive message
      omp_message *recv_message = dynamic_cast<omp_message*>(msg); // might as well move this up
      if (recv_message==nullptr)
	throw(std::string("Could not upcast to omp_message"));
      recv_message->send_msg = send_msg;
    } catch (std::string c) {
      throw(fmt::format("Error <<{}>> in remaining send msg actions",c)); }
  }
};

/*!
  Execute a task by taking it as root and go down the predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.

  This routine is identical to the base routine, except for the omp directives.
*/
void omp_task::execute_as_root() {
  if (get_has_been_executed()) return;

  auto preds = get_predecessors();
  for (int it=0; it<preds.size(); it++) {
    auto tsk = preds.at(it);
#pragma omp task
    tsk->execute_as_root();
  }
#pragma omp taskwait
  { // not sure if this fragment does any good
    while (1) {
      int all = 1;
      for ( auto tsk : preds ) {
        all *= tsk->get_has_been_executed();
      }
      if (all==0) {
#pragma omp taskyield
        ;
      } else break;
    }
  }
  omp_set_lock(&dolock);
  if (!get_has_been_executed()) {
    execute();
    set_has_been_executed();
  }
  omp_unset_lock(&dolock);

};

//snippet ompaccept
/*!
  This is the routine that posts the request for other tasks to supply data.
  We implement this as just listing the predecessors tasks: we can not actually 
  execute them here, because this call happens potentially twice in task::execute.
  The actual execution happens in \ref omp_request::wait.
  \todo I have my doubts abou that set_data_is_filled call. this is the wrong place.
 */
void omp_task::acceptReadyToSend
    ( std::vector<message*> &msgs,request_vector *requests ) {
  if (msgs.size()==0) return;
  if (find_other_task_by_coordinates==nullptr)
    throw(fmt::format("{}: Need a task finding function",get_name()));
  for ( auto m : msgs ) {
    auto sender = m->get_sender(), receiver = m->get_receiver();
    auto halo = m->get_out_object();
    halo->set_data_is_filled(receiver);
    //halo->compute_enclosing_structure();
    int instep = m->get_in_object()->get_object_number();
    requests->add_request
      ( new omp_request( find_other_task_by_coordinates(instep,sender),m,halo ) );
  }
};
//snippet end

void omp_task::declare_dependence_on_task( task_id *id ) {
  int step = id->get_step(); auto domain = id->get_domain();
  try { add_predecessor( find_other_task_by_coordinates(step,domain) );
  } catch ( const char *c ) { throw(std::string("Could not find OMP queue predecessor")); };
};

/****
 **** Kernel
 ****/

/*! Construct the right kind of task for the base class
  method \ref kernel::split_to_tasks.
*/
std::shared_ptr<task> omp_kernel::make_task_for_domain(processor_coordinate &d) {
  return std::shared_ptr<task>( new omp_task(d,this) );
};

/****
 **** Queue
 ****/

//! Find a task number by step/domain coordinates. \todo why not return a task pointer?
int omp_algorithm::find_task( int s,processor_coordinate *d ) {
  int ret;
  for (int n=0; n<tasks.size(); n++) {
    if (tasks[n]->get_step()==s && tasks[n]->get_domain().equals(d)) {
      ret = n; goto exit;
    }
  }
  ret = -1;
 exit:
  //  printf("found step %d domain %d as %d\n",s,d,ret);
  return ret;
};

/*!
  Find all tasks, starting at `this' one, that only depend on
  non-synchronization tasks.
*/
void omp_task::check_local_executability() {
  task_local_executability exec_val = this->get_local_executability();
  if (exec_val==task_local_executability::INVALID) throw(std::string("Invalid executability"));
  else if (exec_val==task_local_executability::UNKNOWN) {
    if (this->get_is_synchronization_point()) {
      exec_val = task_local_executability::NO;
    } else {
      exec_val = task_local_executability::YES;
      auto preds = this->get_predecessors();
      for  ( auto p : preds ) { //(auto p=preds->begin(); p!=preds->end(); ++p) {
        omp_task *op = dynamic_cast<omp_task*>(p.get());
	if (op==nullptr)
	  throw(fmt::format("Could not upcast to omp task"));
        op->check_local_executability();
        task_local_executability opx = op->get_local_executability();
        if (opx==task_local_executability::NO) {
          exec_val = task_local_executability::NO; break;
        }
      }
    }
    this->set_can_execute_locally(exec_val);
  }
};

/*!
  The omp version of algorithm::execute inserts OpenMP directives to make 
  execution parallel. We also split the queue in tasks with and without
  synchronization points, but we'll reconsider that in the future.

  \todo can we move the directives into execute_tasks and use the base method?
  \todo reinstate timer
*/
void omp_algorithm::execute( int(*tasktest)(std::shared_ptr<task> t) ) {
  if (tasktest==nullptr)
    throw(fmt::format("Missing task test in queue::execute"));
  //  auto tstart = get_architecture()->synchronized_timer();
#pragma omp parallel
  {
#pragma omp single
    {
      execute_tasks(tasktest);
    }
#pragma omp barrier
  }
  //  auto tend = get_architecture()->synchronized_timer();
  //register_duration( tstart,tend );
};

/*!
  Execute all tasks in a queue. Since they may not be in the right order,
  we take each as root and go down their predecessor tree. Ultimately
  this uses the fact that executing an already executed task is a no-op.
*/
void omp_algorithm::execute_tasks( int(*tasktest)(std::shared_ptr<task> t) ) {
  if (tasktest==nullptr) throw(fmt::format("Missing task test"));
  for ( auto t : get_exit_tasks() ) {
#pragma omp task untied
    try {
      if ((*tasktest)(t))
        t->execute_as_root();
    } catch ( std::string c ) {
      fmt::print("Error <<{}>> for task <<{}>> execute\n",c,t->as_string());
      throw(std::string("Task queue execute failed"));
    }
  }
#pragma omp taskwait
};
