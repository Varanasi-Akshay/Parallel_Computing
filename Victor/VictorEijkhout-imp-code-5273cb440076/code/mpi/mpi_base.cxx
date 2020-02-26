/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** mpi_base.cxx: Implementations of the MPI derived classes
 ****
 ****************************************************************/

#include <stdarg.h>
#include <unistd.h> // just for sync
#include <iostream>

#include <mpi.h>
#include "mpi_base.h"
#include "imp_base.h"

/****
 **** Basics
 ****/

index_int mpi_allreduce(index_int contrib,MPI_Comm comm) {
  index_int result;
  //  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_INDEX_INT,MPI_SUM,comm);
  return result;
};
double mpi_allreduce_d(double contrib,MPI_Comm comm) {
  int ntids; double result;
  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_DOUBLE,MPI_SUM,comm);
  return result;
};
int mpi_allreduce_and(int contrib,MPI_Comm comm) {
  int ntids, result;
  MPI_Comm_size(comm,&ntids);
  MPI_Allreduce(&contrib,&result,1,MPI_INT,MPI_PROD,comm);
  return result;
};

void mpi_gather32(int contrib,std::vector<int> &gathered,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  if (gathered.size()<ntids)
    throw(fmt::format("gather32 of {} elements into vector of size {}",ntids,gathered.size()));
  int P = ntids; int sendbuf = contrib;
  //  auto gathered = new std::vector<int>; gathered->reserve(P);
  // for (int i=0; i<P; i++)
  //   gathered->push_back(0);
  MPI_Allgather(&sendbuf,1,MPI_INT,gathered.data(),1,MPI_INT,comm);
  //  return gathered;
};

void mpi_gather64(index_int contrib,std::vector<index_int> &gathered,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids; index_int sendbuf = contrib;
  if (gathered.size()<P)
    throw(fmt::format("Vector too small ({}) for gather on {} procs",gathered.size(),P));
  MPI_Allgather(&sendbuf,1,MPI_INDEX_INT,gathered.data(),1,MPI_INDEX_INT,comm);
  // fmt::MemoryWriter w; w.write("gather64:");
  // for (int i=0; i<P; i++) w.write(" {}",gathered->at(i));
  // fmt::print("{}\n",w.str());
};

std::vector<index_int> *mpi_overgather(index_int contrib,int over,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids*over;
  index_int sendbuf[over]; for (int i=0; i<over; i++) sendbuf[i] = contrib;
  auto gathered = new std::vector<index_int>; gathered->reserve(P);
  for (int i=0; i<P; i++)
    gathered->push_back(0);
  MPI_Allgather(sendbuf,over,MPI_INDEX_INT,gathered->data(),over,MPI_INDEX_INT,comm);
  return gathered;
};

//! The root is a no-op for MPI, but see OpenMP
int mpi_reduce_scatter(int *senders,int root,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int P = ntids;
  int *recvcounts = new int[P], nsends;
  for (int i=0; i<P; ++i) recvcounts[i] = 1;
  MPI_Reduce_scatter(senders,&nsends,recvcounts,MPI_INT,MPI_SUM,comm);
  delete recvcounts;
  return nsends;
};

std::vector<index_int> mpi_reduce_max(std::vector<index_int> local_values,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int nvalues = local_values.size();
  std::vector<index_int> global_values(nvalues);
  for (int n=0; n<nvalues; n++) global_values[n] = -1;
  MPI_Allreduce
    (local_values.data(),global_values.data(),nvalues,MPI_INDEX_INT,MPI_MAX,comm);
  return global_values;;
};

std::vector<index_int> mpi_reduce_min(std::vector<index_int> local_values,MPI_Comm comm) {
  int ntids;
  MPI_Comm_size(comm,&ntids);
  int nvalues = local_values.size();
  std::vector<index_int> global_values(nvalues);
  for (int n=0; n<nvalues; n++) global_values[n] = -1;
  MPI_Allreduce
    (local_values.data(),global_values.data(),nvalues,MPI_INDEX_INT,MPI_MIN,comm);
  return global_values;;
};

std::string mpi_architecture::as_string() {
  return fmt::format("MPI architecture of {} procs",nprocs());
};

#ifdef VT
#include "VT.h"
#include "imp_static_vars.h"
void vt_register_kernels() {
  //  VT_classdef("copy kernel",&vt_copy_kernel); // (const char * classname, int * classhandle)
};
#endif

/*!
  Document mpi-specific options.
*/
void mpi_environment::print_options() {
  print_application_options();
  printf("MPI-specific options:\n");
  printf("  -embed : try embedding objects in halos\n");
  printf("  -overlap : post isends and irecvs early\n");
  printf("  -random_source : random resolution of ambiguous origins");
  printf("  -ram : use one-sided routines");
  printf("\n");
  environment::print_options();
}

/*!
  An MPI environment has all the components of a base environment, plus
  - store a communicator and a task id.
  - disable printing for task id not zero

  \todo can we do the mpi_init in the environment constructor?
 */
mpi_environment::mpi_environment(int argc,char **argv) : environment(argc,argv) {
  type = environment_type::MPI;
  MPI_Init(&nargs,&the_args);
  comm = MPI_COMM_WORLD;
  MPI_Comm_set_errhandler(comm,MPI_ERRORS_RETURN);
  
  delete_environment = [this] () -> void { mpi_delete_environment(); };
#ifdef VT
  vt_register_kernels();
  VT_initialize(&nargs,&the_args);
#endif
  arch = make_architecture();
  int mytid = arch->mytid();

  set_is_printing_environment ( mytid==0 );
  if (has_argument("help") || has_argument("h")) {
    if (get_is_printing_environment())
      print_options();
    MPI_Abort(comm,0);
  }
  if (mytid==0 && has_argument("trace")) {
    printf("set trace print\n");
    arch->set_stdio_print();
  }

  // default collectives in the environment
  allreduce =      [this] (index_int i) -> index_int { return mpi_allreduce(i,comm); };
  allreduce_d =    [this] (double i) -> double { return mpi_allreduce_d(i,comm); };
  allreduce_and =  [this] (int i) -> int { return mpi_allreduce_and(i,comm); };

  gather32 =         [this] (int contrib,std::vector<int> &gathered) -> void {
    mpi_gather32(contrib,gathered,comm); };
  gather64 =         [this] (index_int contrib,std::vector<index_int> &gathered) -> void {
    mpi_gather64(contrib,gathered,comm); };
  overgather =     [this] (index_int contrib,int over) -> std::vector<index_int>* {
    return mpi_overgather(contrib,over,comm); };
  reduce_scatter = [this] (int *senders,int root) -> int {
    int procid; MPI_Comm_rank(comm,&procid);
    return mpi_reduce_scatter(senders,procid,comm); };

  // we can not rely on commandline argument on other than proc 0
  MPI_Bcast(&debug_level,1,MPI_INT,0,comm);
  arch->set_collective_strategy( get_collective_strategy() );
  {
    int e = has_argument("embed");
    MPI_Bcast(&e,1,MPI_INT,0,comm);
    if (e)
      arch->set_can_embed_in_beta();
    int v = has_argument("overlap");
    MPI_Bcast(&v,1,MPI_INT,0,comm);
    if (v)
      arch->set_can_message_overlap();
    int r = has_argument("random_source");
    MPI_Bcast(&r,1,MPI_INT,0,comm);
    if (r)
      arch->set_random_sourcing();
    int o = has_argument("rma");
    MPI_Bcast(&o,1,MPI_INT,0,comm);
    if (o)
      arch->set_use_rma();
  }

  // mode-specific summary
  mode_summarize_entities =
    [this] (void) -> result_tuple* { return mpi_summarize_entities(); };
  result_tuple *mpi_summarize_entities();
};

//! \todo make the over quantity variable
architecture *mpi_environment::make_architecture() {
  //  comm = MPI_COMM_WORLD;
  int mytid,ntids;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
  mpi_architecture *a;
  {
    int over = iargument("over",1);
    a = new mpi_architecture(mytid,ntids,over);
    a->set_mpi_comm( comm ); // comm world is already used in make_arch.....
  }
  return a;
};

//! See also the base destructor for trace output.
mpi_environment::~mpi_environment() {
  if (has_argument("dot") && get_architecture()->mytid()==0)
    kernels_to_dot_file();
  if (has_argument("dot"))
    tasks_to_dot_file();
};

//! This destructor routine needs to be called after the base destructor.
void mpi_environment::mpi_delete_environment() {
  int procid;
  MPI_Comm_rank(comm,&procid);
  // if (procid==0)
  //   printf("MPI finalize\n");
  MPI_Finalize();
#ifdef VT
  VT_finalize();
#endif
};

/*!
  Sum all the space, entity counts were global to begin with.
  \todo this does not seem to be called, at least in the templates?!
  \todo task count is kernel count?
*/
result_tuple *mpi_environment::mpi_summarize_entities() {
  auto mpi_results = new result_tuple;
  auto seq_results = local_summarize_entities();
  architecture *arch = get_architecture();
  std::get<RESULT_OBJECT>(*mpi_results)       = std::get<RESULT_OBJECT>(*seq_results);
  std::get<RESULT_KERNEL>(*mpi_results)       = std::get<RESULT_KERNEL>(*seq_results);
  std::get<RESULT_TASK>(*mpi_results)
    = allreduce( std::get<RESULT_TASK>(*seq_results) );
  std::get<RESULT_DISTRIBUTION>(*mpi_results) = std::get<RESULT_DISTRIBUTION>(*seq_results);
  std::get<RESULT_ALLOCATED>(*mpi_results)
    = allreduce( std::get<RESULT_ALLOCATED>(*seq_results) );
  std::get<RESULT_DURATION>(*mpi_results)     = std::get<RESULT_DURATION>(*seq_results);
  std::get<RESULT_ANALYSIS>(*mpi_results)     = std::get<RESULT_ANALYSIS>(*seq_results);
  {
    int my_nmessages = std::get<RESULT_MESSAGE>(*seq_results), all_nmessages;
    all_nmessages = allreduce( my_nmessages );
    std::get<RESULT_MESSAGE>(*mpi_results) = all_nmessages;
  }
  std::get<RESULT_WORDSENT>(*mpi_results)
    = allreduce_d( std::get<RESULT_WORDSENT>(*seq_results) );
  std::get<RESULT_FLOPS>(*mpi_results)
    = allreduce_d( std::get<RESULT_FLOPS>(*seq_results) );

  return mpi_results;
};

//! Open or append the tasks dot file.
void mpi_environment::tasks_to_dot_file() {
  int mytid = get_architecture()->mytid(), ntids = get_architecture()->nprocs();
  FILE *dotfile; std::string s;

  if (mytid>0) {
    int msgi;
    MPI_Recv(&msgi,1,MPI_INTEGER,mytid-1,17,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }

  const char *fname = fmt::format("{}-tasks.dot",get_name()).data();
  if (mytid==0) {
    dotfile = fopen(fname,"w");
    fprintf(dotfile,"digraph G {\n");
  } else
    dotfile = fopen(fname,"a");

  printf("dot for %d\n",mytid);
  s = tasks_as_dot_string();
  fprintf(dotfile,"/* ==== data for proc %d ==== */\n",mytid);
  fprintf(dotfile,"%s\n",s.data());
  fprintf(dotfile,"/* .... end proc %d .... */\n",mytid);

  if (mytid==ntids-1) 
    fprintf(dotfile,"}\n");

  fclose(dotfile);
  fflush(dotfile);
  sync();
  if (mytid<ntids-1) {
    int msgi = 1;
    MPI_Ssend(&msgi,1,MPI_INTEGER,mytid+1,17,MPI_COMM_WORLD);
  }
  // just to make sure that no one deletes our communicator.
  MPI_Barrier(MPI_COMM_WORLD);
};

/*!
  For MPI we let the root print out everything
*/
void mpi_environment::print_all(std::string s) {
  MPI_Comm comm; int ntids,mytid,maxlen;
  get_comm( (void*)&comm );
  MPI_Comm_rank(comm,&mytid);
  MPI_Comm_size(comm,&ntids);
  MPI_Request req;
  int siz = s.size();
  MPI_Reduce( (void*)&siz,&maxlen,1,MPI_INTEGER,MPI_MAX,0,comm);
  { // everyone, including zero, send to zero
    int siz = s.size();
    MPI_Isend( (void*)s.data(),siz,MPI_CHAR, 0,0,comm,&req);
  }
  if (mytid==0) {
    char *buffer = new char[maxlen+2];
    for (int id=0; id<ntids; id++) {
      MPI_Status stat; int n;
      MPI_Recv( buffer,maxlen,MPI_CHAR, id,0,comm,&stat);
      MPI_Get_count(&stat,MPI_CHAR,&n); buffer[n] = '\n'; buffer[n+1] = 0;
      print_to_file(id,buffer);
    }
  }
  MPI_Wait(&req,MPI_STATUS_IGNORE);
};

/****
 **** Decomposition
 ****/

/*!
  A factory for making new distributions from this decomposition
*/
void mpi_decomposition::set_decomp_factory() {
  new_block_distribution = [this] (index_int g) -> distribution* {
    return new mpi_block_distribution(this,g);
  };
};

/****
 **** Distribution
 ****/

void make_mpi_communicator(communicator *cator) {
  cator->the_communicator_mode = communicator_mode::MPI;

  {
    MPI_Comm *mpicom = new MPI_Comm; *mpicom = MPI_COMM_WORLD;
    cator->communicator_context = (void*)mpicom;
  }

  MPI_Comm comm = MPI_COMM_WORLD;
  cator->procid =
    [comm] (void) -> int { int tid; MPI_Comm_rank(comm,&tid); return tid; };
  cator->proc_coord =
    [comm] (decomposition &d) -> processor_coordinate
    { int tid; MPI_Comm_rank(comm,&tid); return processor_coordinate(tid,d); };
  cator->proc_coord_rv =
    [comm] (decomposition &&d) -> processor_coordinate
    { int tid; MPI_Comm_rank(comm,&tid); return processor_coordinate(tid,d); };
  cator->nprocs = [comm] (void) -> int { int np; MPI_Comm_size(comm,&np); return np; };

  cator->allreduce =
    [comm] (index_int contrib) -> index_int { return mpi_allreduce(contrib,comm); };
  cator->allreduce_d =
    [comm] (double contrib) -> double { return mpi_allreduce_d(contrib,comm); };
  cator->allreduce_and =
    [comm] (int contrib) -> int { return mpi_allreduce_and(contrib,comm); };
  cator->gather32 =
    [comm] (int contrib,std::vector<int> &gathered) -> void {
    mpi_gather32(contrib,gathered,comm); };
  cator->gather64 =
    [comm] (index_int contrib,std::vector<index_int> &gathered) -> void {
    return mpi_gather64(contrib,gathered,comm); };
  cator->overgather =
    [comm] (index_int contrib,int over) -> std::vector<index_int>* {
    return mpi_overgather(contrib,over,comm); };
  cator->reduce_scatter =
    [comm] (int *senders,int root) -> int { return mpi_reduce_scatter(senders,root,comm); };
  cator->reduce_max =
    [comm] (std::vector<index_int> local) -> std::vector<index_int> {
    return mpi_reduce_max(local,comm); };
  cator->reduce_min =
    [comm] (std::vector<index_int> local) -> std::vector<index_int> {
    return mpi_reduce_min(local,comm); };

};

//! Basic constructor
mpi_distribution::mpi_distribution( decomposition *d )
  : distribution(d),
    entity(entity_cookie::DISTRIBUTION) {
  make_mpi_communicator(this);
  set_dist_factory(); set_numa(); set_operate_routines(); set_memo_routines();
};

/*! Constructor from parallel structure. See \ref dependency::ensure_beta_distribution
  \todo try to make this delegate to mpi_distribution( dynamic_cast<decomposition*>(struc) )
*/
mpi_distribution::mpi_distribution( parallel_structure *struc )
  : distribution(struc),
    entity(entity_cookie::DISTRIBUTION) {
  make_mpi_communicator(this);
  set_dist_factory(); set_numa(); set_operate_routines(); set_memo_routines();
  try {
    memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in mpi distribution from struct {}",
		      struc->as_string()));
  }
};

//! \todo should this have a pure virtual base?
void mpi_distribution::set_memo_routines() {
  compute_global_first_index = [this] (distribution *dist) -> domain_coordinate {
    auto me = proc_coord(*this); domain_coordinate gf;
    try {
      auto lf = dist->first_index_r(me);
      gf = domain_coordinate( mpi_reduce_min( lf.data(),MPI_COMM_WORLD) );
    } catch (std::string c) {
      throw(fmt::format("Could not mpi global first on {}: <<{}>>",me.as_string(),c));
    }
    // fmt::print("Compute global first of <<{}>> on {}:{} as {}\n",
    // 	       dist->get_name(),proc_coord(*this)->as_string(),lf.as_string(),gf->as_string());
    return gf;
  };
  compute_global_last_index = [this] (distribution *dist) -> domain_coordinate {
    auto ll = dist->last_index_r(proc_coord(*this));
    auto gl = domain_coordinate( mpi_reduce_max( ll.data(),MPI_COMM_WORLD) );
    return gl;
  };
};

/*!
  Union of MPI structures.
  \todo try to be more clever in inferring resulting distribution type
  \todo this can be made a base method now that we have a factory for the mpi_distribution
*/
/*
distribution *mpi_distribution::mpi_dist_distr_union( distribution *other ) {
  if (get_orthogonal_dimension()!=other->get_orthogonal_dimension())
    throw(fmt::format("Incompatible orthogonal dimensions: this={}, other={}",
		      get_orthogonal_dimension(),other->get_orthogonal_dimension()));
  distribution *d = new mpi_distribution(dynamic_cast<decomposition*>(this));
  if (get_is_orthogonal() && other->get_is_orthogonal()) {
    for (int id=0; id<get_dimensionality(); id++) {
      auto new_pidx =
	get_dimension_structure(id)->struct_union(other->get_dimension_structure(id));
      if (new_pidx->outer_size()==0)
	throw(fmt::format("Made empty in dim {} from <<{}>> and <<{}>>",
			  id,get_dimension_structure(id)->as_string(),
			  get_dimension_structure(id)->as_string()));
      d->set_dimension_structure(id,new_pidx);
    }
    d->set_is_orthogonal(true); d->set_is_converted(false);
  } else {
    //printf("Union of multi_structures\n");
    if (get_is_orthogonal()) convert_to_multi_structures();
    if (other->get_is_orthogonal()) other->convert_to_multi_structures();
    for (int is=0; is<domains_volume(); is++) {
      auto pcoord = coordinate_from_linear(is);
      d->set_processor_structure
	( pcoord, get_processor_structure(pcoord)
	  ->struct_union(other->get_processor_structure(pcoord))->force_simplify() );
    }
    d->set_is_orthogonal(false); d->set_is_converted(false);
  }
  d->set_type(distribution_type::GENERAL);
  d->set_orthogonal_dimension( get_orthogonal_dimension() );
  try {
    d->memoize();
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not memoizing in unioned distr {}",d->get_name()));
  }
 return d;
};
*/

/*
 * NUMA
 */
//! \todo need unit tests for those location functions
void mpi_distribution::set_numa() {
  //snippet mpinuma
  // we capture `this' to be able to have the procid.
  compute_numa_structure = [this] (distribution *d) -> void {
    auto pcoord = d->coordinate_from_linear(procid());
    auto numa_struct = d->get_processor_structure(pcoord)->make_clone();
    d->set_numa_structure( numa_struct );
  };
  //snippet end
  //! The first index that this processor can see.
  numa_first_index = [this] (void) ->  const domain_coordinate {
    return first_index_r( proc_coord(*this) );
  };
  //! Total data that this processor can see.
  numa_local_size = [this] (void) -> index_int {
    return volume( coordinate_from_linear(procid()) ); };
  location_of_first_index = [] (distribution &d,processor_coordinate &p) -> index_int {
    return 0; }; 
  location_of_last_index = [] (distribution &d,processor_coordinate &p) -> index_int {
    return d.volume(p)-1; };
  //snippet end
  local_allocation = [this] (void) -> index_int {
    auto p = proc_coord( *dynamic_cast<decomposition*>(this) );
    return distribution::local_allocation_p(p); };
  //! A processor can only see its own part of the structure
  //snippet mpivisibility
  get_visibility = [this] (processor_coordinate &p) -> std::shared_ptr<multi_indexstruct> {
    return get_processor_structure(p); };
  //snippet end
};

void test_local_global_sanity(MPI_Comm comm,index_int ortho,index_int lsize,index_int gsize) {
  int minlocal,maxlocal,minglobal,maxglobal,minortho,maxortho;

  int i_ortho = (int)ortho,i_lsize = (int)lsize,i_gsize = (int) gsize;
  MPI_Allreduce(&i_ortho,&minortho,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_ortho,&maxortho,1,MPI_INT,MPI_MAX,comm);
  if (minortho!=maxortho) {
    printf("orthogonal dimension needs to be uniform %d\n",i_ortho); throw(58);}

  MPI_Allreduce(&i_lsize,&minlocal,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_lsize,&maxlocal,1,MPI_INT,MPI_MAX,comm);
  MPI_Allreduce(&i_gsize,&minglobal,1,MPI_INT,MPI_MIN,comm);
  MPI_Allreduce(&i_gsize,&maxglobal,1,MPI_INT,MPI_MAX,comm);

  if (minlocal<0 && minglobal<0) {
    printf("local/global insufficiently specified %d-%d\n",i_lsize,i_gsize); throw(55);}
  if (minglobal>=0) { // case: global specified
    if (minglobal!=maxglobal) {
      printf("global inconsistently specified %d\n",i_gsize); throw(56);}
    if ( maxlocal>=0 ) {
      printf("can not specify local (l:%d,max:%d) with global %d\n",
	     i_lsize,maxlocal,i_gsize); throw(57);}
  } else { // case: global unspecified
    if (minlocal<0) {
      printf("Can not leave local %d unspecified with global %d\n",
	     i_lsize,i_gsize); throw(57);}
  }
  return;
}

/*! Set the factory routines for creating a new object from this distribution.
  \todo the new_kernel factory does not depend on the distribution. can we move it to distribution
 */
void mpi_distribution::set_dist_factory() {
  // Factory for new distributions
  new_distribution_from_structure = [this] (parallel_structure *strct) -> distribution* {
    return new mpi_distribution(strct ); };
  //
  new_distribution_from_unique_local = [this]
    (std::shared_ptr<multi_indexstruct> strc) -> distribution* {
    auto d = new mpi_distribution( dynamic_cast<decomposition*>(this) );
    d->create_from_unique_local( strc ); d->memoize();
    return d; };
  // Factory for new scalar distributions
  new_scalar_distribution = [this] (void) -> distribution* {
    auto decomp = dynamic_cast<decomposition*>(this);
    if (decomp==nullptr) throw(fmt::format("weird upcast in mpi new_scalar_distribution"));
    return new mpi_scalar_distribution(decomp); };
  // Factory for new objects
  new_object = [this] (distribution *d) -> std::shared_ptr<object>
    { auto o = std::shared_ptr<object>( new mpi_object(d) ); return o; };
  // Factory for new objects from user-supplied data
  new_object_from_data = [this] ( double *d) -> std::shared_ptr<object>
    { return std::shared_ptr<object>( new mpi_object(this,d) ); };
  // Factory for making mode-dependent kernels
  new_kernel_from_object = [] ( std::shared_ptr<object> out ) -> kernel*
    { return new mpi_kernel(out); };
  // Factory for making mode-dependent kernels, for the imp_ops kernels.
  kernel_from_objects = [] ( std::shared_ptr<object> in,std::shared_ptr<object> out ) -> kernel*
    { return new mpi_kernel(in,out); };
  // Set the message factory.
  new_message =
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	    std::shared_ptr<multi_indexstruct> g) -> message* {
    return new mpi_message(this,snd,rcv,g);
  };
  new_embed_message =
    [this] (processor_coordinate &snd,processor_coordinate &rcv,
	    std::shared_ptr<multi_indexstruct> e,
	    std::shared_ptr<multi_indexstruct> g) -> message* {
    return new mpi_message(this,snd,rcv,e,g);
  };
};

//! Set a mask, after detecting unresolved parallel changes.
void mpi_distribution::add_mask( processor_mask *m ) {
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<int> includes = m->get_includes();
  int P = includes.size();
  MPI_Allreduce( MPI_IN_PLACE,includes.data(), P,MPI_INTEGER, MPI_MAX, comm);
  distribution::add_mask( new processor_mask(this,includes) );
};

/****
 **** Sparse matrix / index pattern
 ****/

void mpi_sparse_matrix::add_element( index_int i,index_int j,double v ) {
  if (mystruct->get_component(0)->contains_element(i)) {
    sparse_matrix::add_element(i,j,v);
  } else {
    throw(fmt::format("can not set remote element {} on proc {}",i,mycoord.as_string()));
  }
};

/****
 **** Object
 ****/

//! Allocating is an all-or-nothing activity for MPI processes. \todo fix for masks
void mpi_object::mpi_allocate() {
  if (has_data_status_allocated()) return;
  auto domains = get_domains();
  if (domains.size()>1)
    throw(std::string("Can not allocate mpi more than one domain"));
  try {
    for ( auto dom : domains ) {
      index_int s;
      if (0 && lives_on(dom)) continue;
      s = local_allocation_p(dom);
      if (s<0)
	throw(fmt::format("Negative allocation {} for dom=<<{}>>",s,dom.as_string()));
      //fmt::print("Allocating domain {} with {}\n",dom.as_string(),s);
      create_data(s,get_name());
      register_data_on_domain_number(0,get_numa_data_pointer().get(),s);
    }
  } catch (std::string c) { fmt::print("Error <<{}>>\n",c);
    throw(fmt::format("Could not allocate {}",as_string()));
  }
};

/*! Get processor-specific data. This can fail in case of masks.
  \todo should the mask test be basic
  \todo this looks the same as omp; lose this method or make basic?
*/
double *mpi_object::mpi_get_data(processor_coordinate &p) {
  //  if (lives_on(p)) {
    for ( auto dom : get_domains() )
      if (p.equals(dom)) { int locdom = get_domain_local_number(dom);
	auto pointer = get_data_pointer(locdom);
	if (pointer==nullptr)
	  throw(fmt::format
		("Data pointer is null for domain {}, locally {}",dom.as_string(),locdom));
	return pointer; }
    throw(fmt::format("data request for p={}, not in local domains",p.as_string()));
    //  } else throw("This object has no data on process p\n");
};

/****
 **** Message
 ****/

/*!
  MPI messages have to be sent over. This computes the buffer length;
  the packing is done in #mpi_message_as_buffer
*/
int mpi_message_buffer_length(int dim) {
  return
    (0
     + 1 // cookie
     + 1 // dimension
     + 1 // collective?
     + 2   // sender, receiver
     + 4 // tag contents
     + 3 // dep/in/out object number
     + 1 // trailing cookie
     )*sizeof(int) // 32 bytes
    +2*dim*sizeof(index_int) // src_index, size
    ;
};

/*!
  Return the content of a message as a character buffer,
  as constructed by MPI packing. This is used in
  mpi_task#create_send_structure_for_task.

  \todo figure out a way to supply the MPI communicator. as argument? make this class method?
*/
void mpi_message_as_buffer
    ( architecture *arch,message *msg,std::shared_ptr<std::string> buffer ) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int dim = msg->get_global_struct()->get_dimensionality();
  char *buf = (char*)(void*)buffer->data();
  int pos = 0, buflen = buffer->size();;

  message_tag *tag = msg->get_tag();
  int sender = msg->get_sender().linearize(msg->get_in_object().get()),
    receiver = msg->get_receiver().linearize(msg->get_in_object().get());
  //fmt::print("{}: packing receive message {}->{}\n",arch->mytid(),sender,receiver);
  if (receiver!=arch->mytid())
    throw(fmt::format("Packing msg {}->{} but I am {}",
		      sender,receiver,arch->mytid()));
  auto global_struct = msg->get_global_struct();

  int
    innumber = msg->get_in_object_number(),outnumber = msg->get_out_object_number(),
    depnumber = msg->get_dependency_number(),
    collective = msg->get_is_collective();

  int cookie = -37, long_int;
  MPI_Pack(&cookie,     1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&dim,        1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&collective, 1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&sender,     1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&receiver,   1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(tag->get_contents(),tag->get_length(),MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&depnumber,  1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&innumber,   1,MPI_INT,buf,buflen,&pos,comm);
  MPI_Pack(&outnumber,  1,MPI_INT,buf,buflen,&pos,comm);
  if (!global_struct->is_contiguous())
    throw(fmt::format("Can not send global struct of type {}",global_struct->type_as_string()));
  auto glb = global_struct->first_index_r(), siz = global_struct->local_size_r();
  for (int id=0; id<dim; id++) {
    index_int src_index=glb[id], size=siz[id];
    MPI_Pack(&src_index,1,MPI_INDEX_INT,buf,buflen,&pos,comm); long_int = pos;
    MPI_Pack(&size,     1,MPI_INDEX_INT,buf,buflen,&pos,comm); long_int = pos-long_int;
  }
  MPI_Pack(&cookie,     1,MPI_INT,buf,buflen,&pos,comm);
  if (pos>buflen)
    throw(fmt::format("packing {} but anticipated {} |long|={}\n",
		      pos,buflen,long_int));
};

/*!
  Receive a buffered message and unpack it to a real message. 
  
  This used to be a method of the mpi_message class, but we abandoned that.
  \todo figure out a way to supply the MPI communicator. as argument? make this class method?
  \todo we need to lose the \ref message::in_object_number. make objects findable.
 */
message *mpi_message_from_buffer( std::shared_ptr<task> t,int step,const std::string &buffer) {
  MPI_Comm comm = MPI_COMM_WORLD;
  int mytid = 314159; // only for CHK
  MPI_Status status;  message *rmsg;
  int len = buffer.size(), pos = 0,got,err;
  int sender,receiver;
  char *buf = (char*)(void*)buffer.data();

  err = MPI_Recv(buf,len,MPI_PACKED,MPI_ANY_SOURCE,step,comm,&status); CHK1(err);
  MPI_Get_count(&status,MPI_PACKED,&got);
  if (got<2*sizeof(int))
    throw(std::string("Message buffer way too short"));
  if (got>buffer.size())
    throw(fmt::format("Buffer incoming {}, space {}",got,buffer.size()));

  int cookie, dim;
  err = MPI_Unpack(buf,len,&pos,&cookie,   1,MPI_INT,comm); CHK1(err);
  if (cookie!=-37)
    throw(fmt::format("unpacking something weird {}",cookie));
  err = MPI_Unpack(buf,len,&pos,&dim, 1,MPI_INT,comm); CHK1(err);

  int chkcoll;
  err = MPI_Unpack(buf,len,&pos,&chkcoll, 1,MPI_INT,comm); CHK1(err);

  err = MPI_Unpack(buf,len,&pos,&sender,   1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&receiver, 1,MPI_INT,comm); CHK1(err);
  // fmt::print("{}: unpacking send message {}->{}\n",
  // 	     t->get_domain().as_string(),sender,receiver);
  if (receiver!=status.MPI_SOURCE)
    throw(fmt::format("Unpacking message {}->{}, but coming from {}\n",
		      sender,receiver,status.MPI_SOURCE));

  int tagstuff[4]; // VLE replace that 4 with a class method?
  err = MPI_Unpack(buf,len,&pos,tagstuff,   4,MPI_INT,comm); CHK1(err);

  int depnumber, innumber,outnumber;
  err = MPI_Unpack(buf,len,&pos,&depnumber,1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&innumber, 1,MPI_INT,comm); CHK1(err);
  err = MPI_Unpack(buf,len,&pos,&outnumber,1,MPI_INT,comm); CHK1(err);
  
  dependency *dep = t->get_dependency(depnumber);
  auto out_obj = dep->get_beta_object(),
    in_obj = dep->get_in_object();
  //fmt::print("send msg: in obj={}, out obj={}\n",in_obj->get_name(),out_obj->get_name());

  auto global_struct = std::shared_ptr<multi_indexstruct>( new multi_indexstruct(dim) );
  for (int id=0; id<dim; id++) {
    index_int src_index,size;
    err = MPI_Unpack(buf,len,&pos,&src_index,1,MPI_INDEX_INT,comm); CHK1(err);
    err = MPI_Unpack(buf,len,&pos,&size,     1,MPI_INDEX_INT,comm); CHK1(err);
    global_struct->set_component
      (id,std::shared_ptr<indexstruct>( new contiguous_indexstruct(src_index,src_index+size-1)) );
  }
  err = MPI_Unpack(buf,len,&pos,&cookie,   1,MPI_INT,comm); CHK1(err);
  if (cookie!=-37)
    throw(fmt::format("corruption check failed {}",cookie));
  try {
    auto 
      snd = in_obj->coordinate_from_linear(sender),
      rcv = out_obj->coordinate_from_linear(receiver);
    rmsg = out_obj->new_message(snd,rcv,global_struct);
  } catch (std::string c) { fmt::print("Error <<{}>> creating message from buffer\n",c); };
  rmsg->set_tag( new message_tag(tagstuff) );
  rmsg->set_in_object(in_obj); rmsg->set_out_object(out_obj);
  rmsg->compute_src_index(); rmsg->set_send_type();
  rmsg->set_is_collective(chkcoll);
  
  return rmsg;
};

/*!
  Compute the location from where we start sending,
  both as an index, and as an MPI subarray type.
*/
//snippet mpisrcindex
void mpi_message::compute_src_index() {
  message::compute_src_index();
  int
    ortho = get_in_object()->get_orthogonal_dimension(),
    dim = get_in_object()->get_dimensionality();
  int err;
  err = MPI_Type_create_subarray(dim+1,numa_sizes,struct_sizes,struct_starts,MPI_ORDER_C,MPI_DOUBLE,&embed_type);
  if (err!=0) throw(fmt::format("Type create subarray failed for src: err={}",err));
  MPI_Type_commit(&embed_type);
  if (dim==1) {
    lb = struct_starts[0]; extent = struct_sizes[0];
  } else if (dim==2) {
    lb = struct_starts[0]*numa_sizes[1]+struct_starts[1];
    extent = (struct_starts[0]+struct_sizes[0]-1)*numa_sizes[1] + struct_starts[1]+struct_sizes[1] - lb;
  } else
    throw(fmt::format("Can not compute lb/extent in dim>2"));
  if (extent<=0)
    throw(fmt::format("Zero extent in message <<{}>>",as_string()));
};
//snippet end

//snippet mpitarindex
//! Where does the message land in the beta structure?
void mpi_message::compute_tar_index() {
  message::compute_tar_index();
  int
    ortho = get_out_object()->get_orthogonal_dimension(),
    dim = get_out_object()->get_dimensionality();
  int err;
  err = MPI_Type_create_subarray(dim+1,numa_sizes,struct_sizes,struct_starts,MPI_ORDER_C,MPI_DOUBLE,&embed_type);
  if (err!=0) throw(fmt::format("Type create subarray failed for src: err={}",err));
  MPI_Type_commit(&embed_type);
  lb = struct_starts[dim]; // compute analytical lower bound and extent
  extent = struct_sizes[dim];
  for (int id=dim-1; id>=0; id--) {
    lb += struct_starts[id]*numa_sizes[id+1];
    extent += (struct_sizes[id]-1)*numa_sizes[id+1];
  }
  {
    MPI_Aint true_lb,true_extent;
    MPI_Type_get_true_extent(embed_type,&true_lb,&true_extent);
    if (true_lb!=lb*sizeof(double))
      throw(fmt::format("Computed lb {} mismatch true lb {}",lb,true_lb));
    if (true_extent!=extent*sizeof(double))
      throw(fmt::format("Computed extent {} mismatch true extent {}",extent,true_extent));
  }
  if (extent<=0)
    throw(fmt::format("Zero extent in message <<{}>>",as_string()));
};
//snippet end

/****
 **** Task
 ****/

/*!
  Build the send messages;
*/
void mpi_task::derive_send_messages() {
  auto out = get_out_object();
  int dim = out->get_dimensionality(), step = get_step();
  architecture *arch = dynamic_cast<architecture*>(out.get());
  auto dom = get_domain(); int mytid = dom.linearize(out.get());
  MPI_Comm comm = MPI_COMM_WORLD;

  auto recv_messages = get_receive_messages();
  auto deps = get_dependencies();
  int nsends, nrecvs = recv_messages.size();
  try {
    nsends = get_nsends();
  } catch (std::string c ) {
    fmt::print("Error <<{}>> determining nsends",c);
    throw(std::string("get_nsends fail")); };
  
  std::vector< MPI_Request > mpi_requests;  mpi_requests.reserve(nrecvs);
  std::vector< MPI_Status >  mpi_statuses;  mpi_statuses.reserve(nrecvs); 
  std::vector< std::shared_ptr<std::string> > buffers;       buffers.reserve(nrecvs);
  int mpi_err;
  // turn each recv message into a buffer and asynchronously send it to the future sender
  for ( auto msg : recv_messages ) {
    int buflen = mpi_message_buffer_length(dim)+8;
    MPI_Request req;
    auto  buffer = std::shared_ptr<std::string>(new std::string); buffer->reserve(buflen);
    for (int i=0; i<buflen; i++) buffer->push_back(0);
    try { arch->message_as_buffer(arch,msg,buffer);
    } catch (std::string c) {
      fmt::print("Error <<{}>> for buffering msg {}->{} in <<{}>>\n",
		 c,msg->get_sender().as_string(),msg->get_receiver().as_string(),get_name());
      throw(fmt::format("Could not convert msg to buffer")); }
    buffers.push_back(buffer);
    auto otherdom = msg->get_sender(); int other = otherdom.linearize(out.get());
    MPI_Isend(buffer->data(),buflen,MPI_PACKED,other,step,comm,&req);
    mpi_requests.push_back(req);
  }
  // now receive the buffers that tell you what is requested of you
  {
    //mpi_object *obj = dynamic_cast<mpi_object*>(out);
    //if (obj==nullptr) throw(std::string("Could not cast object to mpi"));
    int buflen = mpi_message_buffer_length(dim)+8;
    std::string buf; buf.reserve(buflen); for (int i=0; i<buflen; i++) buf.push_back(0);
    for (int i=0; i<nsends; ++i) {
      message *msg;
      try { msg = this->message_from_buffer(step,buf);
      } catch (std::string c) {
	fmt::print("Error in message from buffer w obj <<{}>>: <<{}>>\n",out->get_name(),c);
	throw(fmt::format("Could not derive send msgs in <<{}>>",get_name()));
      }
      msg->set_name( fmt::format("send-{}-obj:{}->{}",
	   msg->get_name(),msg->get_in_object_number(),msg->get_out_object_number()) );
      send_messages.push_back(msg);
    }
    //delete buf;
  }
  int irequest = mpi_requests.size();
  mpi_err = MPI_Waitall(irequest,mpi_requests.data(),MPI_STATUSES_IGNORE);
  //mpi_statuses.data());
  // if (mpi_err!=0) {
  //   for (int ireq=0; ireq<irequest; ireq++) {
  //     int errorcode = mpi_statuses[ireq].MPI_ERROR;
  //     if (errorcode!=0) {
  // 	char message[256]; int msglen;
  // 	MPI_Error_string(errorcode,message,&msglen);
  // 	fmt::print("Error [{}] in request # {}\n",message,ireq);
  //     }
  //   }
  // }
  //delete mpi_requests; delete mpi_statuses;
   
  // localize send structures
  for ( auto msg : send_messages ) {
    msg->relativize_to( msg->get_in_object()->get_processor_structure(dom) );
  }
};

/*!
  Post an MPI_Isend for a bunch of messages. The requests are passed back in
  an array that is in/out: this way a task can post messages that really
  belong to a much later task.
  \todo is there a way to get the MPI_Comm without that casting? note that by now none of our base classes are mpi_specific.
  \todo why do we make a new processor coordinate?
*/
request *mpi_task::notifyReadyToSendMsg( message* msg ) {
  MPI_Comm comm = *(MPI_Comm*)( get_out_object()->get_communicator_context() );
    auto vec = msg->get_in_object(), halo = msg->get_out_object();
    processor_coordinate dom(get_domain()); double *data = vec->get_data(dom);
    int k = vec->get_orthogonal_dimension(), ireq=0;
    auto numa_struct = vec->get_processor_structure(dom),
      local_struct = msg->get_local_struct(), global_struct = msg->get_global_struct();
    index_int numa_size = numa_struct->volume();
    MPI_Request req;
    auto sender = msg->get_sender(),receiver = msg->get_receiver();
    if (sender.equals(receiver) && vec->has_data_status_inherited()
	&& vec->get_data_parent()==halo->get_object_number()) { // we can skip certain messages
      msg->set_status( message_status::SKIPPED );
      return nullptr;
    }
    {
      index_int
	src_index = msg->get_src_index(),src_size = local_struct->volume();
      if (k*src_index+k*src_size>k*numa_size)
	throw(fmt::format("Message <<{}>>: send buffer overflow {}+{}>{} (k={})",
			  msg->as_string(),k*src_index,k*src_size,numa_size,k));
    }
    if (halo->get_use_rma())
      throw(std::string("RMA not implemented"));
    mpi_message *mmsg = dynamic_cast<mpi_message*>(msg);
    if (mmsg==nullptr) throw(std::string("Could not convert snd msg to mpi"));
    int sender_no = sender.linearize(vec.get()),
      receiver_no = receiver.linearize(vec.get());
    //snippet mpisend
    MPI_Isend( data,1,mmsg->embed_type,receiver_no,msg->get_tag_value(),comm,&req);
    //snippet end
    return new mpi_request(msg,req);
};

/*!
  Post an MPI_Irecv for a bunch of messages. The requests are passed back in
  an array that is in/out: this way a task can post messages that really
  belong to a much later task.

  \todo should we use get_numa_structure?
  \todo can we use the embed_type in the Iallgather case?
  \todo can we integrate this further with the OMP version?
*/
void mpi_task::acceptReadyToSend
    ( std::vector<message*> &msgs,request_vector *requests ) {
  MPI_Comm comm = MPI_COMM_WORLD;
  for ( auto msg : msgs ) {
    try {
      mpi_message *rmsg = dynamic_cast<mpi_message*>(msg);
      if (rmsg==nullptr) throw(std::string("Could not convert recv msg to mpi"));
      auto vec = rmsg->get_in_object(), halo = rmsg->get_out_object();
      auto dom = get_domain(); 
      int k = halo->get_orthogonal_dimension();
      auto numa_struct = halo->get_processor_structure(dom),
	local_struct = rmsg->get_local_struct(), global_struct = rmsg->get_global_struct();
      index_int size = local_struct->volume(), numa_size = halo->local_allocation_p(dom);
      auto sender = rmsg->get_sender(),receiver = rmsg->get_receiver();
      int sender_no = sender.linearize(vec.get()),
	receiver_no = receiver.linearize(vec.get());
      halo->set_data_is_filled(receiver); // can we put this as a lambda in the request?
      if (sender.equals(receiver) && vec->has_data_status_inherited()
	  && vec->get_data_parent()==halo->get_object_number())
	continue;
      MPI_Request req;
      if (rmsg->get_is_collective()) {
	auto smsgs = get_send_messages();
	if (smsgs.size()!=1)
	  throw(std::string("Weird send msgs for MPI collective"));
	mpi_message *smsg = dynamic_cast<mpi_message*>(smsgs.at(0));
	auto sobject = smsg->get_in_object();
	auto sdata = sobject->get_data(dom);
	auto rdata = halo->get_raw_data();
	std::vector<int> offsets, sizes;
	try {
	  offsets = sobject->get_linear_offsets(); sizes = sobject->get_linear_sizes();
	} catch (std::string c) { fmt::print("Error: {}\n",c);
	  throw(fmt::format("Error getting linear offset/sizes for : {}",halo->as_string())); }
	int send_size = size * !sobject->get_processor_skip(sender_no);
	//MPI_Iallgather(
	//fmt::print("{} contributing {:e} of size {}\n",dom->as_string(),rdata[0],send_size);
	MPI_Iallgatherv(
			sdata,send_size,MPI_DOUBLE,
			//data,size,MPI_DOUBLE, comm,&req);
			rdata,sizes.data(),offsets.data(),MPI_DOUBLE,
			comm,&req);
      } else {
	if (rmsg->lb+rmsg->extent>numa_size)
	  throw(fmt::format("Irecv buffer overflow: lb={} extent={}, available={}",
			    rmsg->lb,rmsg->extent,numa_size));
	double *rdata = halo->get_data(dom);
        //snippet mpirecv
        MPI_Irecv( rdata,1,rmsg->embed_type,sender_no,rmsg->get_tag_value(),comm,&req);
        //snippet end
      }
      requests->add_request( new mpi_request(rmsg,req) );
    } catch (std::string c) { fmt::print("Error <<{}>> for msg <<{}>>\n",c,msg->as_string());
      throw(fmt::format("Could not post receive message for task <<{}>>",this->as_string()));
    }
  }
};

//snippet mpi-depend
/*!
  Create a dependence in the local task graph. Since the task can be on a different
  address space, we declare a dependence on the task from the same kernel,
  but on this domain.
 */
void mpi_task::declare_dependence_on_task( task_id *id ) {
  int step = id->get_step(); auto domain = this->get_domain();
  try {
    add_predecessor( find_other_task_by_coordinates(step,domain) );
  } catch (std::string c) {
    fmt::print("Task <<{}>> error <<{}>> locating <{},{}>.\n{}\n",
	       get_name(),c,step,domain.as_string(),this->as_string());
    throw(std::string("Could not find MPI local predecessor"));
  };
};
//snippet end

/****
 **** Kernel
 ****/

/*! Construct the right kind of task for the base class
  method \ref kernel::split_to_tasks.
*/
std::shared_ptr<task> mpi_kernel::make_task_for_domain(processor_coordinate &dom) {
  auto t = std::shared_ptr<task>( new mpi_task(dom,this) );
  return t;
}

/****
 **** Queue
 ****/
