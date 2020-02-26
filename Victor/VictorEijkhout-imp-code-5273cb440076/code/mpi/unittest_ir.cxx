/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014/5
 ****
 **** Unit tests for the MPI product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** test dumping the Intermediate Representation
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "We can reset the dump","[environment][output][2]" ) {
  REQUIRE( arch->nprocs() > 0 );
  REQUIRE_NOTHROW( env->set_ir_outputfile("env") );
  REQUIRE_NOTHROW( env->print() );
}

TEST_CASE( "Dump an indexstruct","[indexstruct][output][3]" ) {
  REQUIRE( arch->nprocs() > 0 );
  indexstruct *ii;

  REQUIRE_NOTHROW( env->set_ir_outputfile("index") );
  CHECK_NOTHROW( ii = new contiguous_indexstruct(5,7) );
  REQUIRE_NOTHROW( ii->print(env) );

  REQUIRE_NOTHROW( env->set_ir_outputfile("stride") );
  CHECK_NOTHROW( ii = new strided_indexstruct(5,11,3) );
  REQUIRE_NOTHROW( ii->print(env) );
}

TEST_CASE( "Dump a parallel indexstruct","[indexstruct][output][4]" ) {
  REQUIRE( arch->nprocs() > 0 );
  parallel_indexstruct *p;

  SECTION( "from local" ) {
    REQUIRE_NOTHROW( env->set_ir_outputfile("local") );
    CHECK_NOTHROW( p = new parallel_indexstruct(arch) );
    CHECK_NOTHROW( p->create_from_uniform_local_size(23) );
    REQUIRE_NOTHROW( p->print(env) );
  }
  SECTION( "from global" ) {
    REQUIRE_NOTHROW( env->set_ir_outputfile("global") );
    CHECK_NOTHROW( p = new parallel_indexstruct(arch) );
    CHECK_NOTHROW( p->create_from_global_size(511) );
    REQUIRE_NOTHROW( p->print(env) );
  }
}

TEST_CASE( "Dump message","[message][output][5]" ) {
  message *m;
  REQUIRE_NOTHROW( m = new message(ntids-1,0,new contiguous_indexstruct(20,40)) );
  REQUIRE_NOTHROW( env->set_ir_outputfile("mess") );
  REQUIRE_NOTHROW( m->print(env) );
};

TEST_CASE( "Dump distribution","[distribution][output][6]" ) {
  distribution *d;
  REQUIRE_NOTHROW( d = new mpi_block_distribution(arch,500) );  
  d->set_name("500-equal");
  REQUIRE_NOTHROW( env->set_ir_outputfile("dist") );
  REQUIRE_NOTHROW( d->print(env) );
};

TEST_CASE( "Dump object","[object][output][7]" ) {
  distribution *d;
  REQUIRE_NOTHROW( d = new mpi_block_distribution(arch,500) );  
  object *o;
  REQUIRE_NOTHROW( o = new mpi_object(d) );
  REQUIRE_NOTHROW( env->set_ir_outputfile("obj") );
  REQUIRE_NOTHROW( o->print(env) );
};

TEST_CASE( "dump task","[task][output][8]" ) {
  distribution *d1,*d2;
  REQUIRE_NOTHROW( d1 = new mpi_block_distribution(arch,200) );  
  REQUIRE_NOTHROW( d2 = new mpi_replicated_distribution(arch) );
  object *o1,*o2;
  REQUIRE_NOTHROW( o1 = new mpi_object(d1) );
  o1->set_name("residual-vector");
  REQUIRE_NOTHROW( o2 = new mpi_object(d2) );
  o1->set_name("search-direction");
  task *t; kernel *k;
  REQUIRE_NOTHROW( k = new mpi_kernel(o1,o2) );
  REQUIRE_NOTHROW( t = new mpi_task(2,k) );
  t->set_name("do-something");
  REQUIRE_NOTHROW( env->set_ir_outputfile("task") );
  REQUIRE_NOTHROW( t->print(env) );
};

TEST_CASE( "Threepoint queue with gen kernel","[queue][execute][halo][modulo][10]" ) {

  env->set_ir_outputfile("queue");
  
  int nlocal=17,nsteps=4;
  mpi_distribution *block = 
    new mpi_block_distribution(arch,nlocal*ntids);
  index_int
    my_first = block->first_index(mytid), my_last = block->last_index(mytid);
  CHECK( my_first==mytid*nlocal );
  CHECK( my_last==(mytid+1)*nlocal-1 );

  ioperator *no_op,*right_shift_mod,*left_shift_mod;
  no_op = new ioperator("none");
  right_shift_mod = new ioperator(">>1");
  left_shift_mod  = new ioperator("<<1");

  mpi_object
    *xvector = new mpi_object(block),
    **yvector = new mpi_object*[nsteps];
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = new mpi_object(block);
  }

  mpi_algorithm *queue;
  CHECK_NOTHROW( queue = new mpi_algorithm(arch) );
  CHECK_NOTHROW( queue->set_name("threepoint modulo") );
  {
    mpi_kernel *k = new mpi_origin_kernel(xvector);
    //    k->set_localexecutefn(  &vecset );
    CHECK_NOTHROW( queue->add_kernel( k ) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    mpi_kernel *k;
    if (iv==0) {
      k = new mpi_kernel(xvector,yvector[0]);
    } else {
      k = new mpi_kernel(yvector[iv-1],yvector[iv]);
    }
    { fmt::MemoryWriter w; w.write("3pt-compute-y{}",iv);
      k->set_name( w.str() );
    }
    k->add_beta_oper( no_op );
    k->add_beta_oper( left_shift_mod );
    k->add_beta_oper( right_shift_mod );
    k->set_localexecutefn(  &threepointsummod );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  //  CHECK_NOTHROW( queue->print(env) );
  REQUIRE_NOTHROW( env->print_single( queue->header_as_string() ) );
  REQUIRE_NOTHROW( env->print_all( queue->contents_as_string() ) );

  env->close_ir_outputfile();
}

TEST_CASE( "other statistics","[statistics][20]" ) {
  int nlocal=100;
  distribution *blocked;
  REQUIRE_NOTHROW( env->reset_statistics() );
  REQUIRE_NOTHROW( blocked = new mpi_block_distribution(arch,nlocal,-1) );
  SECTION( "explicit object creation" ) {
    object *o1,*o2;
    REQUIRE_NOTHROW( o1 = new mpi_object(blocked) );
    REQUIRE_NOTHROW( o2 = new mpi_object(blocked) );
    //    CHECK( env->n_objects_created()==2 );
  }
  SECTION( "count the halo too" ) {
    object *o1,*o2;
    REQUIRE_NOTHROW( o1 = new mpi_object(blocked) );
    REQUIRE_NOTHROW( o2 = new mpi_object(blocked) );
    //    CHECK( env->n_objects_created()==2 );
    kernel *k;
    REQUIRE_NOTHROW( k = new mpi_kernel(o1,o2) );
    REQUIRE_NOTHROW( k->add_beta_oper( new ioperator("none") ) );
    REQUIRE_NOTHROW( k->analyze_dependencies() );
    //    CHECK( env->n_objects_created()==3 );
  }
}
