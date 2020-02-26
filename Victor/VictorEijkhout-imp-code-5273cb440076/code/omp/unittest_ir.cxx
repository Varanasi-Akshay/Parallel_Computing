/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-7
 ****
 **** Unit tests for the OpenMP product backend of IMP
 **** based on the CATCH framework (https://github.com/philsquared/Catch)
 ****
 **** test dumping the Intermediate Representation
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "omp_base.h"
#include "omp_static_vars.h"
#include "unittest_functions.h"

TEST_CASE( "We can reset the dump","[environment][output][2]" ) {
  REQUIRE( env->nprocs() > 0 );
  REQUIRE_NOTHROW( env->set_outputfile("p.ir") );
  REQUIRE_NOTHROW( env->print() );
}

TEST_CASE( "Dump an indexstruct","[indexstruct][output][3]" ) {
  REQUIRE( env->nprocs() > 0 );
  REQUIRE_NOTHROW( env->set_outputfile("index.ir") );
  indexstruct *ii;
  CHECK_NOTHROW( ii = new indexstruct(5,7) );
  REQUIRE_NOTHROW( ii->print(env) );
}

TEST_CASE( "Dump a parallel indexstruct","[indexstruct][output][4]" ) {
  REQUIRE( env->nprocs() > 0 );
  parallel_indexstruct *p;

  SECTION( "from local" ) {
    REQUIRE_NOTHROW( env->set_outputfile("local.ir") );
    CHECK_NOTHROW( p = new parallel_indexstruct(env->get_architecture()) );
    CHECK_NOTHROW( p->create_from_uniform_local_size(23) );
    REQUIRE_NOTHROW( p->print(env) );
  }
  SECTION( "from global" ) {
    REQUIRE_NOTHROW( env->set_outputfile("global.ir") );
    CHECK_NOTHROW( p = new parallel_indexstruct(env->get_architecture()) );
    CHECK_NOTHROW( p->create_from_global_size(511) );
    REQUIRE_NOTHROW( p->print(env) );
  }
}

TEST_CASE( "Dump message","[message][output][5]" ) {
  message *m;
  REQUIRE_NOTHROW( m = new message(env,ntids-1,0,new indexstruct(20,40)) );
  REQUIRE_NOTHROW( env->set_outputfile("mess.ir") );
  REQUIRE_NOTHROW( m->print(env) );
};

TEST_CASE( "Dump distribution","[distribution][output][6]" ) {
  distribution *d;
  REQUIRE_NOTHROW( d = new omp_distribution(env,"disjoint-block",500) );  
  d->set_name("500-equal");
  REQUIRE_NOTHROW( env->set_outputfile("dist.ir") );
  REQUIRE_NOTHROW( d->print(env) );
};

TEST_CASE( "Dump object","[object][output][7]" ) {
  distribution *d;
  REQUIRE_NOTHROW( d = new omp_distribution(env,"disjoint-block",500) );  
  object *o;
  REQUIRE_NOTHROW( o = new omp_object(d) );
  REQUIRE_NOTHROW( env->set_outputfile("obj.ir") );
  REQUIRE_NOTHROW( o->print(env) );
};

TEST_CASE( "dump task","[task][output][8]" ) {
  distribution *d1,*d2;
  REQUIRE_NOTHROW( d1 = new omp_distribution(env,"disjoint-block",200) );  
  REQUIRE_NOTHROW( d2 = new omp_replicated_scalar(env) );
  object *o1,*o2;
  REQUIRE_NOTHROW( o1 = new omp_object(d1) );
  o1->set_name("residual-vector");
  REQUIRE_NOTHROW( o2 = new omp_object(d2) );
  o1->set_name("search-direction");
  task *t;
  REQUIRE_NOTHROW( t = new omp_task(5,2,o1,o2) );
  t->set_name("do-something");
  REQUIRE_NOTHROW( env->set_outputfile("task.ir") );
  REQUIRE_NOTHROW( t->print(env) );
};

TEST_CASE( "Threepoint queue with gen kernel","[queue][execute][halo][modulo][10]" ) {

  env->set_outputfile("queue.ir");
  
  int nlocal=17,nsteps=4;
  omp_distribution *block = 
    new omp_distribution(env,"disjoint-block",nlocal*ntids);
  ioperator *no_op,*right_shift_mod,*left_shift_mod;
  no_op = new ioperator("none");
  right_shift_mod = new ioperator(">>1");
  left_shift_mod  = new ioperator("<<1");

  omp_object
    *xvector = new omp_object(block),
    **yvector = new omp_object*[nsteps];
  for (int iv=0; iv<nsteps; iv++) {
    yvector[iv] = new omp_object(block);
  }

  omp_algorithm *queue;
  CHECK_NOTHROW( queue = new omp_algorithm(arch) );
  CHECK_NOTHROW( queue->set_name("threepoint modulo") );
  {
    omp_kernel *k = new omp_origin_kernel(xvector);
    k->set_localexecutefn(  &vecset );
    CHECK_NOTHROW( queue->add_kernel( k ) );
  }
  for (int iv=0; iv<nsteps; iv++) {
    omp_kernel *k;
    if (iv==0) {
      k = new omp_kernel(xvector,yvector[0]);
    } else {
      k = new omp_kernel(yvector[iv-1],yvector[iv]);
    }
    k->add_sigma_operator( no_op );
    k->add_sigma_operator( left_shift_mod );
    k->add_sigma_operator( right_shift_mod );
    k->set_localexecutefn(  &threepointsum );
    CHECK_NOTHROW( queue->add_kernel(k) );
  }
  
  CHECK_NOTHROW( queue->analyze_dependencies() );
  CHECK_NOTHROW( queue->print(env) );

}
