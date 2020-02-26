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
 **** unit tests for the indexstruct package
 **** (tests do not actually rely on MPI)
 ****
 ****************************************************************/

#include <stdlib.h>
#include <math.h>

#include "catch.hpp"

#include "mpi_base.h"
#include "mpi_static_vars.h"

TEST_CASE( "contiguous indexstruct","[indexstruct][1]" ) {

  indexstruct *i1,*i2,*i3,*i4;

  SECTION( "basic construction" ) {
    // type testing
    SECTION( "contiguous" ) {
      i1 = new contiguous_indexstruct(0,5);
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==0 );
      CHECK( i1->last_index()==5 );
      CHECK( i1->local_size()==6 );
      CHECK( i1->stride()==1 );
      CHECK( i1->find(0)==0 );
      CHECK( i1->find(5)==5 );
      REQUIRE_THROWS( i1->find(6) );
      REQUIRE_NOTHROW( delete i1 );
    }
    SECTION( "by accretion" ) {
      i1 = new contiguous_indexstruct(2,5);
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1->add_element(3) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1->add_element(5) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==5 );
      REQUIRE_NOTHROW( i1->add_element(6) );
      CHECK( i1->is_contiguous() );
      CHECK( i1->local_size()==5 );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==6 );
      REQUIRE_NOTHROW( i1->add_element(1) );
      CHECK( i1->is_contiguous() );
      CHECK( i1->local_size()==6 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==6 );
      REQUIRE_THROWS( i1->add_element(9) );
    }
    SECTION( "more contiguous" ) {
      i1 = new contiguous_indexstruct(1,7);
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->stride()==1 );
      REQUIRE_THROWS( i1->find(0) );
      REQUIRE_NOTHROW( delete i1 );
    }
    SECTION( "strided" ) {
      i1 = new strided_indexstruct(2,6,2);
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==6 );
      CHECK( i1->local_size()==3 );
      CHECK( i1->stride()==2 );
      CHECK( i1->find(2)==0 );
      REQUIRE_THROWS( i1->find(3) );
      CHECK( i1->find(4)==1 );
      CHECK( i1->find(6)==2 );
      REQUIRE_THROWS( i1->find(7) );
      REQUIRE_THROWS( i1->find(8) );
      REQUIRE_NOTHROW( delete i1 );
    }
  }

  SECTION( "striding and operations" ) {
    i1 = new strided_indexstruct(4,7,2);

    SECTION( "basic stride tests" ) {
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==6 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );

      i2 = new contiguous_indexstruct(4,6);
      CHECK( !i1->equals(i2) );
      CHECK( !i2->equals(i1) );
      REQUIRE_NOTHROW( delete i2 );

      i2 = new strided_indexstruct(4,6,2);
      CHECK( i1->equals(i2) );
      CHECK( i2->equals(i1) );
      REQUIRE_NOTHROW( delete i2 );

      CHECK( i1->contains_element(4) );
      CHECK( !i1->contains_element(5) );
      CHECK( i1->contains_element(6) );
      CHECK( !i1->contains_element(7) );
    }

    SECTION( "strided containment" ) {
      i2 = new contiguous_indexstruct(4,8);
      i3 = new strided_indexstruct(4,8,2);
      i4 = new strided_indexstruct(4,8,4);
      CHECK( i2->contains(i3) );
      CHECK( !i3->contains(i2) );
      CHECK( !i2->equals(i3) );
      CHECK( i2->contains(i4) );
      CHECK( i3->contains(i4) );
      CHECK( !i4->contains(i2) );
      CHECK( !i4->contains(i3) );
    }

    SECTION( "translation forward" ) {
      i1->translate_by(1);
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==5 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation backward" ) {
      i1->translate_by(-2);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_strided() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation through zero" ) {
      i1->translate_by(-5);
      CHECK( i1->is_strided() );
      CHECK( !i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==1 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    REQUIRE_NOTHROW( delete i1 );
  }

  SECTION( "copy indexstruct" ) {
    indexstruct *i1 = new contiguous_indexstruct(7,15), *i2;
    REQUIRE_NOTHROW( i2 = i1->make_clone() );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==15 );
    REQUIRE_NOTHROW( i1->translate_by(1) );
    CHECK( i1->first_index()==8 );
    CHECK( i1->last_index()==16 );
    CHECK( i2->first_index()==7 );
    CHECK( i2->last_index()==15 );  
  }
}

TEST_CASE( "indexed indexstruct","[indexstruct][2]" ) {

  indexstruct *i1,*i2,*i3;

  SECTION( "basic construction" ) {
    SECTION( "correct" ) {
      int len=3; index_int idx[3] = {1,2,4};
      i1 = new indexed_indexstruct(len,idx);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
      REQUIRE_NOTHROW( delete i1 );
    }

    SECTION( "unsorted throws an error" ) {
      int len=4; index_int idx[4] = {1,2,6,4};
      CHECK_THROWS( i1 = new indexed_indexstruct(len,idx) );
    }

    SECTION( "negative indices allowed" ) {
      int len=3; index_int idx[3] = {-1,2,4};
      i1 = new indexed_indexstruct(len,idx);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
      REQUIRE_NOTHROW( delete i1 );
    }

    SECTION( "gradual construction" ) {
      int len=3; index_int idx[3] = {4,9,20};
      i1 = new indexed_indexstruct(len,idx);
      CHECK( i1->local_size()==3 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==20 );
      REQUIRE_NOTHROW( i1->add_element(9) );
      CHECK( i1->local_size()==3 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==20 );
      REQUIRE_NOTHROW( i1->add_element(30) );
      CHECK( i1->local_size()==4 );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==30 );
      REQUIRE_NOTHROW( i1->add_element(1) );
      CHECK( i1->local_size()==5 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==30 );
      REQUIRE_NOTHROW( i1->add_element(10) );
      CHECK( i1->local_size()==6 );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==30 );
    }
  }

  SECTION( "striding and operations" ) {
    int len=5; index_int idx[5] = {1,2,4,7,9};
    indexstruct *i1;
    REQUIRE_NOTHROW( i1 = new indexed_indexstruct(len,idx) );

    SECTION( "basic stride tests" ) {
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==9 );
      CHECK( i1->local_size()==len );

      CHECK( !i1->contains_element(0) );
      CHECK( i1->contains_element(1) );
      CHECK( i1->contains_element(4) );
      CHECK( !i1->contains_element(5) );
      CHECK( !i1->contains_element(6) );
      CHECK( i1->contains_element(7) );

      CHECK( i1->find(1)==0 );
      CHECK( i1->find(7)==3 );
      REQUIRE_THROWS( i1->find(0) );
      REQUIRE_THROWS( i1->find(8) );
    }

    SECTION( "translation forward" ) {
      i1->translate_by(1);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==10 );
      CHECK( i1->local_size()==len );
    }

    SECTION( "translation through zero" ) {
      i1->translate_by(-2);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->local_size()==len );
    }

    REQUIRE_NOTHROW( delete i1 );
  }
}

TEST_CASE( "indexstruct intersections","[indexstruct][intersect][3]" ) {
  indexstruct *i1,*i2,*i3,*i4;
  
  i1 = new contiguous_indexstruct(1,10);
  SECTION( "cont-cont" ) {
    i2 = new contiguous_indexstruct(5,12);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==10 );
    CHECK( !i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );
    
    i2 = new contiguous_indexstruct(10,12); // [1,10] & [10,12] => [10,10]
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==10 );
    CHECK( i3->last_index()==10 );
    CHECK( !i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );
    REQUIRE_THROWS( i4 = i2->relativize(i1) );
    REQUIRE_NOTHROW( i4 = i3->relativize(i1) ); // [10,10] in [1,10] is [9,9]
    CHECK( i4->is_contiguous() );
    CHECK( i4->first_index()==9 );
    CHECK( i4->last_index()==9 );
    REQUIRE_NOTHROW( i4 = i3->relativize(i2) );
    CHECK( i4->is_contiguous() );
    CHECK( i4->first_index()==0 );
    CHECK( i4->last_index()==0 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new contiguous_indexstruct(11,12);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3->is_empty() );
    REQUIRE_NOTHROW( delete i2 );

    i2 = new strided_indexstruct(10,12,2);
    i3 = new strided_indexstruct(8,14,2);
    REQUIRE_NOTHROW( i4 = i2->relativize(i3) );
    CHECK( i4->stride()==2 );
    CHECK( i4->first_index()==2 );
    CHECK( i4->local_size()==2 );
  }

  SECTION( "cont-idx" ) {
    int len=3; index_int idx[3] = {4,8,11};
    i2 = new indexed_indexstruct(len,idx);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( !i3->is_empty() );
    CHECK( i3->is_indexed() );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==8 );
    CHECK( !i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );
    REQUIRE_THROWS( i2->relativize(i1) );

    len=3; index_int idxs[3] = {4,8,10}; 
    i2 = new indexed_indexstruct(len,idxs);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) ); // [1,10] & [4,8,10] => i3 = [4,8,10]
    REQUIRE( !i3->is_empty() );
    CHECK( i3->is_indexed() );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==10 );
    CHECK( i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );

    REQUIRE_NOTHROW( i4 = i3->relativize(i1) ); // [4,8,10] in [1:10] i3 is indexed
    CHECK( i4->is_indexed() );
    CHECK( i4->first_index()==3 );
    CHECK( i4->last_index()==9 );
  }

  SECTION( "idx-idx" ) {
    indexstruct *i5;
    int len=3; index_int idx[3] = {4,8,11};
    i2 = new indexed_indexstruct(len,idx);
    int lenx=5; index_int idxs[5] = {3,8,10,11,12};
    i3 = new indexed_indexstruct(lenx,idxs);
    REQUIRE_NOTHROW( i4 = i2->intersect(i3) );
    REQUIRE( i4!=NULL );
    CHECK( i4->local_size()==2 );
    CHECK( i4->is_indexed() );
    CHECK( i4->first_index()==8 );
    CHECK( i4->last_index()==11 );
    CHECK( !i2->contains(i3) );
    CHECK( i2->contains(i4) );
    CHECK( i3->contains(i4) );

    REQUIRE_THROWS( i5 = i3->relativize(i2) );
    REQUIRE_THROWS( i5 = i2->relativize(i3) );
    REQUIRE_NOTHROW( i5 = i4->relativize(i2) ); // [8,11] in [4,8,11]
    CHECK( i5->is_indexed() );
    CHECK( i5->first_index()==1 );
    CHECK( i5->last_index()==2 );
    REQUIRE_NOTHROW( i5 = i4->relativize(i3) ); // [8,11] in [3,8,10,11,12]
    CHECK( i5->is_indexed() );
    CHECK( i5->first_index()==1 );
    CHECK( i5->last_index()==3 );
  }
}

TEST_CASE( "indexstruct unions","[indexstruct][union][4]" ) {
  indexstruct *i1,*i2,*i3;
  
  SECTION( "convert from stride 1" ) {
    i1 = new contiguous_indexstruct(2,10);
    CHECK( i1->is_contiguous() );
    CHECK_NOTHROW( i2 = i1->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->first_index()==2 );
    CHECK( i2->last_index()==10 );
    CHECK( i2->local_size()==i1->local_size() );

    REQUIRE_NOTHROW( delete i2 );
  }

  SECTION( "convert from stride 2" ) {
    //i1 = new indexstruct(2,10,2);
    i1 = new strided_indexstruct(2,10,2);
    CHECK_NOTHROW( i2 = i1->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->first_index()==2 );
    CHECK( i2->last_index()==10 );
    CHECK( i2->local_size()==i1->local_size() );

    REQUIRE_NOTHROW( delete i2 );
  }

  SECTION( "cont-cont" ) {
    i1 = new contiguous_indexstruct(1,10);
    i2 = new contiguous_indexstruct(5,12);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=nullptr );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==12 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new contiguous_indexstruct(11,13);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==13 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new contiguous_indexstruct(12,13);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=nullptr );
    CHECK( !i3->is_indexed() );
    CHECK( i3->is_composite() );
    CHECK( i3->local_size()==(i1->local_size()+i2->local_size()) );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==13 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    REQUIRE_NOTHROW( delete i1 );
  }

  SECTION( "cont-idx giving indexed" ) {
    i1 = new contiguous_indexstruct(5,8);
    i2 = new strided_indexstruct(8,12,2);
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( !i3->is_contiguous() );
    CHECK( i1->local_size()==4 );
    CHECK( i2->local_size()==3 );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

  SECTION( "cont-idx extending" ) {
    i1 = new contiguous_indexstruct(5,11);
    i2 = new strided_indexstruct(8,12,2);
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->local_size()==3 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i1->local_size()==7 );
    CHECK( i3->is_contiguous() );
    CHECK( i3->local_size()==8 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

  SECTION( "cont-idx extending2" ) {
    i1 = new contiguous_indexstruct(5,11);
    i2 = new strided_indexstruct(6,12,2);
    CHECK( i2->local_size()==4 );
    REQUIRE_NOTHROW( i2 = i2->convert_to_indexed() );
    CHECK( i2->is_indexed() );
    CHECK( i2->local_size()==4 );
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i1->local_size()==7 );
    CHECK( i3->is_contiguous() );
    CHECK( i3->local_size()==8 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

  SECTION( "idx-cont" ) {
    i1 = new strided_indexstruct(8,12,2);
    //    i1->convert_to_indexed();
    i2 = new contiguous_indexstruct(5,8);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

}

indexstruct *itimes2(index_int i) { return new contiguous_indexstruct(2*i,2*i); }

TEST_CASE( "structs and operations","[indexstruct][operate][6]" ) {
  indexstruct
    *i1,
    *i2;
  ioperator *op;

  SECTION( "multiply by constant" ) {
    i1 = new contiguous_indexstruct(5,10);
    REQUIRE_NOTHROW( op = new ioperator("*2") );
    REQUIRE_THROWS( op->operate(1,&i2) );
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    // printf("i2 s/b strided: <<%s>>\n",i2->as_string().data());
    // VLE CHECK( i2->is_strided() );
    CHECK( i2->first_index()==10 );
    CHECK( i2->last_index()==20 );
    CHECK( i2->local_size()==i1->local_size() );
    // CHECK( i2->is_strided() );
    // CHECK( i2->stride()==2 );
  }

  SECTION( "multiply by function" ) {
    i1 = new contiguous_indexstruct(5,10);
    REQUIRE_NOTHROW( op = new ioperator(&itimes2) );
    REQUIRE_NOTHROW( op->operate(1,&i2) );
    CHECK( i2->is_contiguous() );
    CHECK( i2->first_index()==2 );
    CHECK( i2->last_index()==2 );

    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    // printf("i2 s/b strided: <<%s>>\n",i2->as_string().data());
    // VLE CHECK( i2->is_strided() );
    CHECK( i2->first_index()==10 );
    CHECK( i2->last_index()==20 );
  }

  SECTION( "shift strided" ) {
    i1 = new strided_indexstruct(1,10,2);
    CHECK( i1->first_index()==1 );
    CHECK( i1->last_index()==9 );
    CHECK( i1->local_size()==5 );
    SECTION( "bump" ) {
      REQUIRE_NOTHROW( op = new ioperator("<=1") );
    }
    SECTION( "mod" ) {
      REQUIRE_NOTHROW( op = new ioperator("<<1") );
    }
    REQUIRE_NOTHROW( i2 = i1->operate(op) );
    CHECK( i2->first_index()==0 );
    CHECK( i2->last_index()==8 );
    CHECK( i2->local_size()==5 );
  }

  SECTION( "shift strided with truncate" ) {
    i1 = new strided_indexstruct(0,10,2);
    REQUIRE_NOTHROW( op = new ioperator("<=1") );
    REQUIRE_NOTHROW( i2 = i1->operate(op,0,100) );
    CHECK( i2->first_index()==1 );
    CHECK( i2->last_index()==9 );
    CHECK( i2->local_size()==5 );
  }
}

TEST_CASE( "division operation","[indexstruct][operate][7]" ) {
  indexstruct *i; ioperator *op;

  SECTION( "simple division" ) {
    REQUIRE_NOTHROW( op = new ioperator("/2") );
  
    SECTION( "contiguous1" ) {
      i = new contiguous_indexstruct(0,10);
      REQUIRE_NOTHROW( i = i->operate(op) );
      CHECK( i->first_index()==0 );
      CHECK( i->last_index()==5 );
    }
    SECTION( "contiguous2" ) {
      i = new contiguous_indexstruct(0,9);
      REQUIRE_NOTHROW( i = i->operate(op) );
      CHECK( i->first_index()==0 );
      CHECK( i->last_index()==4 );
    }
  }
  SECTION( "contiguous division" ) {
    REQUIRE_NOTHROW( op = new ioperator(":2") );
  
    SECTION( "contiguous1" ) {
      i = new contiguous_indexstruct(0,10);
      REQUIRE_NOTHROW( i = i->operate(op) );
      CHECK( i->first_index()==0 );
      CHECK( i->last_index()==4 );
    }
    SECTION( "contiguous2" ) {
      i = new contiguous_indexstruct(0,9);
      REQUIRE_NOTHROW( i = i->operate(op) );
      CHECK( i->first_index()==0 );
      CHECK( i->last_index()==4 );
    }
  }
}

TEST_CASE( "composite indexstruct","[indexstruct][composite][8]" ) {
  indexstruct *i1,*i2,*icomp;
  SECTION( "two contiguous" ) {
    i1 = new contiguous_indexstruct(3,5);
    i2 = new contiguous_indexstruct(10,12);
    SECTION( "right away" ) {
      REQUIRE_NOTHROW( icomp = new composite_indexstruct(i1) );
      REQUIRE_NOTHROW( icomp->union_with(i2) );
    }
    SECTION( "reverse away" ) {
      REQUIRE_NOTHROW( icomp = new composite_indexstruct(i2) );
      REQUIRE_NOTHROW( icomp->union_with(i1) );
    }
    CHECK( !icomp->is_contiguous() );
    CHECK( icomp->first_index()==3 );
    CHECK( icomp->last_index()==12 );
    CHECK( icomp->local_size()==6 );
  }
}

TEST_CASE( "enumerating indexstructs","[9]" ) {
  indexstruct *idx; int count,cnt=0;
  
  SECTION( "contiguous" ) {
    idx = new contiguous_indexstruct(3,5);
    count = 3;
    for (auto i=idx->begin(); i!=idx->end(); ++i) {
      CHECK( **i==count );
      CHECK( idx->get_ith_element(cnt++)==(**i) );
      count++;
    }
  }
  SECTION( "strided" ) {
    idx = new strided_indexstruct(3,10,2);
    CHECK( idx->first_index()==3 );
    CHECK( idx->last_index()==9 );
    count = 3;
    for (auto i=idx->begin(); i!=idx->end(); ++i) {
      CHECK( **i==count );
      CHECK( idx->get_ith_element(cnt++)==(**i) );
      count += 2;
    }
  }
  SECTION( "indexed" ) {
    index_int *ar = new index_int[4]{2,3,5,8};
    idx = new indexed_indexstruct(4,ar);
    count = 0;
    for (auto i=idx->begin(); i!=idx->end(); ++i) {
      CHECK( **i==ar[count] );
      CHECK( idx->get_ith_element(cnt++)==(**i) );
      count++;
    }
  }
}

// TEST_CASE( "ith element","[indexstruct][10]" ) {
//   indexstruct *i;

//   SECTION( "contiguous" ) {
//     i = new contiguous_indexstruct(2,5);
//     CHECK( i->get_ith_element(
//   }
// }
