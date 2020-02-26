#include <stdlib.h>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "imp_base.h"
#include "imp_base.cxx"

TEST_CASE( "contiguous indexstruct","[indexstruct][1]" ) {

  indexstruct *i1,*i2,*i3,*i4;

  SECTION( "basic construction" ) {
    // type testing
    i1 = new indexstruct(0,5);
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

    i1 = new indexstruct(1,7);
    CHECK( i1->is_contiguous() );
    CHECK( !i1->is_indexed() );
    CHECK( i1->first_index()==1 );
    CHECK( i1->last_index()==7 );
    CHECK( i1->stride()==1 );
    REQUIRE_THROWS( i1->find(0) );
    REQUIRE_NOTHROW( delete i1 );

    i1 = new indexstruct(2,6,2);
    CHECK( i1->is_contiguous() );
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

  SECTION( "striding and operations" ) {
    // allowed to specify an upper bound when striding,
    // but stored is the actually last element
    i1 = new indexstruct(4,7,2);

    SECTION( "basic stride tests" ) {
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==4 );
      CHECK( i1->last_index()==6 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );

      i2 = new indexstruct(4,6);
      CHECK( !i1->equals(i2) );
      CHECK( !i2->equals(i1) );
      REQUIRE_NOTHROW( delete i2 );

      i2 = new indexstruct(4,6,2);
      CHECK( i1->equals(i2) );
      CHECK( i2->equals(i1) );
      REQUIRE_NOTHROW( delete i2 );

      CHECK( i1->can_service(4) );
      CHECK( !i1->can_service(5) );
      CHECK( i1->can_service(6) );
      CHECK( !i1->can_service(7) );
    }

    SECTION( "strided containment" ) {
      i2 = new indexstruct(4,8);
      i3 = new indexstruct(4,8,2);
      i4 = new indexstruct(4,8,4);
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
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==5 );
      CHECK( i1->last_index()==7 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation backward" ) {
      i1->translate_by(-2);
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==2 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    SECTION( "translation through zero" ) {
      i1->translate_by(-5);
      CHECK( i1->is_contiguous() );
      CHECK( !i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==1 );
      CHECK( i1->local_size()==2 );
      CHECK( i1->stride()==2 );
    }

    REQUIRE_NOTHROW( delete i1 );
  }

}

TEST_CASE( "indexed indexstruct","[indexstruct][2]" ) {

  indexstruct *i1,*i2,*i3;

  SECTION( "basic construction" ) {
    SECTION( "correct" ) {
      int len=3; index_int idx[3] = {1,2,4};
      i1 = new indexstruct(len,idx);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
      REQUIRE_NOTHROW( delete i1 );
    }

    SECTION( "unsorted throws an error" ) {
      int len=4; index_int idx[4] = {1,2,6,4};
      CHECK_THROWS( i1 = new indexstruct(len,idx) );
    }

    SECTION( "negative indices allowed" ) {
      int len=3; index_int idx[3] = {-1,2,4};
      i1 = new indexstruct(len,idx);
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==-1 );
      CHECK( i1->last_index()==4 );
      CHECK( i1->local_size()==len );
      REQUIRE_NOTHROW( delete i1 );
    }

  }

  SECTION( "striding and operations" ) {
    int len=5; index_int idx[5] = {1,2,4,7,9};
    indexstruct *i1;
    REQUIRE_NOTHROW( i1 = new indexstruct(len,idx) );

    SECTION( "basic stride tests" ) {
      CHECK( !i1->is_contiguous() );
      CHECK( i1->is_indexed() );
      CHECK( i1->first_index()==1 );
      CHECK( i1->last_index()==9 );
      CHECK( i1->local_size()==len );

      CHECK( !i1->can_service(0) );
      CHECK( i1->can_service(1) );
      CHECK( i1->can_service(4) );
      CHECK( !i1->can_service(5) );
      CHECK( !i1->can_service(6) );
      CHECK( i1->can_service(7) );

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
  
  i1 = new indexstruct(1,10);
  SECTION( "cont-cont" ) {
    i2 = new indexstruct(5,12);
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
    
    i2 = new indexstruct(10,12); // [1,10] & [10,12] => [10,10]
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==10 );
    CHECK( i3->last_index()==10 );
    CHECK( !i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );
    REQUIRE_THROWS( i4 = i2->relativize(i1) );
    REQUIRE_NOTHROW( i4 = i3->relativize(i1) );
    CHECK( i4->is_contiguous() );
    CHECK( i4->first_index()==9 );
    CHECK( i4->last_index()==9 );
    REQUIRE_NOTHROW( i4 = i3->relativize(i2) );
    CHECK( i4->is_contiguous() );
    CHECK( i4->first_index()==0 );
    CHECK( i4->last_index()==0 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new indexstruct(11,12);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3==NULL );
    REQUIRE_NOTHROW( delete i2 );
  }

  SECTION( "cont-idx" ) {
    int len=3; index_int idx[3] = {4,8,11};
    i2 = new indexstruct(len,idx);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_indexed() );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==8 );
    CHECK( !i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );
    REQUIRE_THROWS( i2->relativize(i1) );

    len=3; index_int idxs[3] = {4,8,10}; // [1,10] & [4,8,10] => [4,8,10]
    i2 = new indexstruct(len,idxs);
    REQUIRE_NOTHROW( i3 = i1->intersect(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_indexed() );
    CHECK( i3->first_index()==4 );
    CHECK( i3->last_index()==10 );
    CHECK( i1->contains(i2) );
    CHECK( i1->contains(i3) );
    CHECK( i2->contains(i3) );

    REQUIRE_NOTHROW( i4 = i3->relativize(i1) );
    CHECK( i4->is_indexed() );
    CHECK( i4->first_index()==3 );
    CHECK( i4->last_index()==5 );
  }

  SECTION( "idx-idx" ) {
    indexstruct *i5;
    int len=3; index_int idx[3] = {4,8,11};
    i2 = new indexstruct(len,idx);
    int lenx=5; index_int idxs[5] = {3,8,10,11,12};
    i3 = new indexstruct(lenx,idxs);
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
    i1 = new indexstruct(2,10);
    CHECK( i1->is_contiguous() );
    CHECK_NOTHROW( i1->convert_to_indexed() );
    CHECK( i1->is_indexed() );
    CHECK( i1->first_index()==2 );
    CHECK( i1->last_index()==10 );
    CHECK( i1->local_size()==9 );

    REQUIRE_NOTHROW( delete i1 );
  }

  SECTION( "convert from stride 2" ) {
    i1 = new indexstruct(2,10,2);
    CHECK( i1->is_contiguous() );
    CHECK_NOTHROW( i1->convert_to_indexed() );
    CHECK( i1->is_indexed() );
    CHECK( i1->first_index()==2 );
    CHECK( i1->last_index()==10 );
    CHECK( i1->local_size()==5 );

    REQUIRE_NOTHROW( delete i1 );
  }

  SECTION( "cont-cont" ) {
    i1 = new indexstruct(1,10);
    i2 = new indexstruct(5,12);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==12 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new indexstruct(11,13);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_contiguous() );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==13 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    i2 = new indexstruct(12,13);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    REQUIRE( i3!=NULL );
    CHECK( i3->is_indexed() );
    CHECK( i3->local_size()==(i1->local_size()+i2->local_size()) );
    CHECK( i3->first_index()==1 );
    CHECK( i3->last_index()==13 );
    REQUIRE_NOTHROW( delete i2 );
    REQUIRE_NOTHROW( delete i3 );

    REQUIRE_NOTHROW( delete i1 );
  }

  SECTION( "cont-idx" ) {
    i1 = new indexstruct(5,8);
    i2 = new indexstruct(8,12,2);
    i2->convert_to_indexed();
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

  SECTION( "idx-cont" ) {
    i1 = new indexstruct(8,12,2);
    i1->convert_to_indexed();
    i2 = new indexstruct(5,8);
    REQUIRE_NOTHROW( i3 = i1->struct_union(i2) );
    CHECK( i3->local_size()==6 );
    CHECK( i3->first_index()==5 );
    CHECK( i3->last_index()==12);
  }

}

TEST_CASE( "indexstruct stashing and collapsing","[indexstruct][5]") {
  indexstruct *ii = new indexstruct();

  SECTION( "create in sequence" ) {
    REQUIRE_NOTHROW( ii->add_element(1) );
    REQUIRE_NOTHROW( ii->add_element(2) );
    CHECK( !ii->is_contiguous() );
    CHECK( !ii->is_indexed() );
    REQUIRE_NOTHROW( ii->collapse() );
    CHECK( ii->is_contiguous() );
    CHECK( ii->first_index()==1 );
    CHECK( ii->last_index()==2 );
  }

  SECTION( "create randomly" ) {
    REQUIRE_NOTHROW( ii->add_element(2) );
    REQUIRE_NOTHROW( ii->add_element(4) );
    REQUIRE_NOTHROW( ii->add_element(1) );
    REQUIRE_NOTHROW( ii->add_element(3) );
    CHECK( !ii->is_contiguous() );
    CHECK( !ii->is_indexed() );
    REQUIRE_NOTHROW( ii->collapse() );
    CHECK( ii->is_contiguous() );
    CHECK( ii->first_index()==1 );
    CHECK( ii->last_index()==4 );
  }

  SECTION( "create non-contiguous" ) {
    REQUIRE_NOTHROW( ii->add_element(2) );
    REQUIRE_NOTHROW( ii->add_element(4) );
    REQUIRE_NOTHROW( ii->add_element(1) );
    CHECK( !ii->is_contiguous() );
    CHECK( !ii->is_indexed() );
    REQUIRE_NOTHROW( ii->collapse() );
    CHECK( ii->is_indexed() );
    CHECK( ii->first_index()==1 );
    CHECK( ii->last_index()==4 );
    CHECK( ii->local_size()==3 );
  }

  SECTION( "another contiguous" ) {
    REQUIRE_NOTHROW( ii->add_element(10) );
    REQUIRE_NOTHROW( ii->add_element(12) );
    REQUIRE_NOTHROW( ii->add_element(11) );
    REQUIRE_NOTHROW( ii->add_element(13) );
    REQUIRE_NOTHROW( ii->add_element(12) );
    REQUIRE_NOTHROW( ii->add_element(14) );
    REQUIRE_NOTHROW( ii->collapse() );
    CHECK( ii->is_contiguous() );
    CHECK( !ii->is_indexed() );
    CHECK( ii->first_index()==10 );
    CHECK( ii->last_index()==14 );
    CHECK( ii->local_size()==5 );
  }

  SECTION( "another non-contiguous" ) {
    REQUIRE_NOTHROW( ii->add_element(20) );
    REQUIRE_NOTHROW( ii->add_element(22) );
    REQUIRE_NOTHROW( ii->add_element(21) );
    REQUIRE_NOTHROW( ii->add_element(23) );
    REQUIRE_NOTHROW( ii->add_element(22) );
    REQUIRE_NOTHROW( ii->add_element(24) );
    REQUIRE_NOTHROW( ii->add_element(26) );
    REQUIRE_NOTHROW( ii->collapse() );
    CHECK( !ii->is_contiguous() );
    CHECK( ii->is_indexed() );
    CHECK( ii->first_index()==20 );
    CHECK( ii->last_index()==26 );
    CHECK( ii->local_size()==6 );
  }

}

indexstruct *itimes2(index_int i) { return new indexstruct(2*i,2*i); }

TEST_CASE( "structs from functions","[indexstruct][operate][6]" ) {
  indexstruct
    *i1 = new indexstruct(5,10),
    *i2;
  ioperator *mul2;

  REQUIRE_NOTHROW( mul2 = new ioperator(&itimes2) );
  REQUIRE_NOTHROW( mul2->operate(1,&i2) );
  CHECK( i2->is_contiguous() );
  CHECK( i2->first_index()==2 );
  CHECK( i2->last_index()==2 );

  REQUIRE_NOTHROW( i2 = i1->operate(mul2) );
  CHECK( i2->is_contiguous() );
  CHECK( i2->first_index()==10 );
  CHECK( i2->last_index()==20 );

}

TEST_CASE( "copy indexstruct","[indexstruct][copy][7]" ) {
  indexstruct *i1 = new indexstruct(7,15), *i2;
  REQUIRE_NOTHROW( i2 = new indexstruct( *i1 ) );
  CHECK( i2->first_index()==7 );
  CHECK( i2->last_index()==15 );
  REQUIRE_NOTHROW( i1->translate_by(1) );
  CHECK( i1->first_index()==8 );
  CHECK( i1->last_index()==16 );
  CHECK( i2->first_index()==7 );
  CHECK( i2->last_index()==15 );  
}

TEST_CASE( "copy message","[message][copy][11]" ) {
  indexstruct *i1 = new indexstruct(11,13), *i2,*i3;
  message *m1,*m2;
  CHECK_NOTHROW( m1 = new message(1,5,i1) );

  //  CHECK_THROWS( i2 = m1->get_local_struct() ); // VLE why doesn't this throw?
  CHECK_NOTHROW( i2 = m1->get_global_struct() );
  CHECK( i2->first_index()==11 );
  CHECK( i2->last_index()==13 );

  REQUIRE_NOTHROW( m2 = new message( *m1 ) );
  CHECK_NOTHROW( i2->translate_by( 3 ) );
  CHECK( i2->first_index()==14 );
  CHECK( i2->last_index()==16 );
  
  CHECK_NOTHROW( i3 = m2->get_global_struct() );
  CHECK( i3->first_index()==11 );
  CHECK( i3->last_index()==13 );
}
