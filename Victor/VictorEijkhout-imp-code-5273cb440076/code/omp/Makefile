# -*- Makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-6
####
#### Unit tests for the OpenMP backend of IMP
####
################################################################
info ::
	@echo "OpenMP simulation of an IMP backend code"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
include ../imp/Make.imp

CXX = ${CXX_COMPILER}
CXXFLAGS = ${CXX_OPTFLAGS} -I../imp -I../omp -DIMPisOMP -I${CPPFORMAT_INC}
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${UNITTESTS})"

.PHONY: unittests
unittests : ${UNITTESTS}
UNITTESTS = unittest_distribution unittest_sparse \
    unittest_struct unittest_collective unittest_ir \
    unittest_ops unittest_cg unittest_nbody
UNITTEST_OBJECTS = ${patsubst %,%.o,${UNITTESTS}}
include ../imp/Make.apps
include ../imp/Make.unit
UNITTEST_OBJS += \
    omp_base.o omp_functions.o omp_unittest_functions.o
.SECONDEXPANSION:
# ${APPS_OBJS}
${UNITTESTS} : \
            ${UNITTEST_OBJS} ${ALL_SPECIFIC_OBJECTS} \
            unittest_main.o $$@.o
	${CXX} ${CXXFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -o $@ ${OMPFLAGS} \
	  unittest_main.o $@.o \
	  ${UNITTEST_OBJS} \
	  ${SPECIFIC_OBJECTS_$@} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}
#	${SYMUTIL:-/bin/true} $@
${UNITTEST_OBJS} : omp_base.h

omp_unittest_functions.o : ../imp/unittest_functions.h
${patsubst %,%.o,${UNITTESTS}} : \
  ../imp/imp_base.h ../imp/imp_ops.h \
  ../imp/unittest_functions.h \
  omp_base.h omp_ops.h omp_static_vars.h

${patsubst %,%.o,${UNITTESTS}} : omp_base.h ../imp/imp_base.h \
    ../imp/unittest_functions.h omp_static_vars.h

EXECUTABLES = threepoint 

.SECONDEXPANSION:
# threepoint executable
${EXECUTABLES} : \
    ../imp/imp_base.o ../imp/utils.o \
    threepoint_kernel.o threepoint_local.o \
    $$@.o omp_base.o 
	${CXX} -o $@ ${OMPFLAGS} $^
	@${SYMUTIL} $@

# non-trivial dependencies
threepoint.cxx : ../imp/template_threepoint.cxx
	cat $< | sed s/IMP_/omp_/g > $@
threepoint.o omp_base.o : omp_base.h ../imp/imp_base.h ../imp/utils.h
clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES}

include ../imp/Make.regress
BASICBROKENOPTIONS="~[mask]~[embed]~[modulo]~[multi]~[sync]~[extend]"
BROKENOPTIONS=
REGRESSIONTESTS = unittest_distribution unittest_sparse unittest_struct unittest_collective unittest_ops unittest_nbody unittest_cg
# unittest_operator unittest_ir unittest_ops unittest_cg"
UNITTEST = unittest_struct
unittest :
	@export u=${UNITTEST} ; \
	  echo "\n================\nTesting" $$u ; \
	  rm -f $$u ; make $$u >/dev/null 2>&1 ; \
	  if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi ; \
	  $$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
	    2>&1 | grep "\(passed\|failed\)" > unittest.out 2>&1 ; \
	  cat unittest.out ; \
	  ntests=$$(( ntests+1 ))

clean ::
	@/bin/rm -f unittest
