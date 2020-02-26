# -*- Makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-6
####
#### Unit tests for the MPI+OpenMP backend of IMP
####
################################################################
info ::
	@echo "MPI+OMP simulation of an IMP backend code"
	@echo "through direct product of the two"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
include ../imp/Make.imp

CXX = mpicxx
CXXFLAGS = ${CXX_OPTFLAGS} -I${IMPCODEDIR} -I${OMPCODEDIR} -I${MPICODEDIR} -DIMPisPRODUCT -I${CPPFORMAT_INC}
.SECONDEXPANSION:
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${MPIFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${UNITTESTS})"

.PHONY: unittests
unittests : ${UNITTESTS}
UNITTESTS = unittest_operator unittest_collective \
  unittest_distribution unittest_struct unittest_ops
UNITTEST_OBJECTS = ${patsubst %,%.o,${UNITTESTS}}
include ../imp/Make.apps
include ../imp/Make.unit
UNITTEST_OBJS += \
    product_base.o ${MPICODEDIR}/mpi_base.o ${OMPCODEDIR}/omp_base.o \
    ${OMPCODEDIR}/omp_unittest_functions.o
# unittest_functions.o 
BASEHS = product_base.h ${MPICODEDIR}/mpi_base.h ${OMPCODEDIR}/omp_base.h ../imp/imp_base.h
${patsubst %,%.o,${UNITTESTS}} product_base.o : ${BASEHS} \
    product_ops.h
${UNITTESTS} : unittest_main.o $$@.o ${UNITTEST_OBJS} ${APPS_OBJS}
	${CXX} ${CXXFLAGS} ${CXXUNITFLAGS} ${OMPFLAGS} ${MPIFLAGS} \
		-o $@ ${PRODUCTFLAGS} $^ \
		${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}
${UNITTEST_OBJS} : product_base.h ${MPICODEDIR}/mpi_base.h ${OMPCODEDIR}/omp_base.h \
    ../imp/imp_functions.h ../imp/imp_base.h ../imp/utils.h

${OMPCODEDIR}/omp_unittest_functions.o : ${OMPCODEDIR}/omp_unittest_functions.cxx ../imp/unittest_functions.h
	cd ${OMPCODEDIR} ; make omp_unittest_functions.o
${UNITTEST_OBJS} : ../imp/imp_base.h ../imp/utils.h

unittest_main.o : product_static_vars.h
unittest_operator.o : ../imp/unittest_operator.cxx

include ../imp/Make.regress
BROKENOPTIONS = ~[modulo]
info ::
	@echo "make unittesting [UNITTESTS=list of apps] [NP=nnnn default=${NP}]"
	@echo "     regression (all unit tests)"
NP = 4
REGRESSION_PROCS = 4
REGRESSIONTESTS = unittest_distribution unittest_struct unittest_operator
unittest :
	@export u=${UNITTEST} ; \
	  echo "\n================Testing" $$u ; \
	  rm -f $$u ; make $$u >/dev/null 2>&1 ; \
	  if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi ; \
	  ${MPIRUN} -np ${REGRESSION_PROCS} $$u ${BROKENOPTIONS}${REGRESSIONOPTIONS}"~[hide]" \
	    2>&1 | grep "\(passed\|failed\)" \
	    > unittest.out 2>&1 ; \
	  cat unittest.out ; \

clean ::
	@/bin/rm -f unittest
