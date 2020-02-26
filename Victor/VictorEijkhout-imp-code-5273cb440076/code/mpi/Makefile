# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Unit tests for the MPI backend of IMP
####
################################################################
info ::
	@echo "MPI simulation of an IMP backend code"
	@echo
	@echo "Available commands:"
	@echo

include ../imp/Make.inc
include ../imp/Make.common
IMPCODEDIR = ../imp
OMPCODEDIR = ../omp
MPICODEDIR = ../mpi
include ../imp/Make.imp

CXXFLAGS = ${CXX_OPTFLAGS} -I../imp -I${CPPFORMAT_INC}
%.o : %.cxx
	${MPICXX} ${CXXFLAGS} ${MPIFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

####
#### Unit tests
####
info ::
	@echo "make unittest_TYPE (choices: ${UNITTESTS})"

.PHONY: unittests
unittests : ${UNITTESTS}
UNITTESTS = \
  unittest_startup unittest_index unittest_proc unittest_sparse unittest_distribution \
  unittest_operator unittest_struct unittest_collective \
  unittest_over \
  unittest_kmeans unittest_lapack unittest_ops unittest_cg unittest_nbody \
  unittest_lulesh
UNITTEST_OBJECTS = ${patsubst %,%.o,${UNITTESTS}}
include ../imp/Make.apps
include ../imp/Make.unit
UNITTEST_OBJS += \
    mpi_base.o mpi_functions.o unittest_functions.o
SPECIFIC_OBJECTS_unittest_ops = ${IMPCODEDIR}/laplace_functions.o
.SECONDEXPANSION:
${UNITTESTS} ::  \
            ${UNITTEST_OBJS} ${SPECIFIC_OBJECTS_${@@}} ${ALL_SPECIFIC_OBJECTS} \
            unittest_main.o $$@.o
	@echo "unittests depend on ${ALL_SPECIFIC_OBJECTS}"
	${MPICXX} ${CXXFLAGS} ${CXXUNITFLAGS} -o $@ ${MPIFLAGS} \
	  unittest_main.o $@.o \
	  ${UNITTEST_OBJS} \
	  ${SPECIFIC_OBJECTS_$@} \
	  ${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}
${UNITTEST_OBJS} : mpi_base.h

${patsubst %,%.o,${UNITTESTS}} : \
    ${IMPCODEDIR}/unittest_functions.h ${IMPCODEDIR}/imp_functions.h ${IMPCODEDIR}/indexstruct.hpp \
    mpi_base.h ../imp/imp_base.h mpi_ops.h ../imp/imp_ops.h mpi_static_vars.h

${patsubst %,${IMPCODEDIR}/%.o,${UNITTESTS}} : \
            ${patsubst %,${IMPCODEDIR}/%.cxx,${UNITTESTS}}
	cd ${IMPCODEDIR} ; make $@

# this rule is slightly over generous
unittest_kmeans_main.o unittest_kmeans.o mpi_apps/mpi_kmeans_kernel.o : \
    ${IMPCODEDIR}/kmeans_functions.h

include ../imp/Make.regress
info ::
	@echo "    [REGRESSION_PROCS=nnnn default=${REGRESSION_PROCS}]"
NP = 4
REGRESSION_PROCS = 4
BASICBROKENOPTIONS = "~[mask]~[reuse]~[bin]~[modulo]~[embed]~[extend]"
REGRESSIONTESTS = unittest_proc unittest_distribution unittest_sparse \
    unittest_struct unittest_over \
    unittest_collective unittest_operator unittest_ops \
    unittest_nbody unittest_cg unittest_lulesh unittest_kmeans
# unittest_over 
info ::
	@echo "make unittest UNITTEST=..."
UNITTEST = unittest_struct
unittest :
	@export u=${UNITTEST} ; \
	  echo "\n================\nTesting" $$u ; \
	  rm -f $$u ; make $$u >/dev/null 2>&1 ; \
	  if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi ; \
	  ${MPIRUN} -np ${REGRESSION_PROCS} $$u ${BASICBROKENOPTIONS}${BROKENOPTIONS}"~[hide]" \
	    2>&1 | grep "\(passed\|failed\)" \
	    > unittest.out 2>&1 ; \
	  cat unittest.out

info ::
	@echo "make irtesting [IRFILTER=word]"
IRFILTER = ""
irtesting :
	/bin/rm -f *.ir ; \
	make unittest_ir && mpirun -np 4 unittest_ir `if [ ! -z ${IRFILTER} ] ; then echo [${IRFILTER}] ; fi` && \
	  for i in *.ir ; do echo "==== $$i"; cat $$i ; echo ; done

####
#### Executables: done in mpi_apps directory
####
info ::
	@echo "make applications : re-extract into mpi_apps directory"
.PHONY: applications
EXECUTABLES := threepoint kmeans norm power cg
applications :
	for a in ${EXECUTABLES} ; do \
	  cat ${IMPCODEDIR}/template_$$a.cxx | sed s/IMP_/mpi_/g > mpi_apps/mpi_$$a.cxx ; \
	  done

clean ::
	@/bin/rm -f unittest ${UNITTESTS}
	@cd mpi_apps ; make clean
