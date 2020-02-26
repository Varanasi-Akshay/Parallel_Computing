################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014/5
####
#### OpenMP applications
####
################################################################

info ::
	@echo "OMP applications in the IMP model"
	@echo "Available commands:"

MODE = omp
info ::
	@echo "make EXECUTABLE (where EXECUTABLES=${EXECUTABLES})"
EXECUTABLES := threepoint kmeans power cg heat nbody lulesh

.SECONDEXPANSION:
include ../../imp/Make.inc
include ../../imp/Make.common
IMPCODEDIR = ../../imp
OMPCODEDIR = ../../omp
MPICODEDIR = ../../mpi
include ../../imp/Make.imp
include ../../imp/Make.apps

CXX = mpicxx
CXXFLAGS = ${CXX_OPTFLAGS} -I. -I.. -I../../imp -DIMPisOMP -I${CPPFORMAT_INC}
OPTLEVEL = 2
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

EXECUTABLE_SOURCES := ${patsubst %,%.cxx,${EXECUTABLES}}
EXECUTABLE_OBJECTS := ${patsubst %,%.o,${EXECUTABLES}}

EXECUTABLE_OBJS = \
    ${OMPCODEDIR}/omp_base.o ${OMPCODEDIR}/omp_functions.o ${APPS_OBJS}
${EXECUTABLES} : $$@.cxx $$@.o ${EXECUTABLE_OBJS} ${ALL_SPECIFIC_OBJECTS}
	${CXX} ${CXXFLAGS} -o $@ ${OMPFLAGS} $@.o \
	    ${EXECUTABLE_OBJS} ${SPECIFIC_OBJECTS_$@} \
	    ${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}

# non-trivial dependencies
${EXECUTABLE_OBJECTS} ../omp_base.o : \
    ../omp_base.h ../../imp/imp_base.h \
    ../omp_ops.h ../../imp/imp_ops.h \
    ../../imp/utils.h ../../imp/template_common_header.h ${MODE}_specific_header.h

##
## for regression testing
##
.PHONY: runapp
APP  = heat
NTHREADS = 4
info ::
	@echo "     [NTHREADS=nnn, default=${NTHREADS}]"
runapp :
	@OMP_NUM_THREADS=${NTHREADS} ./${APP} -summary

clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES}
	@/bin/rm -rf *.dSYM
