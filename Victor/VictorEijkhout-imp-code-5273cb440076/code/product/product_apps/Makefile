################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-6
####
#### Product applications
####
################################################################

info ::
	@echo "Product applications in the IMP model"
	@echo "Available commands:"

MODE = product
info ::
	@echo "make EXECUTABLE (where EXECUTABLES=${EXECUTABLES})"
EXECUTABLES := threepoint kmeans power cg gropp nbody heat lulesh

.SECONDEXPANSION:
include ../../imp/Make.inc
include ../../imp/Make.common
IMPCODEDIR = ../../imp
OMPCODEDIR = ../../omp
MPICODEDIR = ../../mpi
PRODUCTCODEDIR = ../../product
include ../../imp/Make.imp
include ../../imp/Make.apps

CXX = mpicxx
CXXFLAGS = ${CXX_OPTFLAGS} -I. -I.. -I../../imp -I../../omp -I../../mpi -DIMPisPRODUCT -I${CPPFORMAT_INC}
OPTLEVEL = 2
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${MPIFLAGS} ${OMPFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

EXECUTABLE_SOURCES := ${patsubst %,%.cxx,${EXECUTABLES}}
EXECUTABLE_OBJECTS := ${patsubst %,%.o,${EXECUTABLES}}

EXECUTABLE_OBJS = \
    ${PRODUCTCODEDIR}/product_base.o ${MPICODEDIR}/mpi_base.o ${OMPCODEDIR}/omp_base.o \
    ../../omp/omp_functions.o ${APPS_OBJS}
${EXECUTABLES} : $$@.o ${EXECUTABLE_OBJS} ${ALL_SPECIFIC_OBJECTS}
	${CXX} ${CXXFLAGS} -o $@ ${MPIFLAGS} ${OMPFLAGS} $@.o \
	    ${EXECUTABLE_OBJS} ${SPECIFIC_OBJECTS_$@} \
	    ${CPPFORMAT_LIB}/libformat.a ${LIBCXX}

BASEHS = ${PRODUCTCODEDIR}/product_base.h ../../mpi/mpi_base.h ../../omp/omp_base.h ../../imp/imp_base.h
${EXECUTABLE_OBJECTS} product_base.o : ${BASEHS} ../product_ops.h
template_specific_header.h : ../../imp/template_IMP_header.h
	@cat ../../imp/template_IMP_header.h | sed -e 's/IMP_/mpi_/g' > template_specific_header.h 

clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES} *.stf.* *.o[0-9]*
	@/bin/rm -rf *.dSYM
