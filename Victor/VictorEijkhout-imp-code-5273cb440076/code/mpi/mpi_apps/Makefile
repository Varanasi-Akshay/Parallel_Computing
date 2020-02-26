################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### MPI applications
####
################################################################

info ::
	@echo "MPI applications in the IMP model"
	@echo "Available commands:"

MODE = mpi
info ::
	@echo "make EXECUTABLE (where EXECUTABLES=${EXECUTABLES})"
EXECUTABLES := power cg cg0 cgr cgm gropp nbody heat lulesh-2d lulesh kmeans sstep \
    laplace-bilinear grids balance

.SECONDEXPANSION:
include ../../imp/Make.inc
include ../../imp/Make.common
IMPCODEDIR = ../../imp
OMPCODEDIR = ../../omp
MPICODEDIR = ../../mpi
include ../../imp/Make.imp
include ../../imp/Make.apps

CXXFLAGS = ${CXX_OPTFLAGS} -I. -I.. -I../../imp -DIMPisMPI -I${CPPFORMAT_INC}
OPTLEVEL = 2
%.o : %.cxx
	${MPICXX} ${CXXFLAGS} ${MPIFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} -c $<

EXECUTABLE_SOURCES := ${patsubst %,%.cxx,${EXECUTABLES}}
EXECUTABLE_OBJECTS := ${patsubst %,%.o,${EXECUTABLES}}

EXECUTABLE_OBJS = \
    ${MPICODEDIR}/mpi_base.o ${MPICODEDIR}/mpi_functions.o ${APPS_OBJS}
${EXECUTABLES} : $$@.cxx $$@.o ${EXECUTABLE_OBJS} ${ALL_SPECIFIC_OBJECTS}
	${MPICXX} ${CXXFLAGS} -o $@ ${MPIFLAGS} $@.o \
	    ${EXECUTABLE_OBJS} ${SPECIFIC_OBJECTS_$@} \
	    ${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}

# non-trivial dependencies
${EXECUTABLE_OBJECTS} ../mpi_base.o : \
    ../mpi_base.h ../../imp/imp_base.h \
    ../mpi_ops.h ../../imp/imp_ops.h \
    ../../imp/utils.h ../../imp/template_common_header.h ${MODE}_specific_header.h

##
## for regression testing
##
.PHONY: runapp
NP = 4
info ::
	@echo "     [NP=nnn, default=${NP}]"
APP  = heat
runapp :
	@mpirun -np ${NP} ${APP} -summary

clean ::
	@/bin/rm -f ${EXECUTABLES} ${EXECUTABLE_SOURCES} *.stf.* *.o[0-9]* \
	  *.itapr *.prot *.stf stampedetest.o* vncserver.out
	@/bin/rm -rf *.dSYM
