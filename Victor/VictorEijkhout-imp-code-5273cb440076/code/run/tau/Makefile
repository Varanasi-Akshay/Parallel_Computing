################################################################
################################################################
####
#### This makefile is part of the source of 
#### IMP
#### by Victor Eijkhout, copyright 2013-6
####
#### This is the makefile for the tau runs
####
################################################################
################################################################

info ::
	@echo "Do: make app (where app=${APPS})"

APPS = sstep

${APPS} :
	cd ../../mpi/mpi_apps ; \
	  make $@

ifdef TACC_TAU_DIR
  CC = tau_cc.sh
  CXX = tau_cxx.sh
else
  CC  = mpicc -std=c99
  CXX = mpicxx
endif

DEBUG=OPT
OPTLEVEL=2
OPT_flag = -O${OPTLEVEL} -g
DEBUG_flag = -O0 -g

info ::
	@echo "    debug flag: DEBUG=DEBUG/OPT (default: ${DEBUG})]"
	@echo "    [CORESPERNODE=nnn (default: ${CORESPERNODE})]"
CORESPERNODE = 16
%.o : %.c
	${CC} ${${DEBUG}_flag} -D${DEBUG} -DCORES_PER_NODE=${CORESPERNODE} \
	  -c $*.c
%.o : %.cxx
	${CXX} ${${DEBUG}_flag} -D${DEBUG} -DCORES_PER_NODE=${CORESPERNODE} \
	  -c $*.cxx

# rules for submitting to the queue and doing tau analysis
info ::
	@echo
	@echo "make submit EXECUTABLE=<any prog> OPTIONS=<whatever>"
	@echo "     do qsub jobscript, where the jobscript does"
	@echo "     ibrun EXECUTABLE OPTIONS"
	@echo "     with the tau environment variables set to catch the trace"
	@echo "     files in a directory named tautrace_EXECUTABLE"
EXECUTABLE = init
OPTIONS = 
submit :
	@export TAU_EXT=`if [ ! -z "${EXECUTABLE}" ] ; then echo "_" ; fi`${EXECUTABLE} ; \
	export TAU_DUMP_DIR=`pwd`/tautrace$$TAU_EXT ; \
	  echo "tau output to: <$$TAU_DUMP_DIR>" ; \
	  rm -rf $${TAU_DUMP_DIR}; mkdir -p $${TAU_DUMP_DIR} ; \
	  TAU_TRACE=1 TAU_PROFILE=1 \
	  TRACEDIR=$${TAU_DUMP_DIR} \
	  PROFILEDIR=$${TAU_DUMP_DIR} \
	    EXECUTABLE=${EXECUTABLE} OPTIONS="${OPTIONS}" \
	      sbatch jobscript

info ::
	@echo "make idevrun EXECUTABLE=<any prog> [OPTIONS=<whatever>]"
	@echo "     do ibrun EXECUTABLE OPTIONS"
	@echo "     with the tau environment variables set to catch the trace"
	@echo "     files in a directory named tautrace_EXECUTABLE"
idevrun :
	@if [ -z "${EXECUTABLE}" ] ; then \
	   echo "Usage: make idevrun EXECUTABLE=... [OPTIONS=...]"; exit 1 ; fi
	@export TAU_EXT="_${EXECUTABLE}" ; \
	export TAU_DUMP_DIR=`pwd`/tautrace$$TAU_EXT ; \
	  rm -rf $${TAU_DUMP_DIR}; mkdir -p $${TAU_DUMP_DIR} ; \
	  TAU_TRACE=1 TAU_PROFILE=1 \
	  TRACEDIR=$${TAU_DUMP_DIR} \
	  PROFILEDIR=$${TAU_DUMP_DIR} \
	    ibrun ${EXECUTABLE} ${OPTIONS}
info ::
	@echo "make tau EXECUTABLE=..."
	@echo "     do a tau postprocessing of running EXECUTABLE;"
	@echo "     this leaves a file taulog_EXECUTABLE.slog2"
tau :
	@export HOME=`pwd` ; \
	export TAU_EXT=`if [ ! -z "${EXECUTABLE}" ] ; then echo "_" ; fi`${EXECUTABLE} ; \
	export TAU_DUMP_DIR=`pwd`/tautrace$$TAU_EXT ; \
	  cd $$TAU_DUMP_DIR ; \
	  echo ".. analyzing files in <$$TAU_DUMP_DIR>" ; \
	  rm -f tau.trc tau.edf ; \
	  tau_treemerge.pl ; \
	  tau2slog2 tau.trc tau.edf -o $$HOME/taulog$$TAU_EXT.slog2

clean ::
	@rm -f *.dat
