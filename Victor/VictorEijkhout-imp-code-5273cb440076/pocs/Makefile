################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Proof-of-concept codes
####
################################################################

info ::

OMPFLAGS_intel    = -openmp
OMPFLAGS_gcc      = -fopenmp
OMPFLAGS = ${OMPFLAGS_${TACC_FAMILY_COMPILER}}

ifdef TACC_TAU_DIR
  CC = tau_cc.sh
else
  CC = mpicc
endif
ifdef TACC_ITAC_DIR
  TRACE_OPTIONS = -trace
endif

info ::
	@echo "make program, where program in ${PROGRAMS}"
PROGRAMS = hello transpose \
    sstep0 sstep1 sstep2 \
    ${OVERLAPS}
OVERLAPS = overlap1 overlap2 overlap3 overlap4
info ::
	@echo "    [OPTLEVEL=... (default: ${OPTLEVEL})]"
OPTLEVEL = 2
info ::
	@echo "    [DFLAGS= ... compiler defines]"
DFLAGS =
% : %.c
	${CC} -O${OPTLEVEL} -g -std=c99 ${OMPFLAGS} ${TRACE_OPTIONS} ${DFLAGS} \
	    -lm \
	    -o $@ $*.c
.PHONY: overlaps
info ::
	@echo "make overlaps : overlap codes"

overlaps : ${OVERLAPS}
${OVERLAPS} : laptypes.c lapalloc.c
.PHONY: overlaprun
info ::
	@echo "make overlaprun"
overlaprun :
	@( \
	for s in 1000 3000 10000 30000 ; do \
	  for t in 1 10 100 ; do \
	    echo "Testing overlaps with N=$$s and thick=$$t" ; \
	    ( make clean ; make overlaps DFLAGS="-DN=$$s -DTHICK=$$t" ) >/dev/null 2>&1 ; \
	    for o in 2 3 4 ; do \
	      ibrun overlap$$o ; \
	    done ; \
	  done ; \
	done \
	) 2>run.err | tee run.out

info ::
	@echo "make vtrun PROGRAM=<any prog> OPTIONS=<whatever>"
vtrun :
	@if [ -z "${PROGRAM}" ] ; then \
	   echo "Usage: make ibrun program option"; exit 1 ; fi
	@if [ ${TACC_FAMILY_MPI} = mvapich2 ] ; then \
	   echo "Needs impi for itac tracing" ; exit 1 ; fi
	VT_LOGFILE_FORMAT=STFSINGLE ibrun -trace ${PROGRAM} ${OPTIONS}
info ::
	@echo "make idevrun PROGRAM=<any prog> OPTIONS=<whatever>"
idevrun :
	@if [ -z "${PROGRAM}" ] ; then \
	   echo "Usage: make ibrun program option"; exit 1 ; fi
	@export TAU_EXT="_${PROGRAM}" ; \
	export TAU_DUMP_DIR=`pwd`/tautrace$$TAU_EXT ; \
	  rm -rf $${TAU_DUMP_DIR}; mkdir -p $${TAU_DUMP_DIR} ; \
	  TAU_TRACE=1 TAU_PROFILE=1 \
	  TRACEDIR=$${TAU_DUMP_DIR} \
	  PROFILEDIR=$${TAU_DUMP_DIR} \
	    ibrun ${PROGRAM} ${OPTIONS}

info ::
	@echo "make tau PROGRAM=... (this does the post-processing)"
tau :
	@export HOME=`pwd` ; \
	export TAU_EXT=`if [ ! -z "${PROGRAM}" ] ; then echo "_" ; fi`${PROGRAM} ; \
	export TAU_DUMP_DIR=`pwd`/tautrace$$TAU_EXT ; \
	  cd $$TAU_DUMP_DIR ; \
	  echo ".. analyzing files in <$$TAU_DUMP_DIR>" ; \
	  rm -f tau.trc tau.edf ; \
	  tau_treemerge.pl ; \
	  tau2slog2 tau.trc tau.edf -o $$HOME/taulog$$TAU_EXT.slog2

.PHONY: clean
info ::
	@echo "make clean"
clean :
	@/bin/rm -rf *~ *.o *.o[0-9]* *.dSYM *.stf.* tautrace* *.prot
	@/bin/rm -f ${PROGRAMS} 
