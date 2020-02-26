################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Proof-of-concept code for overlapping computation communication
####
################################################################

info ::

ifdef TACC_TAU_DIR
  CC = tau_cc.sh
else
  CC = mpicc
endif
ifdef TACC_ITAC_DIR
  TRACE_OPTIONS = -trace
endif

OVERLAPS = \
    overlap1 overlap2 overlap2f \
    overlap3 overlap4 \
    overlap4h overlap4x
OTHER_OVERLAPS =  overlap1p overlap3 overlap4 overlap4t overlap4p

PROGRAMS = ${OVERLAPS} ${OTHER_OVERLAPS}
info ::
	@echo "make program, where program in ${PROGRAMS}"
info ::
	@echo "    [OPTLEVEL=... (default: ${OPTLEVEL})]"
OPTLEVEL = 2
info ::
	@echo "    [DFLAGS= ... compiler defines]"
DFLAGS =
% : %.c
	@echo "Compiling with `which ${CC}`"
	${CC} -O${OPTLEVEL} -g -std=c99 -fopenmp \
	    ${TRACE_OPTIONS} ${DFLAGS} \
	    -o $@ $*.c \
	    -lm
.PHONY: overlaps listoverlaps listsizes
info ::
	@echo "make overlaps : make all overlap codes"
	@echo "make listoverlaps : list all overlap codes"
	@echo "make listsizes MACHINE=... : list domain sizes"
overlaps : ${OVERLAPS}
listoverlaps :
	@echo ${OVERLAPS}
${OVERLAPS} : lapdefs.c laptypes.c lapalloc.c lapwrite.c \
    post1.c post2.c update1.c update2.c \
    faketypes.c
listsizes :
	@if [ "${TACC_SYSTEM}" = "stampede" ] ; then \
	  echo "1000 3000 9000" ; \
	else \
	  echo "1000 3000 10000" ; \
	fi

.PHONY: overlaprun
info ::
	@echo "make overlaprun"
overlaprun :
	@( \
	for s in 1000 3000 10000 ; do \
	  for t in 1 10 100 ; do \
	    echo "Testing overlaps with N=$$s and thick=$$t" ; \
	    ( make clean ; make overlaps DFLAGS="-DN=$$s -DTHICK=$$t" ) >/dev/null 2>&1 ; \
	    for o in 1 2 3 4 ; do \
	      ibrun overlap$$o ; \
	    done ; \
	  done ; \
	done \
	) 2>run.err | tee overlap.${TACC_SYSTEM}.out
	sed -i -e '/TACC/d' -e '/check/d' overlap.${TACC_SYSTEM}.out

.PHONY: charts bigcharts tables
info ::
	@echo "make charts,bigcharts,tables [MACHINE=... (default: ${MACHINE})]"
MACHINE = stampede2
charts :
	cd ${MACHINE} && \
	  for o in *.out ; do \
	    python ../chart.py $$o ; \
	  done
bigcharts :
	cd ${MACHINE} && \
	  for r in ${OVERLAPS} ; do \
	    python ../chart.py ${MACHINE} $$r ; \
	  done
tables :
	cd ${MACHINE} && \
	  python ../tabulate.py ${MACHINE}

.PHONY: clean
info ::
	@echo "make clean"
clean :
	@/bin/rm -rf *~ *.o *.dSYM *.stf.* tautrace* *.prot *test.e* laptime*.out \
	    overlap*.e[0-9]*
	@/bin/rm -f ${PROGRAMS} overlap*-*-*
	@for o in `make listoverlaps` ; do \
	  rm -f $$o-* ; \
	 done
