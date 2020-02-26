################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Proof-of-concept code for communication avoiding
####
################################################################

info ::

##
## set include packages
##
CPPFORMAT_INC = ${HOME}/Installation/cppformat/cppformat
CPPFORMAT_LIB = ${HOME}/Installation/cppformat/installation/lib
LIBFORMAT = libfmt.a

BALANCES = \
    balance balance2
OTHER_BALANCES =

PROGRAMS = ${BALANCES} ${OTHER_BALANCES}
info ::
	@echo "make program, where program in ${PROGRAMS}"
info ::
	@echo "    [OPTLEVEL=... (default: ${OPTLEVEL})]"
OPTLEVEL = 2
info ::
	@echo "    [DFLAGS= ... compiler defines]"
DFLAGS =
.SECONDEXPANSION:
%.o : %.cxx
	${CXX} -c $< \
	    -O${OPTLEVEL} -g -std=c++14 \
	    ${TRACE_OPTIONS} ${DFLAGS} -I${CPPFORMAT_INC}
${BALANCES} : $$@.o
	@echo "Compiling with `which ${CXX}`"
	${CXX} -O${OPTLEVEL} -g -std=c++14 \
	    ${TRACE_OPTIONS} ${DFLAGS} ${CPPFORMAT_LIB}/${LIBFORMAT} \
	    -o $@ $@.o \
	    -lm
.PHONY: balances listbalances listsizes
info ::
	@echo "make balances : make all balance codes"
balances : ${BALANCES}
info ::
	@echo "make listbalances : list all balance codes"
listbalances :
	@echo ${BALANCES}

.PHONY: dots
info ::
	@echo "make dots : convert dot files to pdf"
dots :
	dot -O -Tpdf nodegraph*dot

.PHONY: clean
info ::
	@echo "make clean"
clean :
	@/bin/rm -rf *~ a.out *.o *.dSYM *.stf.* tautrace* *.prot *test.e* laptime*.out \
	    balance*.e[0-9]*
	@/bin/rm -f ${PROGRAMS} balance*-*-*
	@for o in `make listbalances` ; do \
	  rm -f $$o-* ; \
	 done
