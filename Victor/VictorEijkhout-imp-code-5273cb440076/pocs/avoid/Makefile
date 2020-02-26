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
## set compiler and options
##
ifdef TACXX_TAU_DIR
  CXX = tau_cxx.sh
else
  CXX = g++
endif
ifdef TACXX_ITAC_DIR
  TRACE_OPTIONS = -trace
endif

##
## set include packages
##
CPPFORMAT_INC = ${HOME}/Installation/cppformat/cppformat
CPPFORMAT_LIB = ${HOME}/Installation/cppformat/installation/lib
LIBFORMAT = libfmt.a

AVOIDS = \
    avoid1 avoid2 avoid_hpl
OTHER_AVOIDS =

PROGRAMS = ${AVOIDS} ${OTHER_AVOIDS}
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
	    -O${OPTLEVEL} -g -std=c++17 \
	    ${TRACE_OPTIONS} ${DFLAGS} -I${CPPFORMAT_INC}
${AVOIDS} : $$@.o
	@echo "Compiling with `which ${CXX}`"
	${CXX} -O${OPTLEVEL} -g -std=c++17 \
	    ${TRACE_OPTIONS} ${DFLAGS} ${CPPFORMAT_LIB}/${LIBFORMAT} \
	    -o $@ $@.o \
	    -lm
#avoid1.o avoid2.o : tasklib.hpp
$(patsubst %,%.o,${AVOIDS}) : tasklib.hpp

.PHONY: avoids listavoids listsizes
info ::
	@echo "make avoids : make all avoid codes"
avoids : ${AVOIDS}
${AVOID} : tasklib.hpp
info ::
	@echo "make listavoids : list all avoid codes"
listavoids :
	@echo ${AVOIDS}

.PHONY: dots dotview
info ::
	@echo "make dots : convert dot files to pdf"
dots :
	dot -O -Tpdf nodegraph*dot
dotview :
	for n in nodegraph*dot.pdf ; do \
	  xpdf $$n ; done

.PHONY: clean veryclean
info ::
	@echo "make clean, veryclean"
clean :
	@/bin/rm -rf *~ a.out *.o *.dSYM *.stf.* tautrace* *.prot *test.e* laptime*.out \
	    avoid*.e[0-9]*
	@/bin/rm -f ${PROGRAMS} avoid*-*-*
	@for o in `make listavoids` ; do \
	  rm -f $$o-* ; \
	 done
veryclean : clean 
	@/bin/rm -f *.dot *.dot.pdf
