################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Mode-independent unit tests
####
################################################################

info ::
	@echo "Base code of an IMP backend"
	@echo "Available commands:"

include Make.inc
include Make.common
CXXFLAGS = ${CXX_OPTFLAGS}

##
## Library stuff
##
%.o : %.cxx
	${CXX} ${CXXFLAGS} ${MPIFLAGS} ${CXXUNITFLAGS} \
	  -I${CATCHDIR} -I${CPPFORMAT_INC} \
	  -c $<

imp_base.o utils.o : utils.h
imp_base.o : imp_base.h imp_base.cxx indexstruct.hpp

##
## Unittests, independent of any mode
##

UNITTESTS = unittest_indexstruct
.SECONDEXPANSION:
unittest_indexstruct : unittest_indexstruct.o unittest_main.o indexstruct.o
	${CXX} ${CXXFLAGS} ${CXXUNITFLAGS} -I${CATCHDIR} \
	    -o $@ $^ \
	    ${CPPFORMAT_LIB}/${LIBFORMAT} ${LIBCXX}
${patsubst %,%.o,${UNITTESTS}} : imp_base.h imp_base.cxx
unittest_indexstruct.o indexstruct.o : indexstruct.hpp

include ../imp/Make.regress
REGRESSIONTESTS = unittest_indexstruct
unittest :
	@export u=${UNITTEST} ; \
	  echo "\n================\nTesting" $$u ; \
	  rm -f $$u ; make $$u >/dev/null 2>&1 ; \
	  if [ ! -f $$u ] ; then echo "Could not make $$u"; exit 1; fi ; \
	  ./$$u 2>&1 | grep "\(passed\|failed\)" > unittest.out 2>&1 ; \
	  cat unittest.out
