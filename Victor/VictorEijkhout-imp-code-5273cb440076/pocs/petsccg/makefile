
CFLAGS	         = -std=c99
FFLAGS	         =
CPPFLAGS         =
FPPFLAGS         =
LOCDIR           = src/ksp/ksp/examples/tutorials/
EXAMPLESC        = petsccg
MANSEC           = KSP
CLEANFILES       = petsccg
NP               = 1

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

petsccg: petsccg.o  chkopts
	-${CLINKER} -o petsccg petsccg.o  ${PETSC_KSP_LIB}
	${RM} petsccg.o

include ${PETSC_DIR}/lib/petsc/conf/test
