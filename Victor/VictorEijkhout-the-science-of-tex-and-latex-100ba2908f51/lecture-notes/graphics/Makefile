info ::
	@echo "make all"

all : gpfiles epsfiles

.SUFFIXES: .pdf .gp .eps .epsf
.gp.pdf :
	gnuplot -xrm 'gnuplot*Line1color:blue' $*.gp > $*.pdf
gpfiles :
	for g in *.gp ; do make $${g/.gp/.pdf} ; done

.eps.pdf :
	ps2pdf $*.eps $*.pdf
.epsf.pdf :
	ps2pdf $*.epsf $*.pdf
epsfiles :
	for e in *.eps ; do make $${e/.eps/.pdf} ; done
	for e in *.epsf ; do make $${e/.epsf/.pdf} ; done

include $(MAKEFILES_DIR)/Makefile.clean
