#
# makefile for slides
#
info ::

LATEXTARGET = latex
MORE_TEXINPUTS = ../lecture-notes:../lecture-notes/automata:../lecture-notes/trees:../lecture-notes/graphics

CHAPTER = tex
info ::
	@echo "make slides/pdfview CHAPTER=<name>"
	@echo "  current default is $(CHAPTER)"
	@echo
slides :
	export MORE_TEXINPUTS ; \
	  MORE_TEXINPUTS="$(MORE_TEXINPUTS)" \
	  make pdf LATEXTARGET=$(CHAPTER) 
	mv $(CHAPTER).pdf $(CHAPTER)-slides.pdf
	cp $(CHAPTER)-slides.pdf ../handouts
pdfview :
	xpdf -g 750x630+520+140 $(CHAPTER)-slides.pdf &

clean ::
	@rm -f *.out *.nav *.snm

include $(MAKEFILES_DIR)/Makefile.tex
include $(MAKEFILES_DIR)/Makefile.clean
include $(MAKEFILES_DIR)/Makefile.sync
