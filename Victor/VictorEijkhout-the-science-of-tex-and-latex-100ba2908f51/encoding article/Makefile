
info ::
	@echo

LATEXTARGET = course
MORE_TEXINPUTS = ${HOME}/Software/TeX/macros:../tex-course/lecture-notes:./graphics

CHAPTER = encoding
info ::
	@echo "make article"
	@echo
article :
	export TEXINPUTS ; \
	  echo $(CHAPTER) \
	  | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex chapter
	bibtex chapter
	export TEXINPUTS ; \
	  echo $(CHAPTER) \
	  | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex chapter
	export TEXINPUTS ; \
	  echo $(CHAPTER) \
	  | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex chapter
	mv chapter.pdf $(CHAPTER)-article.pdf
pdfview :
	xpdf -g 750x800+415+40 $(CHAPTER)-article.pdf &

include $(MAKEFILES_DIR)/Makefile.tex
include $(MAKEFILES_DIR)/Makefile.sync
include $(MAKEFILES_DIR)/Makefile.clean

clean ::
	for c in $(CHAPTERS) ; do \
	  rm -f $${c}-an*.tex $${c}-ex*.tex $${c}-nx.tex ; \
	done
	rm -f Chapter-ex*.tex Chapter-an*.tex
