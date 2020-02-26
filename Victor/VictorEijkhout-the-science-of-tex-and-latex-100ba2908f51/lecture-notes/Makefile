.PHONY: info
info ::
	@echo

LATEXTARGET = course
MORE_TEXINPUTS = ../Parsing:../Breaking:./trees:./automata:./graphics:${HOME}/Software/TeX/LaTeX
FULLNAME = TeX\ LaTeX\ course.pdf
info ::
	@echo "make book"
book :
	make clean ; make epsfiles mpfiles
	for c in ${CHAPTERS} ; do make chapter exercises CHAPTER=$$c ; done
	make format
	makeindex course ; bibtex course
	make format
	make format
	make copy
copy :
	cp course.pdf ../handouts/${FULLNAME}
	cp course.pdf ${FULLNAME}
format :
	export TEXINPUTS=../Parsing:../Breaking:./trees:./automata:./graphics:${HOME}/Software/TeX/LaTeX:${TEXINPUTS} ; pdflatex course

info ::
	@echo "make upload: copy to dropbox"
upload :
	cp course.pdf ${HOME}/DropBox/TACC/Projects/${FULLNAME}


CHAPTERS = latex tex tex1 tex2 parsing lex yacc hashing \
    dynamic python completeness paragraph page \
    bezier raster lambda encoding
CHAPTER = parsing
info ::
	@echo "make tutorial/chapter, pdfview CHAPTER=<name>"
	@echo "  current default is $(CHAPTER)"
	@echo
tutorial chapter :
	export TEXINPUTS ; \
	  echo $(CHAPTER) \
	  | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex chapter
	mv chapter.pdf $(CHAPTER)-tutorial.pdf
	cp $(CHAPTER)-tutorial.pdf ../handouts
pdfview :
	xpdf -g 750x800+415+40 $(CHAPTER)-tutorial.pdf &

info ::
	@echo "make exercises/answers CHAPTER=<name>"
	@echo "  current default is $(CHAPTER)"
	@echo
exercises :
	export TEXINPUTS ; \
	  echo $(CHAPTER) \
	  | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex exercises
	mv exercises.pdf $(CHAPTER)-exercises.pdf
	cp $(CHAPTER)-exercises.pdf ../handouts
answers :
	export TEXINPUTS ; \
	   echo $(CHAPTER) \
	   | TEXINPUTS="$(TEXINPUTS):$(MORE_TEXINPUTS)" pdflatex answers
	mv answers.pdf $(CHAPTER)-answers.pdf
	cp $(CHAPTER)-answers.pdf ../answers

mpfiles :
	cd automata ; make mpfiles
epsfiles :
	cd graphics ; make epsfiles

include $(MAKEFILES_DIR)/Makefile.tex
include $(MAKEFILES_DIR)/Makefile.sync
include $(MAKEFILES_DIR)/Makefile.clean

clean ::
	for c in $(CHAPTERS) ; do \
	  rm -f $${c}-an*.tex $${c}-ex*.tex $${c}-nx.tex ; \
	done
	rm -f Chapter-ex*.tex Chapter-an*.tex
