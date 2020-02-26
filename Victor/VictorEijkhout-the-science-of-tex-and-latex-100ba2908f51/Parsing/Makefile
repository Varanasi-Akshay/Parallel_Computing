info ::


.SUFFIXES:
.SUFFIXES: .l .y .o

info ::
	@echo "make lextest [LEXFILE=<name>] [LEXINPUT=<name>]"
LEXFILE = texlexer
LEXINPUT_texlexer = texlexerinput
LEXINPUT_numberword = numidwords
LEXINPUT_numberword2 = numidwords
LEXINPUT_postfix = postsums
LEXINPUT_pp = sloppyp
LEXINPUT_pp2 = sloppyp
LEXINPUT_pp3 = sloppyp2

.l.o :
	lex -t $*.l > $*.c
	cc -c $*.c -o $*.o
lextest : $(LEXFILE).o
	cc $(LEXFILE).o -o $(LEXFILE) -ll
	@make lexrun
lexrun  :
	@if [ ! -z "$(LEXINPUT)" ] ; then \
	    (cat $(LEXINPUT) | ./$(LEXFILE)) ; \
	elif [ -f "$(LEXINPUT_$(LEXFILE))" ] ; then \
	    (cat $(LEXINPUT_$(LEXFILE)) | ./$(LEXFILE)) ; \
	elif [ -f "$(LEXFILE)-input" ] ; then \
	    (cat $(LEXFILE)-input | ./$(LEXFILE)) ; \
	fi

info ::
	@echo "make yacctest [YACCFILE=<name>] [YACCINPUT=<name>]"
YACCFILE = tex
YACCINPUT_calc1 = sums1
YACCINPUT_calc2 = sums2
YACCINPUT_calc3 = sums3
YACCINPUT_tex = latexsample.tex
YACCINPUT_words = somewords
.y.o :
	if [ ! -f $*.h ] ; then touch $*.h ; fi
	yacc -d -t -o $*.c $*.y 
	cc -c -o $*.o $*.c ;
	rm $*.c
yacctest : $(YACCFILE).o $(YACCFILE)-lex.o
	cc $(YACCFILE).o $(YACCFILE)-lex.o -o $(YACCFILE) -ly -ll
	@make yaccrun
yaccrun  :
	@if [ ! -z "$(YACCINPUT)" ] ; then \
	   ( cat ${YACCINPUT} | ./$(YACCFILE) ) ; \
	 elif [ -f "$(YACCINPUT_$(YACCFILE))" ] ; then \
	   ( cat ${YACCINPUT_$(YACCFILE)} | ./$(YACCFILE) ) ; \
	 elif [ -f "$(YACCFILE)-input" ] ; then \
	   ( cat ${YACCFILE}-input | ./$(YACCFILE) ) ; \
	 fi

yyaccrun  :
	@if [ -f "$(YACCINPUT_$(YACCFILE))" ] ; then \
	   echo "testing on $(YACCINPUT_$(YACCFILE))...." ; \
	   (cat $(YACCINPUT_$(YACCFILE)) | ./$(YACCFILE)) ; \
	   echo "done." ; \
	fi

info ::
	@echo "make clean"
clean :
	@rm -f *.o *~ $(LEXFILE) y.tab.h y.tab.c
	@for file in *.l ; do \
	    rm -f $${file/.l/.c} $${file/.l/} ; \
	done
	@for file in *.y ; do \
	    rm -f $${file/.y/.c} $${file/.y/.h} $${file/.y/} ; \
	done
