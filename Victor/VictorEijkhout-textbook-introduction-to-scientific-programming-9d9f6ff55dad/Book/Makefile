# -*- makefile -*-
################################################################
####
#### This makefile is part of the course
#### Introduction to Scientific Programming in C++ and Fortran
#### by Victor Eijkhout (eijkhout@tacc.utexas.edu)
#### copyright 2017/8 Victor Eijkhout
####
#### guide/Makefile : makefile for public version of the textbook
####
################################################################

.PHONY: info
info ::

info ::
	@echo "make all : code snippets, book"
.PHONY: all
all : snippets
	make manual book FILE=${FILE}

FILE = ispbook
.PHONY: onepass format book manual pdf
info ::
	@echo "make onepass/format book [FILE=(default: ${FILE})]"
format onepass :
	@mkdir -p frames
	TEXINPUTS=${TEXINPUTS}:chapters:snippets.code:graphics pdflatex --shell-escape ${FILE}
BOOKNAME = EijkhoutIntroSciProgramming
book :
	@echo "Formatting as student book"
	@cp q.tex qa.tex
	make pdf FILE=${FILE}
	cp ${FILE}.pdf ${BOOKNAME}.pdf
	cp ${BOOKNAME}.pdf ..
pdf :
	make snippets
	make onepass FILE=${FILE}
	make extract_exercises extract_slides
	makeindex ${FILE}
	make onepass FILE=${FILE}


####
#### Extract code to snippet directory
####
info ::
	@echo "make frames"
	@echo "  make snippets, extract_slides"
	@echo "  make extract_exercises, exercises"
.PHONY: snippets extract_exercises extract_slides exercises frames
frames :
	make onepass snippets extract_exercises extract_slides
snippets :
	@echo "extracting code snippets" && \
	export root=`pwd`/.. && \
	mkdir -p snippets.code && \
	cd snippets.code && \
	  for d in \
	        $$root/code/array \
	        $$root/code/basic \
	        $$root/code/cast \
	        $$root/code/const \
	        $$root/code/func \
	        $$root/code/geom \
	        $$root/code/hello \
	        $$root/code/infect \
	        $$root/code/io \
	        $$root/code/loop \
	        $$root/code/namespace \
	        $$root/code/object \
	        $$root/code/pointer \
	        $$root/code/primes \
	        $$root/code/proto \
	        $$root/code/recur \
	        $$root/code/sort \
	        $$root/code/stl \
	        $$root/code/string \
	        $$root/code/struct \
	        $$root/code/template \
	        $$root/code/tree \
	        \
	        $$root/code/f \
	        $$root/code/basicf \
	        $$root/code/arrayf \
	        $$root/code/ftype \
	        $$root/code/funcf \
	        $$root/code/geomf \
	        $$root/code/iof \
	        $$root/code/loopf \
	        $$root/code/objectf \
	        $$root/code/pointerf \
	        $$root/code/slice \
	        $$root/code/stringf \
	        $$root/code/structf \
	        \
	        $$root/tester/ijloop \
	           ; do \
	    echo ".. in $$d" ; \
	    for c in $$d/*.cxx $$d/*.F90 $$d/*.h \
	        ; do \
	      cat $$c | awk -v fn=$${c##*/} -v fd=$$d \
	          'BEGIN {f=0; ff=0; file="foo"} \
	           /answersnippet end/  {f=0} \
	           /codesnippet end/    {f=0} \
	           /examplesnippet end/ {f=0} \
	           f==1 && ff==1 { print "//" FS fd "/" fn > file; ff=0 } \
	           f==1 {print $$0 > file} \
	           /answersnippet/  && $$2==file { print "  /* ... */" > file } \
	           /codesnippet/    && $$2==file { print "  /* ... */" > file } \
	           /examplesnippet/ && $$2==file { print "  /* ... */" > file } \
	           /answersnippet/  && !/end/ {f=1; ff=0; file=$$2 } \
	           /codesnippet/    && !/end/ {f=1; ff=0; file=$$2 } \
	           /examplesnippet/ && !/end/ {f=1; ff=1; file=$$2 } \
	          ' ; \
	    done >/dev/null 2>&1 ; \
	  done && \
	  for s in `ls` ; do \
	    python ../cutspace.py $$s ; \
	  done && \
	  echo ".. sanitizing white space for examples in `pwd`" ; \
	  echo ".. extracted"
nocut :
	continue
# extra awk clause for listing file name: f==1 && ff==0 {print "// " fn > file; ff=1}
VERBOSE = 0
extract_exercises :
	@echo "labeling exercises" && \
	 cd exercises && \
	  for e in *-ex[0-9]*.tex ; do \
	    if [ `cat $$e | grep label | wc -l` -eq 1 ] ; then \
	      export labelname=`cat $$e | grep label | sed -e s/^.*label.// -e s/\}//` ; \
	      if [ ${VERBOSE} -eq 1 ] ; then \
	        echo "Copy $$e to $$labelname.tex" ; \
	      fi ; \
	      cp $$e $$labelname.tex ; \
	      export answername=`echo $$e | sed s/-ex/-an/` ; \
	      export labelname=`echo $$labelname | sed s/ex:/an:/` ; \
	      if [ -e ../answers/$$answername ] ; then \
	        if [ ${VERBOSE} -eq 1 ] ; then \
	          echo "Copy $$answername to $$labelname.tex" ; \
	        fi ; \
	        ( cd ../answers ; cp $$answername $$labelname.tex ) ; \
	      fi ; \
	    fi ; \
	  done
exercises : onepass extract_exercises
extract_slides :
	@echo "labeling slides" && \
	 cd frames && \
	  for e in *.tex ; do \
	    if [ `cat $$e | grep label | wc -l` -eq 1 ] ; then \
	      export nam=`cat $$e | grep label | sed -e s/^.*label.// -e s/\}//` ; \
	      if [ $$e != $$nam.tex ] ; then \
	        cp $$e $$nam.tex ; \
	      fi ; \
	    fi ; \
	  done

clean ::
	@/bin/rm -f exercises/*
	@/bin/rm -f chapters/*~

info ::
	@echo "make clean : modest cleanup"
	@echo "make total_clean : also remove snippets and exercises"
.PHONY: clean total_clean
clean ::
	@/bin/rm -rf *~ *.o *.{aux,ilg,ind,idx,log,out,nav,toc} ${PROGRAMS} *.dSYM
total_clean : clean
	@/bin/rm -rf frames/* answers/*
