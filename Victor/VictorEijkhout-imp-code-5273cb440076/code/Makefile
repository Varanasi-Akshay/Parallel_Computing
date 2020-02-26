# -*- makefile -*-
################################################################
####
#### This makefile is part of the prototype implementation of
#### the Integrative Model for Parallelism
####
#### copyright Victor Eijkhout 2014-7
####
#### Root level makefile for IMP code
####
################################################################

info ::

include imp/Make.inc

####
#### upload : if you are victor, upload docs to dropbox
####
info ::
	@if [ `whoami` = "eijkhout" ] ; then \
	  echo "make upload" ; fi
.PHONY: upload
upload :
	cd ../docs ; \
	  ln -f latex/refman.pdf .. ; \
	  if [ `whoami` = "eijkhout" ] ; then \
	    cp latex/refman.pdf ${HOME}/DropBox/IMP/ ; \
	  fi
notarget :
	  for f in *.tex ; do \
	    sed -e 's/\[H\]/[htp]/' -ibak $$f ; done ; \
	  pdflatex refman ; pdflatex refman

# code summary
.PHONY: summary history
info ::
	@echo "make summary, history"
summary:
	@for d in imp mpi omp product ; do \
	  ( cd $$d ; \
	    echo "Code base for $$d:" ; \
	    cat *base.{h,cxx} *functions.{h,cxx} *_ops.h 2>/dev/null \
	    | awk 'BEGIN {skip=0 } \
	        /^\/\*/ { skip=1 } skip==0 {print} /\*\// { skip=0 } ' \
	    | grep -v "^[ \t]*//" \
	    | grep -v "^[ \t]*$$" \
	    | wc -l ; \
	  ) ; \
	done
	@echo "Unittests:" ; \
	for d in mpi omp product ; do \
	  ( cd $$d ; \
	    cat unittest_*.cxx \
	    | grep -v "^[ \t]*//" \
	    | grep -v "^[ \t]*$$" ; \
	  ) ; \
	done | wc -l 
history :
	hg history \
	| awk '/changeset:/ { c=$$2 } /date:/ {$$1=""; d = $$0} /summary:/ {$$1=""; print c $$0 d}' \
	| tr : ' '  \
	| awk '{$$2=""; print}' \
	| sort -n

info ::
	@echo "make clean"
# recursive cleanup of all the code directories
# not: benchmarking test
clean ::
	@/bin/rm -f *~
	@for d in imp mpi omp product ; do \
	  ( cd $$d ; pwd ; \
	    if [ -f Makefile ] ; then make clean ; fi ; \
	    if [ -d $${d}_apps ] ; then \
	      ( cd $${d}_apps ; pwd ; make clean ) ; \
	    fi \
	  ) ; \
	done
clean ::
	@/bin/rm -f imp/out

