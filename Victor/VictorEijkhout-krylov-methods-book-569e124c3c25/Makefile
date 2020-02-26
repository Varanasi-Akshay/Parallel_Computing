info :: 

LATEXTARGET = truth
XPDF_GEOMETRY = screen

info ::
	@echo "make onepass, pdf"
onepass :
	pdflatex ${LATEXTARGET}
pdf :
	make onepass
	makeindex ${LATEXTARGET}
	make onepass
	make onepass

info ::
	@echo "make upload"
upload :
	cp truth.pdf ${HOME}/Dropbox/Scicomp/EijkhoutCG.pdf

clean :: 
	@/bin/rm -f *~ *.{aux,blg,ilg,log,ind,sta,toc}

