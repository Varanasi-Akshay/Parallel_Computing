info ::
	@echo "make mpfiles" ; echo
mpfiles :
	for f in *.mp ; do mpost $$f ; done

clean ::
	for file in *.fin ; do \
	  rm -f $${file/.fin/.log} $${file/.fin/.ps} $${file/.fin/.pdf} \
	    $${file/.fin/.1} ; \
	done

include $(MAKEFILES_DIR)/Makefile.clean
