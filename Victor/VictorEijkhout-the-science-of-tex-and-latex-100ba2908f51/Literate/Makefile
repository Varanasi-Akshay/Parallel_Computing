info ::
	@echo "make hello"

.SUFFIXES: .nw
.nw.o :
	@echo "... tangle ..."
	@notangle -L $*.nw > $*.c ; \
	  if [ $$? -ne 0 ] ; then rm $*.c ; exit 1; fi
	@echo "... compile ..."
	@$(CC) -c -o $*.o $*.c
	@mv $*.c $*.c.bak

hello : hello.o
	@echo "... link ..."
	@$(CC) -o hello hello.o

clean ::
	for file in *.nw ; do \
	  /bin/rm -f $${file/.nw/.c} $${file/.nw/.c.bak} ; \
	done
	/bin/rm hello

include $(MAKEFILES_DIR)/Makefile.clean
