info ::
	@echo "make run [PYFILE=<default: para>]"

PYFILE = para
run :
	@( (cat para.in | para ) ; echo EOF) | python $(PYFILE).py
info ::
	@echo "make filter"
filter : para.o
	cc -o para para.o -ll

include $(MAKEFILES_DIR)/Makefile.clean
include $(MAKEFILES_DIR)/Makefile.sync
