info :
	@echo "you can do : make primeprogram"

.SECONDEXPANSION:
primeprogram : $$@.o
	g++ -o $@ $^
primeprogram.o : primeprogram.cxx
	g++ -c $^

clean :
	/bin/rm -f primeprogram *~ *.o
