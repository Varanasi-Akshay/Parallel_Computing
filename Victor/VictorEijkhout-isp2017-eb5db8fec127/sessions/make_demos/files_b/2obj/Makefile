info :
	@echo "you can do: make primeprogram"

primeprogram : primeprogram.o
	g++ -o  primeprogram  primeprogram.o
primeprogram.o : primeprogram.cxx
	g++ -c primeprogram.cxx

clean :
	rm -f primeprogram *~ *.o
