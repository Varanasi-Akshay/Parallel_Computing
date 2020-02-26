set terminal pdf
set xrange [0:1]
plot x**3, 3*x**2*(1-x), 3*x*(1-x)**2, (1-x)**3
