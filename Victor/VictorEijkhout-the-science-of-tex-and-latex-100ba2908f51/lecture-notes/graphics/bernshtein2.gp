set terminal pdf
set xrange [0:1]
set yrange [0:2]
B1(x) = x**3
B2(x) = 3*x**2*(1-x)
B3(x) = 3*x*(1-x)**2
B4(x) = (1-x)**3
plot B1(x)+2*B2(x)+B3(x)+B4(x)

