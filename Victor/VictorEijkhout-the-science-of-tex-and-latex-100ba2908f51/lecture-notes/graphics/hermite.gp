#
# 4 cubic Hermite polynomials
#
set terminal pdf
set xrange [0:1]
set yrange [-.2:1.2]
P1(x) = 2*x**3-3*x**2+1
P2(x) = x**3-2*x**2+x
P3(x) = -2*x**3+3*x**2
P4(x) = x**3-x**2
plot P1(x), P2(x), P3(x), P4(x) title ""
