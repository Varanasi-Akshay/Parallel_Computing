#
# Hermite interpolation
#
set terminal pdf
set multiplot
set xrange [0:1]
set yrange [0:1.3]
P1(x) = 2*x**3-3*x**2+1
P2(x) = x**3-2*x**2+x
P3(x) = -2*x**3+3*x**2
P4(x) = x**3-x**2
p1y = .3
p1slope = -2
p2y = 1
p2slope = -2
plot p1y*P1(x) + p1slope*P2(x) \
  + P3(x) + p2slope*P4(x) title ""
set parametric
set style function lines
plot [t=0:.1] t,  p1y+t*p1slope \
  title "" with lines 2
plot [t=0:.1] 1-t,p2y-t*p2slope \
  title "" with lines 2
