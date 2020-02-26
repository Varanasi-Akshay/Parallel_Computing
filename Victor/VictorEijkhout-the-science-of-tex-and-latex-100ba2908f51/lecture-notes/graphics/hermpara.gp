#
# Parametric Hermite curve
#
set terminal pdf
set parametric
set multiplot
set xrange [0:1]
set yrange [0:.7]
P1(t) = 2*t**3-3*t**2+1
P2(t) = t**3-2*t**2+t
P3(t) = -2*t**3+3*t**2
P4(t) = t**3-t**2
p1x  = .1 ; p1y  = .2
p1dx =  1 ; p1dy =  0
p2x  = .9 ; p2y  = .3
p2dx =  0 ; p2dy = -1
plot [t=0:1] \
  p1x*P1(t)+p1dx*P2(t)+p2x*P3(t)+p2dx*P4(t), \
  p1y*P1(t)+p1dy*P2(t)+p2y*P3(t)+p2dy*P4(t) \
  title ""
plot [t=0:.2] p1x+t*p1dx,p1y+t*p1dy \
  title "" with lines 2
plot [t=0:.2] p2x-t*p2dx,p2y-t*p2dy \
  title "" with lines 2
