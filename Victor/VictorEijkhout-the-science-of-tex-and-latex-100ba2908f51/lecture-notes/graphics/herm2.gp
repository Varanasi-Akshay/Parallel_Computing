set terminal pdf
set parametric
set multiplot
set dummy t
set xrange [0:1]
set yrange [0:.8]
P1(t) = 2*t**3-3*t**2+1
P2(t) = t**3-2*t**2+t
P3(t) = -2*t**3+3*t**2
P4(t) = t**3-t**2
p1x = .1 ; p1y = .2
p2x =  1 ; p2y =  0
p3x = .5 ; p3y = .3
p4x =  0 ; p4y = -1
plot [t=0:1] \
  p1x*P1(t)+p2x*P2(t)+p3x*P3(t)+p4x*P4(t), \
  p1y*P1(t)+p2y*P2(t)+p3y*P3(t)+p4y*P4(t) \
  title ""
p5x = .9 ; p5y = .6
p6x =  0 ; p6y = -1
plot [t=0:1] \
  p3x*P1(t)+.5*p4x*P2(t)+p5x*P3(t)+p6x*P4(t), \
  p3y*P1(t)+.5*p4y*P2(t)+p5y*P3(t)+p6y*P4(t) \
  title "" with lines 2
plot [t=0:1] \
  p3x*P1(t)+2*p4x*P2(t)+p5x*P3(t)+p6x*P4(t), \
  p3y*P1(t)+2*p4y*P2(t)+p5y*P3(t)+p6y*P4(t) \
  title "" with lines 2

