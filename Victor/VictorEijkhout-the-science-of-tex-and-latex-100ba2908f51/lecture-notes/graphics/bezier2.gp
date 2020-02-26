set terminal pdf
set xrange [0:6]
set yrange [0:3]
set parametric
B1(x) = x**3
B2(x) = 3*x**2*(1-x)
B3(x) = 3*x*(1-x)**2
B4(x) = (1-x)**3
P1x = .5  ; P1y = .2
P2x = 1.2 ; P2y = .4
P3x = 2.2 ; P3y = 1.3
P4x =   3 ; P4y = 1.2
P5x = 2*P4x-P3x
P5y = 2*P4y-P3y
P6x = 4.5 ; P6y = .2
P7x = 5   ; P7y = 2.5
set multiplot
plot [t=0:1] \
  P1x*B1(t)+P2x*B2(t)+P3x*B3(t)+P4x*B4(t), \
  P1y*B1(t)+P2y*B2(t)+P3y*B3(t)+P4y*B4(t) \
  title ""
plot [t=0:1] \
  P4x*B1(t)+P5x*B2(t)+P6x*B3(t)+P7x*B4(t), \
  P4y*B1(t)+P5y*B2(t)+P6y*B3(t)+P7y*B4(t) \
  title ""
plot [t=-1:1] \
  P4x+t*(P5x-P4x),P4y+t*(P5y-P4y) \
  title "" with lines 2
plot [t=0:1] \
  P1x+t*(P2x-P1x),P1y+t*(P2y-P1y) \
  title "" with lines 3
plot [t=0:1] \
  P7x+t*(P6x-P7x),P7y+t*(P6y-P7y) \
  title "" with lines 3
