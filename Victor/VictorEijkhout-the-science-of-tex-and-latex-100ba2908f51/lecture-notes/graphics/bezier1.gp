set terminal pdf
set xrange [0:6]
set yrange [0:2]
set parametric
B1(x) = x**3
B2(x) = 3*x**2*(1-x)
B3(x) = 3*x*(1-x)**2
B4(x) = (1-x)**3
P1x = 1 ; P1y = 1
P2x = 2 ; P2y = 2
P3x = 4 ; P3y = 0
P4x = 5 ; P4y = 1
set multiplot
plot [t=0:1] P1x*B1(t)+P2x*B2(t)+P3x*B3(t)+P4x*B4(t), \
             P1y*B1(t)+P2y*B2(t)+P3y*B3(t)+P4y*B4(t) \
     title "Bezier curve"
set style function lines
plot [t=0:1] t*P2x+(1-t)*P1x,t*P2y+(1-t)*P1y title "" with lines 2
plot [t=0:1] t*P3x+(1-t)*P1x,t*P3y+(1-t)*P1y title "" with lines 2
plot [t=0:1] t*P2x+(1-t)*P4x,t*P2y+(1-t)*P4y title "" with lines 2
plot [t=0:1] t*P3x+(1-t)*P4x,t*P3y+(1-t)*P4y title "" with lines 2
