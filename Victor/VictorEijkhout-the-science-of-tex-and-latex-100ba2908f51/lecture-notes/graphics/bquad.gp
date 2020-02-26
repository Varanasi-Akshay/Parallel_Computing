set terminal pdf
set xrange [0:1]
set yrange [0:2]
set parametric
set multiplot
B1(x) = 1-2*x+x**2
B2(x) = (x-x**2)*2
B3(x) = x**2
p1x = .2; p1y = .2
p2x = .4; p2y = 1
p3x = .9; p3y = .1
plot [t=0:1] p1x*B1(t)+p2x*B2(t)+p3x*B3(t), \
             p1y*B1(t)+p2y*B2(t)+p3y*B3(t) \
     title "Quadratic Bezier curve"
set style function lines
plot [t=0:1] t*p2x+(1-t)*p1x,t*p2y+(1-t)*p1y title "" with lines 2
plot [t=0:1] t*p3x+(1-t)*p2x,t*p3y+(1-t)*p2y title "" with lines 2
