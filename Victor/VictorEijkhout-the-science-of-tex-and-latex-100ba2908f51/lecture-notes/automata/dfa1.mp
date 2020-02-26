% Creator: Finomaton 0.8
% Creation-Date: Mon Sep 06 20:01:38 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s1("0");
s1.c = (15.5, -69.0);
pickup pencircle scaled 1.2bp;
drawboxed(s1);
pickup defaultpen;
draw fullcircle scaled 0.85(ypart s1.n - ypart s1.s) shifted s1.c;

circleit.s30("1");
s30.c = (58.0, -31.0);
drawboxed(s30);
draw fullcircle scaled 0.85(ypart s30.n - ypart s30.s) shifted s30.c;

circleit.s32("2");
s32.c = (58.0, -105.5);
drawboxed(s32);
draw fullcircle scaled 0.85(ypart s32.n - ypart s32.s) shifted s32.c;


% Next, draw the lines
drawarrow directionpoint dir 216 of bpath s30..controls (57.0, -0.0) and (57.0, -0.0)..directionpoint dir 135 of bpath s30;

drawarrow directionpoint dir 208 of bpath s32..controls (56.0, -63.0) and (56.0, -63.0)..directionpoint dir 156 of bpath s32;

drawarrow (s1.c)..controls (28.5, -44.0) and (s30.c)..(s30.c) cutbefore bpath s1 cutafter bpath s30;

drawarrow (s1.c)..controls (31.0, -91.5) and (s32.c)..(s32.c) cutbefore bpath s1 cutafter bpath s32;


% Finally, create labels
label("a", (19.0, -41.5));
label("b", (27.0, -93.5));
label("b", (70.5, -86.5));
label("a", (73.5, -15.0));

endfig;
end
