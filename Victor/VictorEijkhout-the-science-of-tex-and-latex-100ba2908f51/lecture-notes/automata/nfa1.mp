% Creator: Finomaton 0.8
% Creation-Date: Mon Sep 06 19:53:36 EDT 2004

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

circleit.s3("1");
s3.c = (61.5, -32.5);
drawboxed(s3);
draw fullcircle scaled 0.85(ypart s3.n - ypart s3.s) shifted s3.c;

circleit.s5("2");
s5.c = (61.5, -103.0);
drawboxed(s5);
draw fullcircle scaled 0.85(ypart s5.n - ypart s5.s) shifted s5.c;


% Next, draw the lines
drawarrow (s1.c)..controls (31.5, -42.5) and (s3.c)..(s3.c) cutbefore bpath s1 cutafter bpath s3;

drawarrow (s1.c)..controls (34.0, -91.0) and (s5.c)..(s5.c) cutbefore bpath s1 cutafter bpath s5;

drawarrow directionpoint dir 231 of bpath s3..controls (59.5, -6.5) and (59.5, -6.5)..directionpoint dir 135 of bpath s3;

drawarrow directionpoint dir 209 of bpath s5..controls (60.5, -66.5) and (60.5, -66.5)..directionpoint dir 139 of bpath s5;


% Finally, create labels
label(btex $\epsilon$ etex, (19.0, -41.5));
label(btex $\epsilon$ etex, (27.0, -93.5));
label("b", (68.0, -76.0));
label("a", (70.5, -14.5));

endfig;
end
