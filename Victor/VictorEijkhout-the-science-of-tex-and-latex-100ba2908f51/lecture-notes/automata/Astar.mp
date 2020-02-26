% Creator: Finomaton 0.8
% Creation-Date: Wed Sep 08 17:24:27 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s194("0");
s194.c = (32.0, -65.0);
pickup pencircle scaled 1.2bp;
drawboxed(s194);
pickup defaultpen;

circleit.s196("s");
s196.c = (95.0, -65.0);
drawboxed(s196);

circleit.s198("s");
s198.c = (153.0, -65.0);
drawboxed(s198);

circleit.s200("1");
s200.c = (225.5, -65.0);
drawboxed(s200);
draw fullcircle scaled 0.85(ypart s200.n - ypart s200.s) shifted s200.c;


% Next, draw the lines
drawarrow (s194.c)..controls (64.5, -54.5) and (s196.c)..(s196.c) cutbefore bpath s194 cutafter bpath s196;

drawarrow (s196.c)..controls (130.5, -56.5) and (s198.c)..(s198.c) cutbefore bpath s196 cutafter bpath s198;

drawarrow (s198.c)..controls (189.5, -58.0) and (s200.c)..(s200.c) cutbefore bpath s198 cutafter bpath s200;

drawarrow (s198.c)..controls (133.5, -91.0) and (s196.c)..(s196.c) cutbefore bpath s198 cutafter bpath s196;


% Finally, create labels
label(btex $\epsilon$ etex, (56.5, -50.0));
label("A", (123.0, -52.0));
label(btex $\epsilon$ etex, (120.5, -92.5));
label(btex $\epsilon$ etex, (184.5, -51.0));

endfig;
end
