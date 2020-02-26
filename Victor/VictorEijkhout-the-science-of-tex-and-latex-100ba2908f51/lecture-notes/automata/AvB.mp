% Creator: Finomaton 0.8
% Creation-Date: Tue Aug 24 20:49:00 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s42("0");
s42.c = (70.5, -36.0);
pickup pencircle scaled 1.2bp;
drawboxed(s42);
pickup defaultpen;

circleit.s44("1");
s44.c = (150.5, -34.0);
drawboxed(s44);
draw fullcircle scaled 0.85(ypart s44.n - ypart s44.s) shifted s44.c;

circleit.s46("0");
s46.c = (71.5, -78.0);
pickup pencircle scaled 1.2bp;
drawboxed(s46);
pickup defaultpen;

circleit.s48("1");
s48.c = (150.5, -76.5);
drawboxed(s48);
draw fullcircle scaled 0.85(ypart s48.n - ypart s48.s) shifted s48.c;

circleit.s69("s");
s69.c = (10.0, -54.5);
drawboxed(s69);

circleit.s71("s");
s71.c = (212.0, -53.5);
drawboxed(s71);


% Next, draw the lines
drawarrow (s42.c)..controls (111.5, -24.5) and (s44.c)..(s44.c) cutbefore bpath s42 cutafter bpath s44;

drawarrow (s46.c)..controls (113.0, -66.0) and (s48.c)..(s48.c) cutbefore bpath s46 cutafter bpath s48;

drawarrow (s69.c)..controls (41.0, -38.0) and (s42.c)..(s42.c) cutbefore bpath s69 cutafter bpath s42;

drawarrow (s69.c)..controls (38.5, -74.0) and (s46.c)..(s46.c) cutbefore bpath s69 cutafter bpath s46;

drawarrow (s44.c)..controls (181.5, -36.0) and (s71.c)..(s71.c) cutbefore bpath s44 cutafter bpath s71;

drawarrow (s48.c)..controls (179.0, -68.5) and (s71.c)..(s71.c) cutbefore bpath s48 cutafter bpath s71;


% Finally, create labels
label("A", (111.0, -19.5));
label("B", (112.0, -79.5));
label(btex $\epsilon$ etex, (32.5, -30.5));
label(btex $\epsilon$ etex, (38.0, -63.5));
label(btex $\epsilon$ etex, (187.5, -32.5));
label(btex $\epsilon$ etex, (179.5, -58.5));

endfig;
end
