% Creator: Finomaton 0.8
% Creation-Date: Tue Sep 07 10:57:13 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s183("0");
s183.c = (27.0, -69.5);
pickup pencircle scaled 1.2bp;
drawboxed(s183);
pickup defaultpen;

circleit.s185("1");
s185.c = (102.5, -68.0);
drawboxed(s185);
draw fullcircle scaled 0.85(ypart s185.n - ypart s185.s) shifted s185.c;


% Next, draw the lines
drawarrow (s183.c)..controls (64.5, -58.5) and (s185.c)..(s185.c) cutbefore bpath s183 cutafter bpath s185;


% Finally, create labels
label(btex $\epsilon$ etex, (62.5, -49.0));

endfig;
end
