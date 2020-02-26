% Creator: Finomaton 0.8
% Creation-Date: Tue Sep 14 16:07:58 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s85("0");
s85.c = (26.0, -71.0);
pickup pencircle scaled 1.2bp;
drawboxed(s85);
pickup defaultpen;

circleit.s87("1");
s87.c = (76.5, -29.5);
drawboxed(s87);

circleit.s89("6");
s89.c = (76.5, -105.0);
drawboxed(s89);

circleit.s92("2");
s92.c = (123.5, -29.5);
drawboxed(s92);

circleit.s94("7");
s94.c = (223.5, -71.5);
drawboxed(s94);
draw fullcircle scaled 0.85(ypart s94.n - ypart s94.s) shifted s94.c;

circleit.s97("5");
s97.c = (190.0, -27.5);
drawboxed(s97);


% Next, draw the lines
drawarrow (s85.c)..controls (45.0, -41.0) and (s87.c)..(s87.c) cutbefore bpath s85 cutafter bpath s87;

drawarrow (s87.c)..controls (97.5, -13.5) and (s92.c)..(s92.c) cutbefore bpath s87 cutafter bpath s92;

drawarrow (s85.c)..controls (41.5, -94.0) and (s89.c)..(s89.c) cutbefore bpath s85 cutafter bpath s89;

drawarrow directionpoint dir 204 of bpath s89..controls (76.5, -63.0) and (76.5, -63.0)..directionpoint dir 133 of bpath s89;

drawarrow (s97.c)..controls (217.5, -48.0) and (s94.c)..(s94.c) cutbefore bpath s97 cutafter bpath s94;

drawarrow (s89.c)..controls (211.0, -109.5) and (s94.c)..(s94.c) cutbefore bpath s89 cutafter bpath s94;


% Finally, create labels
label(btex $\epsilon$ etex, (32.0, -37.0));
label(btex $\epsilon$ etex, (45.0, -80.5));
label(btex A--Z etex, (90.0, -80.5));
label("B", (89.5, -8.0));
label(btex E--G--I etex, (164.5, -14.0));
label(btex \dots etex, (164.5, -24.0));
label("N", (213.5, -37.5));
label(btex a--z etex, (169.0, -94.5));

endfig;
end
