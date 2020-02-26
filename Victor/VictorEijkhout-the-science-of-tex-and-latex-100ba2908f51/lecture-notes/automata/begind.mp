% Creator: Finomaton 0.8
% Creation-Date: Wed Sep 08 18:33:32 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s224("0");
s224.c = (23.5, -67.5);
pickup pencircle scaled 1.2bp;
drawboxed(s224);
pickup defaultpen;

circleit.s226("1");
s226.c = (79.5, -40.5);
drawboxed(s226);

circleit.s228("6");
s228.c = (77.0, -97.0);
drawboxed(s228);
draw fullcircle scaled 0.85(ypart s228.n - ypart s228.s) shifted s228.c;

circleit.s231("2");
s231.c = (133.0, -40.5);
drawboxed(s231);

circleit.s233("7");
s233.c = (132.5, -97.5);
drawboxed(s233);
draw fullcircle scaled 0.85(ypart s233.n - ypart s233.s) shifted s233.c;

circleit.s236("3");
s236.c = (183.5, -40.5);
drawboxed(s236);

circleit.s238("8");
s238.c = (181.5, -97.0);
drawboxed(s238);
draw fullcircle scaled 0.85(ypart s238.n - ypart s238.s) shifted s238.c;


% Next, draw the lines
drawarrow (s224.c)..controls (41.5, -45.5) and (s226.c)..(s226.c) cutbefore bpath s224 cutafter bpath s226;

drawarrow (s224.c)..controls (39.5, -87.5) and (s228.c)..(s228.c) cutbefore bpath s224 cutafter bpath s228;

drawarrow (s226.c)..controls (105.0, -23.5) and (s231.c)..(s231.c) cutbefore bpath s226 cutafter bpath s231;

drawarrow (s231.c)..controls (156.5, -24.5) and (s236.c)..(s236.c) cutbefore bpath s231 cutafter bpath s236;

drawarrow (s226.c)..controls (106.0, -69.0) and (s233.c)..(s233.c) cutbefore bpath s226 cutafter bpath s233;

drawarrow (s231.c)..controls (162.5, -69.5) and (s238.c)..(s238.c) cutbefore bpath s231 cutafter bpath s238;

drawarrow directionpoint dir 310 of bpath s228..controls (80.0, -126.5) and (80.0, -126.5)..directionpoint dir 414 of bpath s228;

drawarrow directionpoint dir 317 of bpath s233..controls (135.0, -129.0) and (135.0, -129.0)..directionpoint dir 414 of bpath s233;

drawarrow directionpoint dir 330 of bpath s238..controls (181.5, -131.0) and (181.5, -131.0)..directionpoint dir 410 of bpath s238;


% Finally, create labels
label("B", (29.5, -39.0));
label(btex $\neg{}$B etex, (44.5, -77.0));
label("E", (96.0, -14.0));
label(btex $\neg{}$E etex, (110.5, -59.5));
label("G", (150.0, -18.5));
label(btex $\neg{}$G etex, (168.0, -59.5));
label(btex A--Z etex, (62.0, -119.5));
label(btex A--Z etex, (118.5, -118.5));
label(btex A--Z etex, (170.0, -117.0));
label("text", (214.0, -37.5));
label(btex \dots etex, (214.0, -37.5));
label(btex \dots etex, (212.5, -94.5));

endfig;
end
