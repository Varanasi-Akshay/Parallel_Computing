% Creator: Finomaton 0.8
% Creation-Date: Sun Sep 12 10:34:43 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s1("s");
s1.c = (23.5, -84.5);
drawboxed(s1);

circleit.s3("s");
s3.c = (80.0, -20.0);
drawboxed(s3);

circleit.s5("s");
s5.c = (79.0, -51.0);
drawboxed(s5);

circleit.s7("s");
s7.c = (79.0, -83.5);
drawboxed(s7);

circleit.s9("s");
s9.c = (116.5, -20.0);
drawboxed(s9);

circleit.s11("s");
s11.c = (117.5, -50.5);
drawboxed(s11);

circleit.s13("s");
s13.c = (155.5, -50.5);
drawboxed(s13);

circleit.s15("s");
s15.c = (186.5, -50.0);
drawboxed(s15);

circleit.s17("s");
s17.c = (255.0, -81.5);
drawboxed(s17);

circleit.s19("s");
s19.c = (78.0, -121.0);
drawboxed(s19);

circleit.s21("s");
s21.c = (150.5, -120.0);
drawboxed(s21);

circleit.s23("s");
s23.c = (186.5, -17.5);
drawboxed(s23);


% Next, draw the lines
drawarrow (s23.c)..controls (241.5, -31.0) and (s17.c)..(s17.c) cutbefore bpath s23 cutafter bpath s17;

drawarrow (s1.c)..controls (41.0, -34.0) and (s3.c)..(s3.c) cutbefore bpath s1 cutafter bpath s3;

drawarrow (s1.c)..controls (47.5, -58.5) and (s5.c)..(s5.c) cutbefore bpath s1 cutafter bpath s5;

drawarrow (s1.c)..controls (51.5, -83.5) and (s7.c)..(s7.c) cutbefore bpath s1 cutafter bpath s7;

drawarrow (s1.c)..controls (48.5, -106.0) and (s19.c)..(s19.c) cutbefore bpath s1 cutafter bpath s19;

drawarrow (s19.c)..controls (113.5, -122.0) and (s21.c)..(s21.c) cutbefore bpath s19 cutafter bpath s21;

drawarrow directionpoint dir 204 of bpath s7..controls (s5.c) and (s5.c)..directionpoint dir 128 of bpath s7;

drawarrow directionpoint dir 231 of bpath s21..controls (148.091900815, -93.81627004) and (148.091900815, -93.81627004)..directionpoint dir 135 of bpath s21;

drawarrow (s21.c)..controls (199.0, -92.0) and (s17.c)..(s17.c) cutbefore bpath s21 cutafter bpath s17;

drawarrow (s15.c)..controls (227.5, -57.0) and (s17.c)..(s17.c) cutbefore bpath s15 cutafter bpath s17;

drawarrow (s7.c)..controls (141.5, -76.0) and (s17.c)..(s17.c) cutbefore bpath s7 cutafter bpath s17;

drawarrow (s17.c)..controls (77.5, -213.5) and (s1.c)..(s1.c) cutbefore bpath s17 cutafter bpath s1;


% Finally, create labels
label("E", (97.5, -46.0));
label("N", (135.5, -45.0));
label("D", (170.5, -45.0));
label("B", (97.5, -11.0));
label("E G I N", (148.5, -13.0));
label(btex 0--9 etex, (88.0, -67.0));
label("a-z", (110.0, -115.0));
label("a-z, 0-9", (142.5, -96.0));
label(btex $\epsilon$ etex, (229.5, -24.5));
label(btex $\epsilon$ etex, (33.5, -34.0));
label(". . . . . . . . . . . . ", (152.5, -19.5));
label(btex $\epsilon$ etex, (215.5, -121.0));

endfig;
end
