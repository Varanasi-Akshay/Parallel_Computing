% Creator: Finomaton 0.8
% Creation-Date: Tue Aug 17 17:47:04 EDT 2004

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
s1.c = (21.5, -71.0);
drawboxed(s1);

circleit.s3("1");
s3.c = (60.0, -36.5);
drawboxed(s3);

circleit.s5("2");
s5.c = (60.5, -99.5);
drawboxed(s5);

circleit.s7("3");
s7.c = (112.0, -9.0);
drawboxed(s7);

circleit.s9("4");
s9.c = (113.5, -67.5);
drawboxed(s9);

circleit.s11("5");
s11.c = (114.5, -127.0);
drawboxed(s11);

circleit.s13("6");
s13.c = (168.0, -30.0);
drawboxed(s13);

circleit.s15("7");
s15.c = (168.5, -97.0);
drawboxed(s15);

circleit.s17("8");
s17.c = (221.0, -62.0);
drawboxed(s17);


% Next, draw the lines
drawarrow (s1.c)..controls (44.0, -52.0) and (s3.c)..(s3.c) cutbefore bpath s1 cutafter bpath s3;

drawarrow (s1.c)..controls (40.5, -85.5) and (s5.c)..(s5.c) cutbefore bpath s1 cutafter bpath s5;

p29 = (s3.c)..controls (86.5, -18.5) and (s7.c)..(s7.c) cutbefore bpath s3 cutafter bpath s7;
drawarrow p29;
label.top(btex 1 etex, point 0.5 of p29);

p33 = (s3.c)..controls (87.5, -47.0) and (s9.c)..(s9.c) cutbefore bpath s3 cutafter bpath s9;
drawarrow p33;
label.top(btex 3 etex, point 0.38 of p33);

p37 = (s3.c)..controls (81.0, -69.5) and (s11.c)..(s11.c) cutbefore bpath s3 cutafter bpath s11;
drawarrow p37;
label.top(btex 4 etex, point 0.21 of p37);

p41 = (s5.c)..controls (86.5, -59.5) and (s7.c)..(s7.c) cutbefore bpath s5 cutafter bpath s7;
drawarrow p41;
label.top(btex 4 etex, point 0.23 of p41);

p45 = (s5.c)..controls (97.0, -83.5) and (s9.c)..(s9.c) cutbefore bpath s5 cutafter bpath s9;
drawarrow p45;
label.top(btex 2 etex, point 0.7 of p45);

p49 = (s5.c)..controls (84.0, -113.5) and (s11.c)..(s11.c) cutbefore bpath s5 cutafter bpath s11;
drawarrow p49;
label.top(btex 2 etex, point 0.5 of p49);

p53 = (s7.c)..controls (144.5, -43.5) and (s15.c)..(s15.c) cutbefore bpath s7 cutafter bpath s15;
drawarrow p53;
label.top(btex 1 etex, point 0.21 of p53);

p57 = (s7.c)..controls (138.0, -15.5) and (s13.c)..(s13.c) cutbefore bpath s7 cutafter bpath s13;
drawarrow p57;
label.top(btex 5 etex, point 0.5 of p57);

p67 = (s9.c)..controls (144.5, -49.5) and (s13.c)..(s13.c) cutbefore bpath s9 cutafter bpath s13;
drawarrow p67;
label.top(btex 2 etex, point 0.27 of p67);

p71 = (s9.c)..controls (142.0, -79.0) and (s15.c)..(s15.c) cutbefore bpath s9 cutafter bpath s15;
drawarrow p71;
label.top(btex 4 etex, point 0.33 of p71);

p75 = (s11.c)..controls (145.0, -101.0) and (s13.c)..(s13.c) cutbefore bpath s11 cutafter bpath s13;
drawarrow p75;
label.top(btex 4 etex, point 0.26 of p75);

p79 = (s11.c)..controls (144.0, -116.0) and (s15.c)..(s15.c) cutbefore bpath s11 cutafter bpath s15;
drawarrow p79;
label.top(btex 3 etex, point 0.39 of p79);

p83 = (s13.c)..controls (192.5, -42.5) and (s17.c)..(s17.c) cutbefore bpath s13 cutafter bpath s17;
drawarrow p83;
label.top(btex 5 etex, point 0.5 of p83);

p87 = (s15.c)..controls (195.5, -81.0) and (s17.c)..(s17.c) cutbefore bpath s15 cutafter bpath s17;
drawarrow p87;
label.top(btex 2 etex, point 0.5 of p87);


% Finally, create labels

endfig;
end
