% Creator: Finomaton 0.8
% Creation-Date: Mon Oct 04 20:22:11 EDT 2004

input boxes
% Breadth of arrowheads (MetaPost default is 45)
ahangle := 35;
% Length of arrowheads (MetaPost default is 4bp)
ahlength := 4;
beginfig(1);
% for temporary paths (might be unused)
path p[];

% First, define and draw all the states
circleit.s1("2");
s1.c = (28.0, -30.0);
drawboxed(s1);

circleit.s3("3");
s3.c = (27.5, -78.0);
drawboxed(s3);

circleit.s5("1");
s5.c = (100.5, -51.0);
drawboxed(s5);


% Next, draw the lines
drawarrow (s1.c)..controls (61.5, -39.0) and (s5.c)..(s5.c) cutbefore bpath s1 cutafter bpath s5;

drawarrow (s3.c)..controls (63.5, -69.0) and (s5.c)..(s5.c) cutbefore bpath s3 cutafter bpath s5;


% Finally, create labels

endfig;
end
