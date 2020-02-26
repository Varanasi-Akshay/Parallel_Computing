%x COMM

%%

.          |
\n         ECHO;
"/*"       BEGIN COMM;
<COMM>"*/" BEGIN INITIAL;
<COMM>.    |
<COMM>\n   ;

%%
