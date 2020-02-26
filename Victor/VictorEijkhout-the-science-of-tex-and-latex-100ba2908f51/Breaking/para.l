letter [a-zA-Z0-9]
punct [.,;:!?()'"]

%%

({punct}|{letter})+ {ECHO; printf("\n");}
.  |
\n ;
