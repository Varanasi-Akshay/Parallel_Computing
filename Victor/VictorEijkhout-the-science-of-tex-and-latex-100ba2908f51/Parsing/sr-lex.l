%{

#include "sr.h"
extern int yylval;

%}

%%

[0-9]+ {yylval = atoi(yytext); return N;}
[+\n-] {return *yytext;}

