%{
#include <stdlib.h>
#include <stdio.h>
#include "calc2.h"
void yyerror(char*);
%}

%%

[ \t]+ ;
(([0-9]+(\.[0-9]*)?)|([0-9]*\.[0-9]+)) {
            yylval.dval = atof(yytext);
            return DOUBLE;}
[-+*/=]    {return *yytext;}
"("        {return *yytext;}
")"        {return *yytext;}
[a-z]      {yylval.ivar = *yytext; 
            return NAME;}
\n         {return *yytext;}
.          {char msg[25];
            sprintf(msg,"%s <%s>","invalid character",yytext);
            yyerror(msg);}
