%{
#include <stdlib.h>
#include <stdio.h>
#include "calc3.h"
void yyerror(char*);
int varindex(char *var);
%}

%%

[ \t]+ ;
(([0-9]+(\.[0-9]*)?)|([0-9]*\.[0-9]+)) {
            yylval.dval = atof(yytext);
            return DOUBLE;}
[-+*/=]    {return *yytext;}
"("        {return *yytext;}
")"        {return *yytext;}
[a-z][a-z0-9]* {
            yylval.ivar = varindex(yytext);
            return NAME;}
\n         {return *yytext;}
.          {char msg[25];
            sprintf(msg,"%s <%s>","invalid character",yytext);
            yyerror(msg);}
