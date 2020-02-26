%{

#include "anbn.h"

%}

%%

[ab] {return *yytext;}
\n   {return *yytext;}
.    {yyerror("unrecognized character");}
