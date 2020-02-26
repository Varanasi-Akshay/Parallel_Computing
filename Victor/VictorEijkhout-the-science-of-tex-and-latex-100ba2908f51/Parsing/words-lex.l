%{

#include "words.h"
int find_word(char*);
extern int yylval;
%}

%%

[a-zA-Z]+ {yylval = find_word(yytext);
	   return WORD;}
.         ;
\n        ;

%%
