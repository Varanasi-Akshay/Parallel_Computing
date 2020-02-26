%{

#include <stdlib.h>
#include <string.h>
#include "tex1.h"
int findcs(char *name);
int registercs(char *name,int nargs);
extern int yylval;
extern int verticalmode;
extern int lineno;

%}

letter [a-zA-Z]
letgit [a-zA-Z0-9]
nonletspace [^a-zA-Z ]
ws [ \t]

%s N
%s S
%s M

%%


\\begin          {int ics; ics = findcs(yytext+1);
                  BEGIN S; return BEGINCS;}
\\end            {int ics; ics = findcs(yytext+1);
                  BEGIN S; return ENDCS;}
\\{letter}+      {int ics; BEGIN S;
		  ics = findcs(yytext+1);
		  if (ics==-1) ics = registercs(yytext+1,-2);
		  yylval = ics; return CONTROLSEQ;
                 }
\\{nonletspace}  {BEGIN M; return CONTROLSYM;}
\\[ ]            {BEGIN S; return CONTROLSPACE;}
\{               {BEGIN M; return GROUPOPEN;}
\}               {BEGIN M; return GROUPCLOSE;}
\%.*\n           lineno++;
<N>{ws}+         ;
<S>{ws}+         ;
<M>{ws}+         {BEGIN S; yylval = ' '; return CHAR;}
{letgit}+        {BEGIN S; yylval = (int)strdup(yytext); return WORD;}
[^ \t\n]         {BEGIN M; yylval = *yytext; return CHAR;}
<N>\n            {lineno++; /*printf("newline->.\n");*/}
<M>\n            {BEGIN N; lineno++; /*printf("newline->space\n");*/
                  if (verticalmode) ; /*printf("..space ignored\n");*/
                  else return ' ';}
<S>\n            {BEGIN N; lineno++; /*printf("newline->.\n");*/}

%%
