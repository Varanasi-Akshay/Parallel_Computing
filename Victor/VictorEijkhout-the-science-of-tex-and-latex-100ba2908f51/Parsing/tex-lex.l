%{

#include <stdlib.h>
#include "tex.h"
int findcs(char *name);
int registercs(char *name,int nargs);
extern int yylval;

%}

letter [a-zA-Z]
nonletspace [^a-zA-Z ]
ws [ \t]

%s N
%s S
%s M

%%


\\{letter}+      {int ics; BEGIN S;
		  ics = findcs(yytext+1);
		  if (ics==-1) ics = registercs(yytext+1,-2);
		  yylval = ics; return CONTROLSEQ;}
\\{nonletspace}  {BEGIN M; return CONTROLSYM;}
\\[ ]            {BEGIN S; return CONTROLSPACE;}
\{               {BEGIN M; return GROUPOPEN;}
\}               {BEGIN M; return GROUPCLOSE;}
\%.*\n           ;
<N>{ws}+         ;
<S>{ws}+         ;
<M>{ws}+         {BEGIN S; return SPACE;}
[^ \t\n]         {BEGIN M; yylval = *yytext; return CHAR;}
<N>\n            ;
<M>\n            {ECHO; BEGIN N; return SPACE;}
<S>\n            {ECHO; BEGIN N;}

%%
