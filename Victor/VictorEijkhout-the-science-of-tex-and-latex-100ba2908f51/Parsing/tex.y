%{

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define NCSNAMES 100
char *csnames[NCSNAMES]; int csnargs[NCSNAMES]; int ncs=0;

#define ARGLEN 5000
char argtext[ARGLEN]; int argloc,reading_arg=0;

%}

%token CHAR CONTROLSEQ CONTROLSYM CONTROLSPACE GROUPOPEN GROUPCLOSE SPACE

%%

text :
	CHAR
	| CHAR text {
	   if (reading_arg) {
	     argtext[argloc++] = yylval;
	   } else {
	     output_char(yylval);
	   }
	  }
	| CONTROLSEQ {
	   int ics,nargs; ics = yylval; nargs = csnargs[ics];
	  }
	| SPACE
	| CONTROLSYM
	| CONTROLSPACE
	| GROUPOPEN text GROUPCLOSE


%%

int registercs(char *name,int nargs)
{
  if (ncs==NCSNAMES-1) {
    printf("Can not register any more control sequences\n"); exit;}
  csnames[ncs] = strdup(name); printf("registring <%s> as %d\n",name,ncs);
  csnargs[ncs] = nargs;
  return ncs++;
}

int findcs(char *name)
{
  int i; printf("finding <%s>\n",name);
  for (i=0; i<ncs; i++)
    if (strcmp(name,csnames[i])==0) return i;
  return -1;
}
void output_char(int c)
{
  printf("%c",c);
  return;
}

int main(void)
{
  registercs("documentclass",1);
  yydebug=1; yyparse();
  return 0;
}
