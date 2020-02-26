%{

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define NCSNAMES 100
char *csnames[NCSNAMES]; int csnargs[NCSNAMES]; int ncs=0;
int env[100],nenv=0;
int verticalmode = 1;
int lineno = 1;

%}

%token LETTER CHAR WORD
%token BEGINCS ENDCS CONTROLSEQ CONTROLSYM CONTROLSPACE
%token GROUPOPEN GROUPCLOSE

%%

latexfile : 
        documentclass environment
	| error wordarg environment {printf("No document class\n");}
        ;
documentclass :
        CONTROLSEQ wordarg 
            {int ics; ics = findcs("documentclass");
	      if ($1!=ics) {
		printf("Expecting \\documentclass\n"); YYABORT;}
	      printf("Using documentclass <%s>\n",$2);}
environment :
        env_open text env_close ;
env_open :
        BEGINCS wordarg {env_push($2);}
env_close :
        ENDCS wordarg 
            {int open=env_pop();
             if (!(strcmp((char*)open,(char*)$2)==0))
	       yyerror("Environment mismatch");
	    }
text :  ;
        | WORD text
        | environment text ;
wordarg :
        GROUPOPEN WORD GROUPCLOSE {$$ = $2;}
        ;
spaces : ;
        | ' ' spaces ;

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
  int loc,i; 
  loc = -1; /*printf("finding <%s>",name);*/
  for (i=0; i<ncs; i++)
    if (strcmp(name,csnames[i])==0) {
      loc = i;
    }
  /*printf("=%d\n",loc);*/
  return loc;
}

void env_push(int e) {
  printf("opening environment <%s>\n",(char*)e);
  env[nenv++] = e;
}

int env_pop(void) {
  int e = env[--nenv];
  printf("need closing: <%s>\n",(char*)e);
  return e;
}

void output_char(int c)
{
  printf("%c",c);
  return;
}

void yyerror(char *s)
{
  printf("Parsing failed in line %d because of %s\n",lineno,s);
  return;
}

int main(void)
{
  registercs("documentclass",1);
  registercs("begin",1); registercs("end",1);
  yydebug=0; 
  yyparse();
  return 0;
}
