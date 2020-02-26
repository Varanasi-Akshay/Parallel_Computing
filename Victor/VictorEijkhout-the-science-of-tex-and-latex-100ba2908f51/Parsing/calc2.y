%{
#include <stdlib.h>
#include <stdio.h>
int yylex(void);
double var[26];
%}

%union { double dval; int ivar; }
%token <dval> DOUBLE
%token <ivar> NAME
%type <dval> expr
%type <dval> mulex
%type <dval> term

%%

program:
        line program
        | line 
line:
        expr '\n'             { printf("%g\n",$1); }
        | NAME '=' expr '\n'  { var[$1] = $3; }
expr:
        expr '+' mulex      { $$ = $1 + $3; }
        | expr '-' mulex    { $$ = $1 - $3; }
        | mulex             { $$ = $1; }
mulex:
        mulex '*' term      { $$ = $1 * $3; }
        | mulex '/' term    { $$ = $1 / $3; }
        | term              { $$ = $1; }
term:
        '(' expr ')'        { $$ = $2; }
        | NAME              { $$ = var[$1]; }
        | DOUBLE           { $$ = $1; }

%%

void yyerror(char *s)
{
    fprintf(stderr,"%s\n",s);
    return;
}

int main(void)
{
    /*yydebug=1;*/
    yyparse();
    return 0;
}

