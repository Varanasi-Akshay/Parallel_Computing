%{
 int lno=1;
%}

%%

.  {printf("%d,",atoi(yytext));}
\n {lno++; printf("\n");}

%%

int main() {
  yylex();
  return 0;
}
