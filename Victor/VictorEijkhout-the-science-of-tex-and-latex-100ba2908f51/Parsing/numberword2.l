digit [0-9]
letter [A-Za-z]
punct [,.;!?]
blank [ \t]
nonblank [^ \t\n]

%{

  int line=0,word=0,number=0,character=0;

%}

%%

{digit}+          {number++;word++;}
{nonblank}+       {word++; character+=yyleng;}
\n                {line++;character++;}
.                 character++;

%%
int main () {
  yylex();
  printf("%d %d %d %d\n",line,word,number,character);
  return 0;
}
