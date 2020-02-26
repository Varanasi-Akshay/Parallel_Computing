%token N

%%

P : E '\n' {printf("Result: %d\n",$$);}

E : E '+' E {$$ = $1+$3;}
| E '-' E {$$ = $1-$3;}
  | N

%%

  int main(void)
{
  yyparse();
}
