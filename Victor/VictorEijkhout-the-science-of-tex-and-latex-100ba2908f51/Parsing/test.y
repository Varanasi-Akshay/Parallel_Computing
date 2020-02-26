%token DIGIT SPACE

%%

all : one spaces two ;
one : DIGIT ;
two : DIGIT ;
spaces : SPACE ;
        | spaces SPACE ;

%%

int main(void) {
  yydebug = 1; yyparse();
  return 0;
}
