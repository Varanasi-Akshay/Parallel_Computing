%{
  void   calc();

  float  result=0;
  int    numcount=0, opcount=0;
  int    nums[500];
  char   ops[500];
%}
digit    [0-9]
operand  [-+*/]

%%

{digit}+  nums[numcount++] = atoi(yytext);
{operand} ops[opcount++] = (*yytext);
\n        calc();

%%

int main() {
  yylex();
  printf("The result is: %f \n", result);
  return 0;
}

void calc() { 
  int op1, op2, i;
  char op;

  if (numcount != opcount + 1) {
    printf("Illegal input steam!\n");
    exit(1);
  }

  if (numcount == 1) {
    result = nums[0];
    return;
  }

  op2 = nums[numcount-1];
  op1 = nums[numcount-2];
  op = ops[0];

  if (op == '+')
{
printf("op1=%d op2=%d op=%c\n", op1, op2, op);
    result = op1 + op2;
printf("result=%f\n", result);
}
  else if (op == '-')
    result = op1 - op2;
  else if (op == '*')
    result = op1 * op2;
  else if (op == '/')
    result = 1.0 * op1 / op2;
  else {
    printf("Illegal operand!\n");
    exit(1);
  }

printf("result=%d\n", result);

  for (i=0; i<opcount-1; i++) {
    op1 = nums[numcount-3-i];
    op2 = result;
    op = ops[i+1];

    if (op == '+')
      result = op1 + op2;
    else if (op == '-')
      result = op1 - op2;
    else if (op == '*')
      result = op1 * op2;
    else if (op == '/')
      result = 1.0 * op1 / op2;
    else {
      printf("Illegal operand!\n");
      exit(1);
    }
  }

  return;
}

  
