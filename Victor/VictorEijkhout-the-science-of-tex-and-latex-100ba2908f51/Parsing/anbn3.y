%{

int depth=0;

%}

%%

S : AB '\n'     {printf("depth=%d\n",depth);}
  | AB B '\n'   {printf("excess of b\n");}

AB : 'a' AB 'b' {depth++;}
   | ;

B : 'b'
  | 'b' B

%%
