%{

int depth=0;

%}

%%

S : AB '\n'     {printf("depth=%d\n",depth);}
  | A AB '\n'   {printf("too many a's\n");}
  | AB B '\n'   {printf("too many b's\n");}
AB : 'a' AB 'b' {depth++;}
   |            ;
A : 'a'
  | 'a' A
B : 'b'
  | 'b' B

%%
