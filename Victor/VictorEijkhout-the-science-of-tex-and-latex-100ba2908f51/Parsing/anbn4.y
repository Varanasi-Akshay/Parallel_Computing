%{

int depth=0;

%}

%%

S : A B '\n'     {if (depth==0) printf("matched\n");
                  else if (depth>0) printf("too many a\n");
                  else printf("too many b\n"); }

A : 'a' A {depth++;}
  | ;

B : ;
  | 'b' B {depth--;}

%%
