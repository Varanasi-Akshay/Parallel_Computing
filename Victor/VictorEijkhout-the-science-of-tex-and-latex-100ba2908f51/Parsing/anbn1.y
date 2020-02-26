%{

int depth=0;

%}

%%

S : AB '\n'     {printf("depth=%d\n",depth);}
AB : 'a' AB 'b' {depth++;}
   |            ;

%%
