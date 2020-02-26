%{
#include "test.h"
%}

digit [0-9]

%%

[0-9] {return DIGIT;}
[ ] {return SPACE;}

%%
