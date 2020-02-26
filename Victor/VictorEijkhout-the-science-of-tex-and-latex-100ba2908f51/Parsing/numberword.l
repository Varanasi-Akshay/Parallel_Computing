digit [0-9]
letter [A-Za-z]
punct [,.;!?]
nonblank [^ \t\n]

%{

  int number=0,word=0,unknown=0,ident=0;

%}

%%

{digit}+{punct}?  {number++;}
{letter}+{punct}? {word++;}
{letter}+{digit}+{nonblank}* {ident++;}

{nonblank}*       {unknown++;}

%%
int main () {
  yylex();
  printf("#words=%d, #numbers=%d, #ident=%d, #other=%d\n",
	 word,number,ident,unknown);
  return 0;
}
