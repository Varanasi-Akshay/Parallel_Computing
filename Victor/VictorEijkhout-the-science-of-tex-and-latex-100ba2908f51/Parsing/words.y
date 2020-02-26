%{

#include <stdlib.h>
#include <string.h>
  int yylex(void);
#include "words.h"
  int nwords=0;
#define MAXWORDS 100
  char *words[MAXWORDS];
%}

%token WORD

%%

text : ;
       | text WORD  ; {
            if ($2<0) printf("new word\n");
            else printf("matched word %d\n",$2);
                      }

%%

int find_word(char *w)
{
  int i;
  for (i=0; i<nwords; i++)
    if (strcmp(w,words[i])==0) {
      return i;
    }
  words[nwords++] = strdup(w);
  return -1;
}

int main(void)
{
  yyparse();
  printf("there were %d unique words\n",nwords);
}

