%{

%}

letter [a-zA-Z]
nonletspace [^a-zA-Z ]
ws [ \t]

%s N
%s S
%s M

%%


\\{letter}+ {printf("<cseq: %s>",yytext+1); BEGIN S;}
\\{nonletspace}       {printf("<csym: %s>",yytext+1); BEGIN M;}
\\[ ] {printf("<cspace>"); BEGIN S;}
\{ {printf("<{>"); BEGIN M;}
\} {printf("<}>"); BEGIN M;}
\%.*\n {printf("<comment>\n");}
<N>{ws}+  ;
<S>{ws}+  ;
<M>{ws}+ {printf("<space>"); BEGIN S;}
[^ \t\n] {ECHO; BEGIN M;}
<N>\n     {printf("<par>"); ECHO;}
<M>\n     {printf("<space>"); ECHO; BEGIN N;}
<S>\n     {ECHO; BEGIN N;}

%%

int main()
{
	yylex();
	return 0;
}
