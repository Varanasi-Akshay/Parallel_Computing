BEGIN { dx=0; m=0 }
dx==0            { print }
dx>0 && /^0/ && m==0     { print rmem ; m = 0 }
dx>0 && /^0/             { rmem = $0 ; m = 0 }
dx>0 && !/^0/            { print rmem $0 ; m = 1 }
/begin.DoxyCode/ { dx = 1; m = 1 }
/end.DoxyCode/   { dx = 0 }
