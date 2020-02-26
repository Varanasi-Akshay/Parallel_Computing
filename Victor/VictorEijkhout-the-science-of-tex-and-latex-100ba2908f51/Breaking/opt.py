# list of indices in a row where elements are nonzero
row = [1,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,1,1]

# speedup factors for blocks of size 1,2,3,4 &c
redux = [1, .75, .4, .28]


time = (len(row)+1)*[0]
prev = (len(row)+1)*[0]
for p in range(len(row)):
    for b in range(len(redux)):
        q = p+b+1
        if q>=len(row)+1: continue
        if sum(row[p:q])==0:
            if time[p]<time[q]:
                time[q] = time[p]; prev[q] = p
            continue
        t = time[p]+(b+1)*redux[b]
        if time[q]==0:
            time[q] = t; prev[q] = p
        elif time[q]>t:
            time[q] = t; prev[q] = p
print "best time is",time[len(row)]

chain = []
p = len(row)
while p>0:
    chain.insert(0,p)
    p = prev[p]
print "block end points:",chain
