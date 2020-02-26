#!/usr/bin/env python

table = [ [1, 5, 4, 3, 2, 9, 8, 7, 6], #0
          [6, 2, 9, 1, 3, 4, 2, 3, 4], #1 & #2
          [1, 1, 2, 4, 2, 2, 5, 6, 7],
          [8, 2, 7, 3, 6, 5, 5, 1, 1], #3, #4, #5
          [3, 3, 2, 1, 8, 2, 2, 4, 5],
          [5, 4, 1, 1, 2, 7, 4, 3, 3],
          [9, 1, 2, 3, 7, 6, 2, 3, 5], #6, #7, #8
          [1, 4, 2, 5, 6, 2, 8, 3, 2],
          [3, 1, 5, 6, 8, 9, 3, 2, 2]
          ]
#table = [ [1,5,4],[2,3,3],[4,1,2]]
ntowns = len(table)

path = ntowns * [0]
def shortest_path(start,through,lev):
    global calls
    calls = calls+1
    if len(through)==0:
        return table[start][0]
    else:
        l = 0
        for dest in through:
            left = through[:]; left.remove(dest)
            ll = table[start][dest]+shortest_path(dest,left,lev+1)
            if l==0 or ll<l:
                l = ll
        return l

to_visit = range(1,ntowns); calls = 0
s = shortest_path(0,to_visit,0)
print "shortest route has length",s,"(",calls,"calls)"

to_visit = range(1,ntowns-1); calls = 0
s = shortest_path(0,to_visit,0)
print "shortest route has length",s,"(",calls,"calls)"

to_visit = range(1,ntowns-2); calls = 0
s = shortest_path(0,to_visit,0)
print "shortest route has length",s,"(",calls,"calls)"

to_visit = range(1,ntowns-3); calls = 0
s = shortest_path(0,to_visit,0)
print "shortest route has length",s,"(",calls,"calls)"

def shortest_path_through(set,current):
    global calls
    calls = calls+1
    if len(set)==0: return table[current][0]
    else:
        shortest = 0
        for through in set:
            remain = set[:]; remain.remove(through)
            ldist = table[current][through]+\
                    shortest_path_through(remain,through)
            if shortest==0 or ldist<shortest: shortest = ldist
        return shortest

cities = range(1,ntowns)
shortest = 0
calls = 0
for first in cities:
    remain = cities[:]; remain.remove(first)
    lshort = table[0][first] + shortest_path_through(remain,first)
    if shortest==0 or lshort<shortest: shortest = lshort
    
print "shortest route again",shortest,"(",calls,"calls)"

# last = ntowns-1
# shortest = 0
# for semi in range(ntowns-1):
#     remain = range(ntowns)
#     remain.remove(semi)
#     ldist = table[semi][last]+shortest_path_through(remain,semi,last)
#     if shortest==0 or ldist<shortest: shortest = ldist
