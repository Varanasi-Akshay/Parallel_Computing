#! /usr/bin/env python

import sys

max_line_length = 60

#
# we simulate stretch/shrink by having double space by default
#
def line_length(words):
    l = 2*(len(words)-1)
    for w in words:
        l += len(w)
    return l
#
# ratio = -1 : shrink each double space to one
# ratio = 1 : stretch each double space to three
#
def compute_ratio(line):
    spaces = len(line)-1
    need = 1.*(max_line_length-line_length(line))
    #print "ratio:",need,spaces
    if spaces==0: return 10000
    else: return need/spaces

def set_paragraph(para):
    for l in range(len(para)-1):
        line = para[l]
        set_line(line)
    set_last_line(para[len(para)-1])
def set_line(line):
    shortfall = max_line_length-line_length(line)
    for w in range(len(line)-1):
        sys.stdout.write(line[w])
        if shortfall>0:
            sys.stdout.write('   '); shortfall = shortfall-1
        elif shortfall<0:
            sys.stdout.write(' '); shortfall = shortfall+1
        else:
            sys.stdout.write('  ')
    sys.stdout.write(line[len(line)-1])
    print "  ",compute_ratio(line)
def set_last_line(line):
    for w in range(len(line)-1):
        sys.stdout.write(line[w])
        sys.stdout.write('  ')
    sys.stdout.write(line[len(line)-1])
    print

paragraph = []
while 1:
    try:
        a = raw_input()
        paragraph.extend(a.split())
    except (EOFError):
        break

def final_report():
    global cost,nwords,paragraph
    print "Can break this paragraph at cost",minimum_cost(nwords-1)
    cur = len(paragraph)-1; broken = []
    while cur!=-1:
        prev = best_prev(cur)
        #print cur," cost:",cost[cur],"; connect to",prev
        line = paragraph[prev+1:cur+1]
        broken.insert(0,line)
        cur = prev;
    set_paragraph(broken)

def init_costs():
    global cost
    nul = [0,0,0]
    cost = len(paragraph)*[ 0 ]
    for i in range(len(paragraph)):
        cost[i] = nul[:]
        for j in range(3):
            cost[i][j] = {'cost':10000, 'from':-2}
    for j in range(3):
        cost[0][j] = {'cost':10000, 'from':-1}
def stretch_type(ratio):
    if ratio<-.5: return 0
    elif ratio<.5: return 1
    else: return 2
def minimum_cost_and_type(w):
    global cost
    c = 10000; t = 0
    for type in range(3):
        nc = cost[w][type]['cost']
        if nc<c:
            c = nc; t = type
    return [c,t]
def minimum_cost(w):
    [c,t] = minimum_cost_and_type(w)
    return c
def update_cost(a,w,ratio):
    global cost
    type = stretch_type(ratio)
    to_here = 100*abs(ratio)**2
    if a>0:
        [from_there,from_type] = minimum_cost_and_type(a-1)
        to_here += from_there
    else: from_there = 0
    print "so far best of type",type,":",\
          cost[w][type]['cost'],"over",cost[w][type]['from']
    if cost[w][type]['cost']==0 or to_here<cost[w][type]['cost']:
        cost[w][type]['cost'] = to_here; cost[w][type]['from'] = a-1
        print "better breakpoint over",a-1,":",\
              from_there,"+",abs(ratio),"=",to_here
def report_cost(w):
    global cost
    print "minimum cost breaking after",w,":",
    for t in range(3):
        print cost[w][t]['cost'],"over",cost[w][t]['from'],";",
    print
def best_prev(cur):
    global cost
    best = 0; prev = -2
    for t in range(3):
        np = cost[cur][t]['from']
        if np!=-2:
            nc = cost[cur][t]['cost']
            if best==0 or nc<best:
                best = nc; prev = np
    if prev==-2:
        print ">>>> ERROR could not find connection from",cur
    else: return prev

init_costs()
active = [0]
nwords = len(paragraph)
for w in range(1,nwords):
    # compute the cost of breaking after word w
    print "Recent word",w
    for a in active:
        line = paragraph[a:w+1]
        if w==nwords-1:
            ratio = 0 # last line will be set perfect
        else:
            ratio = compute_ratio(line)
            print "..line=",line,"; ratio=",ratio
        if ratio<-1:
            active.remove(a)
            print "active point",a,"removed"
        else:
            update_cost(a,w,ratio)
    report_cost(w)
    active.append(w)
    print
final_report()
