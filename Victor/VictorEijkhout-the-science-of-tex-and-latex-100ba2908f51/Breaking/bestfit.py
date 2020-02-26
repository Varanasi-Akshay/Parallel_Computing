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
    print
    cur = len(paragraph)-1; broken = []
    while cur!=-1:
        prev = best_prev(cur)
        print cur," cost:",cost[cur],"; connect to",prev
        line = paragraph[prev+1:cur+1]
        broken.insert(0,line)
        cur = prev;
    print
    set_paragraph(broken)

def init_costs():
    global cost
    cost = [ {'cost':0, 'from':0} for i in range(len(paragraph)) ]
    cost[0] = {'cost':10000, 'from':-1}
def update_cost(a,w,ratio):
    global cost
    to_here = 100*abs(ratio)**3
    if a>0:
        from_there = cost[a-1]['cost']
        to_here = to_here+from_there
    else: from_there = 0
    #print "so far best:",cost[w]['cost'],"over",cost[w]['from']
    if cost[w]['cost']==0 or to_here<cost[w]['cost']:
        cost[w]['cost'] = to_here; cost[w]['from'] = a-1
        print "better breakpoint over",a-1,":",\
              from_there,"+",abs(ratio),"=",to_here
def report_cost(w):
    global cost
    print "minimum cost breaking after",w,":",\
          cost[w]['cost'],"over",cost[w]['from']
def minimum_cost(w):
    return cost[w]['cost']
def best_prev(w):
    return cost[w]['from']

init_costs()
active = [0]
nwords = len(paragraph)
for w in range(1,nwords):
    # compute the cost of breaking after word w
    print "Recent word",w
    for a in active:
        line = paragraph[a:w+1]
        ratio = compute_ratio(line)
        if w==nwords-1 and ratio>0:
            ratio = 0 # last line will be set perfect
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
