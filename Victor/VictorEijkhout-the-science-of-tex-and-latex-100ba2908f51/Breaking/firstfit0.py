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

def init_costs():
    global cost
    cost = len(paragraph)*[0]; cost[0] = 10000
def update_cost(a,w,ratio):
    global cost
    to_here = abs(ratio)
    if a>0: to_here = to_here+cost[a-1]
    if cost[w]==0 or to_here<cost[w]:
        cost[w] = to_here
        print "better breakpoint:",cost[a-1],"+",abs(ratio),"=",to_here
def report_cost(w):
    global cost
    print "minimum cost:",cost[w]
def final_report():
    global cost,nwords
    print "Can break this paragraph at cost",cost[nwords-1]

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
