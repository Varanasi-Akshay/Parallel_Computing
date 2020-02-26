#!/usr/bin/env python

import re
import sys

if len(sys.argv)<2:
    print "Usage: %d outputfile" % sys.argv[0]
    sys.exit(1)

types = { "1 overlap1":"no communication", 
          "2 overlap2f":"contiguous data",
          "3 overlap2":"packed data",
          # "4 overlap4":"ordered for overlap",
          "5 overlap4h":"pure helper thread",
          "6 overlap4x":"helper, mixed",
          # "7 overlap1p":"seq, two thread",
          # "8 overlap4t":"nthread=2, regions",
          # "9 overlap4p":"nthread=2, total",
      }
outfilename = sys.argv[1]
with open(outfilename,'r') as outfile:
    Nold = 1
    hold = 0
    norm = 37
    times = {}
    # for t in types.keys():
    #     times[ t.split()[1] ] = 0
    labels = []
    for line in outfile:
        if re.search("thick",line):
            line = line.strip().split()
            N = line[3].split("=")[1]; N = int(N)
            h = line[5].split("=")[1]; h = int(h)
            if h!=hold :
                if len(times)>0:
                    str = "(x10^{%d}): " % norm
                    for t in sorted(types.keys()):
                        tt = t.split()[1]
                        if tt in times.keys():
                            tim = times[tt]
                        else: time = 0
                        str += "%s : %5.2f, " % (types[t],tim)
                    print str
                hold = h
                times = {}
                for t in types.keys():
                    times[ t.split()[1] ] = 0
                labels = []; nold = 37
            if N>Nold:
                print ; Nold = N
            print "N=%d, boundary thickness=%d" % (N,h)
        elif re.search("runtime",line):
            line = line.strip().split()
            m = line[0] # method
            labels.append(m)
            # normalization
            n = line[2].split("e")[1]; n = int(n)
            if norm==37:
                norm = n
            # elif n!=norm:
            #     print "Normalization problem: %d vs %d" % (n,norm)
            # time
            t = float(line[2].split("e")[0])
            for i in range(0,n-norm):
                t *= 10
            times[m] = t
    str = "(x10^{%d}): " % norm
    for t in sorted(types.keys()):
        tt = t.split()[1]
        if tt in times.keys():
            tim = times[tt]
        else: time = 0
        str += "%s : %5.2f, " % (types[t],tim)
    print str
                    # print "recording types:",types
                    # print "recording times:",times
