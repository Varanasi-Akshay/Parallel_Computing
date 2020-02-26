#! /usr/bin/env python

import matplotlib.pyplot as plt
import re
import subprocess
import sys

if len(sys.argv)<2:
    print "Usage: %s stampname" % sys.argv[0]
    sys.exit(1)

def parse_one_run(stampname):
    print "\n****************\n %s\n****************" % stampname
    big10 = 1000000000000
    communications = []; computations = []
    max_stamp = 0
    def number(n):
        return int( (int(n)%big10) / 1000 )
    with open(stampname,"r") as stampfile:
        Post1 = []; Start1 = []; Stop1 = []; Post2 = []; Start2 = []; Stop2 = []
        communication = []; computation = []; pcolours = []
        min_stamp = None; nprocs = None
        for step in stampfile:
            step = step.strip()
            if nprocs is None:
                nprocs,msec = step.split()
                nprocs = int(nprocs); msec = int(msec)
                print "detecting %d processors" % nprocs
                continue
            if len(step)==0:
                communications.append(communication)
                computations.append(computation)
                communication = []; computation = []; pcolours = []
                min_stamp = None
                continue

            # read a line of time stamps
            stamps = [ number(n) for n in step.split() ]

            # save the first post1 as min_stamp
            if min_stamp is None:
                min_stamp = stamps[0]
            post1,posted1,start1,stop1,post2,posted2,start2,stop2 = [ s-min_stamp for s in stamps ]
            communication.append( (post1,posted1-post1) )
            communication.append( (post2,posted2-post2) )
            # Computation episode 1
            computation.append( (start1,stop1-start1) )
            pcolours.append( 'blue' )
            # Computation episode 2
            computation.append( (start2,stop2-start2) )
            pcolours.append( 'orange' )
            # sanity check
            if start2<stop1:
                print "Strange stop1/start2 ordering:",stop1,start2

        communications.append(communication)
        computations.append(computation)
    return msec,communications,computations,pcolours
        
# print Post1,"\n", Start1,"\n", Stop1,"\n", Post2,"\n", Start2,"\n", Stop2

def plot_one_run(ax,msec,communications,computations,pcolours,scale=1,extratext=""):
    last_computation = computations[-1][-1]
    max_stamp = last_computation[0]+last_computation[1]
    fontsize = 12/scale
    voffset = 0; yticks = []; ylabels = []; procno=0
    for m,p in zip(communications,computations):
        #print "communication:",m
        yticks.append(voffset)
        ax.broken_barh( m, (voffset,4), facecolors='green', label='MPI' )
        ylabels.append("")

        #print "computation:",p
        yticks.append(voffset+5)
        ax.broken_barh( p, (voffset+5,4), facecolors=pcolours, label='Work')
        ylabels.append(str(procno))

        voffset += 10; procno += 1

    ax.set_ylim(0, voffset)
    ax.set_xlim(0,1.1*max_stamp)
    #ax.set_xlabel('time',fontsize=fontsize)
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(ylabels)
    ax.grid(True)
    ax.text(.05*max_stamp,voffset-4,stampname+"\ntotal runtime: %d%s" % (msec,extratext),
            verticalalignment='top',horizontalalignment='left',
            bbox={'facecolor':'white'}, fontsize=fontsize
            )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=fontsize)

#fig.suptitle()


if len(sys.argv)==2:
    #
    # chart for a single run
    #
    stampname = sys.argv[1]
    fig,ax = plt.subplots()
    msec,communications,computations,pcolours = parse_one_run(stampname)
    plot_one_run(ax,msec,communications,computations,pcolours)
    pdfname = re.sub("out","pdf",stampname)
else:
    #
    # chart for machine / program combination
    #
    machine = sys.argv[1]
    overlap = sys.argv[2]
    proc = subprocess.Popen(["make", "-f","../Makefile","listsizes","TACC_SYSTEM=%s" % machine],
                            stdout=subprocess.PIPE)
    (domsizes, err) = proc.communicate()
    domsizes = domsizes.strip().split()
    domsizes = [ int(s) for s in domsizes ]
    stampname = re.sub( "overlap","laptime-",overlap )
    fig,axs = plt.subplots(nrows=3,ncols=3)
    for isize,s in enumerate(domsizes): # [1000,3000,10000]
        # row of sizes
        axsize = axs[isize]
        for ithick,t in enumerate([1,10,100]):
            # specific subplot
            ax = axsize[ithick]
            stamprun = "%s-%d-%d.out" % (stampname,s,t)
            msec,communications,computations,pcolours = parse_one_run(stamprun)
            plot_one_run(ax,msec,communications,computations,pcolours,\
                         scale=3,extratext="\nsize: %d, thick: %d" % (s,t))
    pdfname = "laptime-%s.pdf" % overlap

#plt.show()
print "Writing to",pdfname
fig.savefig(pdfname)

# [(10, 50), (100, 20), (130, 10)], (20, 9), facecolors=('red', 'yellow', 'green'))
