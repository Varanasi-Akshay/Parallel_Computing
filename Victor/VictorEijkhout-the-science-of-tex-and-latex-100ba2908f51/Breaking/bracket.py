def multcost(v123):
    if len(v123)!=3:
        print "ERROR: should be length 3"
        return 0
    else:
        return v123[0]*v123[1]*v123[2]
print multcost([5,2,3])
def linearcost(v):
    if len(v)==3:
        return multcost(v)
    else:
        vnew = v[2:]; vnew.insert(0,v[0])
        return multcost(v[:3])+linearcost(vnew)
print linearcost([5,2,3,4])
# 5*2*3 + 5*3*4 = 30+60 = 90
print linearcost([5,2,3,4,10])
# 5*2*3 + 5*3*4 + 5*4*10 = 30+60+200 = 290
def splitcost1(v,i):
    if len(v)==3:
        return multcost(v)
    else:
        inew = i; vnew = []; vnew[:] = v;
        if inew>len(v)-2: inew=len(v)-2;
        del vnew[inew];
        return multcost(v[(inew-1):(inew+2)])+splitcost1(vnew,inew)
print splitcost1([5,2,3,4,10],1)
# again 290
print splitcost1([5,2,3,4,10],3)
# 3*4*10 + 2*3*10 + 5*2*10 = 120+60+100 = 280
def mincost(v,l):
    if len(v)==3:
        return multcost(v)
    else:
        mc = 0
        for i in range(1,len(v)-1):
            vnew = []; vnew[:] = v;
            del vnew[i];
            mcc = multcost(v[(i-1):(i+2)])+mincost(vnew,l+1)
            if mc==0 or mcc<mc: mc = mcc
        return mc
print mincost([5,2,3,4,10],0)
