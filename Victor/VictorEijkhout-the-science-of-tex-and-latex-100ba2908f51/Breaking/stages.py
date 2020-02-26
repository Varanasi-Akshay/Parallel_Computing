table = [ [0, 5, 4, 0, 0, 0, 0, 0, 0], #0
          [0, 0, 0, 1, 3, 4, 0, 0, 0], #1 & #2
          [0, 0, 0, 4, 2, 2, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 5, 1, 0], #3, #4, #5
          [0, 0, 0, 0, 0, 0, 2, 4, 0],
          [0, 0, 0, 0, 0, 0, 4, 3, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 5], #6 & #7
          [0, 0, 0, 0, 0, 0, 0, 0, 2]
          ]
final = len(table); 

# test the table
for t in range(final):
    if len(table[t])!= final+1: print "bad table"

# the wrong way
calls = 0
def cost_from(n):
    global calls
    # if you're at the end, it's free
    if n==final: return 0
    # otherwise range over cities you can reach
    # and keep minimum value
    val = 0
    for m in range(n+1,final+1):
        # for all later cities
        local_cost = table[n][m]
        if local_cost==0: continue
        calls += 1
        # if there is a connection from here,
        # compute the minimum cost
        local_cost += cost_from(m)
        if val==0 or local_cost<val:
            val = local_cost
    return val
print "recursive minimum cost is",cost_from(0),"; #calls=",calls

# compute cost backwards
calls = 0
cost = (final+1)*[0]
for t in range(final-1,-1,-1):
    for i in range(final+1):
        local_cost = table[t][i]
        if local_cost==0: continue
        calls += 1
        local_cost += cost[i]
        if cost[t]==0 or local_cost<cost[t]:
            cost[t] = local_cost
print "backward minimum cost is",cost[0],"; #calls=",calls

# compute cost forward
cost = (final+1)*[0]
for t in range(final):
    for i in range(final+1):
        local_cost = table[t][i]
        if local_cost == 0: continue
        cost_to_here = cost[t]
        newcost = cost_to_here+local_cost
        if cost[i]==0 or newcost<cost[i]:
            cost[i] = newcost
print "forward minimum cost is",cost[final]

# # compute cost forward
# reach = [0]; cost = [0]
# def isin(e,l):
#     res = [1 for i in l if i==e]
#     return len(res)
# for t in range(final):
#     for i in range(final+1):
#         local_cost = table[t][i]
#         if local_cost == 0: continue
#         j = reach.index(t); cost_to_here = cost[j]
#         newcost = cost_to_here+local_cost
#         if isin(i,reach)==0:
#             print "init: cost to",i,"is",newcost
#             reach.append(i)
#             cost.append(newcost)
#         else:
#             j = reach.index(i); c = cost[j]
#             if newcost<c:
#                 print "cost to",i,"is lowered to",newcost
#                 cost[j] = newcost
#             else:
#                 print "no improvement of",newcost,"over",c
# print cost

