def cost(ar,pre,post):
    print "cost of",pre,ar,post
    if len(ar)==1:
        return 0
    else:
        cr = 0; ir = 0
        for i in range(len(ar)-1):
            arl = ar[:(i+1)]; arr = ar[(i+1):]
            if arl[len(arl)-1][1]!=arr[0][0]:
                print "Incompatible:",arl,arr
            cl = cost(arl,pre,post); cr = cost(arr,pre,post)
            c = cl+cr+ arl[0][0] * arr[0][0] * arr[len(arr)-1][1]
            print "split",arl," / ",arr," => ",c
            if cr==0:
                cr = c; ir = i
            elif c<cr:
                cr = c; ir = i
        print "minimum cost",cr,"splitting at",ir
        return cr

sizes = [25,10,17,18,19,43,41,7]
matrices = []
for i in range(len(sizes)-1):
    matrices.append(sizes[i:(i+2)])
cost(matrices,[],[])
