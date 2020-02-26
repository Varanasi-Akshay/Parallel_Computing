nul = 3*[0]
a = 3*[ 0 ]
for i in range(3):
    a [i] = nul[:]
    for j in range(3):
        a[i][j] = {'a':0,'b':0}
a[0][1]['a'] = 1
print a
