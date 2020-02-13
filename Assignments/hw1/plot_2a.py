import matplotlib.pyplot as plt

x=[1,2,4,8,16,32,64]
#Data for ideal speedup
y1=[1.621600,1.621600/2,1.621600/4,1.621600/8,1.621600/16,1.621600/32,1.621600/64]

#Data for actual speedup
y2=[ 1.621600,0.8444000,0.4419000,0.2293000,0.2027000,0.1966000,0.2132000 ]
ideal,=plt.plot(x,y1,'+',label='Ideal speedup')
actual,=plt.plot(x,y2,'o',label='Actual speedup')
plt.legend(handles=[ideal,actual])
plt.xlabel('No.of threads')
plt.ylabel('Time in sec')
plt.title('Comparision of Ideal and Actual speedup')
plt.show()
