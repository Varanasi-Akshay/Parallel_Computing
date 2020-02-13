import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8]

y=[0.7690,0.7690/2,0.7690/3,0.7690/4,0.7690/5,0.7690/6,0.7690/7,0.7690/8]
#Data for Static without initialization
y1=[0.7690,
      0.5616,
      0.3562,
      0.3007,
      0.2423,
      0.2073,
      0.2006,
      0.2106
]

#Data for Static with initialization
y2=[  0.7701,
      0.3744,
      0.2700,
      0.2018,
      0.1678,
      0.1411,
      0.1723,
      0.1564]

ideal,=plt.plot(x,y,'x',label='ideal')      
ser_init,=plt.plot(x,y1,'+',label='ser_init')
par_init,=plt.plot(x,y2,'o',label='par_init')
plt.legend(handles=[ser_init,par_init,ideal])
plt.xlabel('No.of threads')
plt.ylabel('Time in sec')
plt.title('Comparision of serial and parallel initialization')
plt.show()
