import matplotlib.pyplot as plt


x=[1,2,3,4,5,6,7,8]
#Data for Static 100
y1=[0.7707,
      1.6463,
      1.1882,
      0.8837,
      0.7564,
      0.6691,
      0.6389,
      0.5765]

#Data for dynamic 100

y2=[  0.7695,
      2.3326,
      1.5369,
      1.4161,
      1.1303,
      0.9693,
      0.8602,
      0.8066]


#Data for Static 
y3=[0.7701,0.3744,0.2700,0.2018,0.1678,0.1411,0.1723,0.1564]

#Data for dynamic 100000

y4=[0.7665,0.5917,0.3939,0.3337,0.2632,0.2335,0.2126,0.2113]

y1,=plt.plot(x,y1,'+',label='Static 100')
y2,=plt.plot(x,y2,'o',label='Dynamic 100')      
y3,=plt.plot(x,y3,'x',label='Static')
y4,=plt.plot(x,y4,'*',label='Dynamic 100000')
plt.legend(handles=[y1,y2,y3,y4])
plt.xlabel('No.of threads')
plt.ylabel('Time in sec')
plt.title('Comparision of various scheduling methods')
plt.show()
