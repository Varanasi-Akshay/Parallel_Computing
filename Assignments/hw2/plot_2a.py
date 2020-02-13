import matplotlib.pyplot as plt

x=[2,4,8,16,32,64]
#Data for ideal speedup
y1=[0.231220960617065,0.231220960617065/2,0.231220960617065/4,0.231220960617065/8,0.231220960617065/16,0.231220960617065/32]

#Data for actual speedup
y2=[ 0.231220960617065,0.119583129882812,5.530500411987305E-02,2.948880195617676E-002,1.750612258911133E-002,1.016092300415039E-002 ]
ideal,=plt.plot(x,y1,'x',label='Ideal runtime')
actual,=plt.plot(x,y2,'o',label='Actual runtime')
plt.legend(handles=[ideal,actual])
plt.xlabel('No.of tasks')
plt.ylabel('Time in sec')
plt.title('Comparision of Ideal and Actual runtime')
plt.show()
