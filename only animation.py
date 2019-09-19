"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Code to create animations of simulations

"""


#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import helpful_functions as hf
#---------------------------


### Parameters
number = 75
iteration = 26
date = 2908
timestep = 0.05
removing_vx = False
removestr = ''

### Import data
x,y,vx,r,occ = hf.parameters_005anim(date,'MLC1',number,iteration)

iter = x.shape[0] #number of iterations

if removing_vx :
    removestr = 'removingvx'
    time = x.shape[1]
    new_x = np.zeros([iter,time])
    new_x[:,0]=x[:,0]
    avgvx=np.average(vx)
    for t in range(1,time):
        new_x[:,t]=new_x[:,t-1]+(vx[:,t]-avgvx)*timestep
        x[x[:,t]<-4]+=8
        x[x[:,t]>4]-=8
        
    x = new_x


fig, ax = plt.subplots()

### INITIALIZATION
tab_line=[]
for j in range (iter):
    if j == 25 :
        line,= ax.plot(x[j,0],y[j,0],marker='o',color='k',markersize=r[j]*100)
    else : 
        line,= ax.plot(x[j,0],y[j,0],marker='o',markersize=r[j]*100)
    tab_line.append(line)
ax.set_xlim(-4,4)
ax.set_ylim(-2,2) 
sizeline=(x[0].size)
plt.axhline(1.5,color = 'k') #wall at the top
plt.axhline(-1.5,color ='k') #wall at the bottom
plt.title(str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", timestep of 0.05")

def updatefig(i):
    """Updates figure each time function is called
    and returns new figure 'axes'
    """
    if i <2 : #this is to have only one point per person. If you want more points (it will then look like a snake), you only need to raise 2 at some number, and to put (i-number) below the else condition
        for j in range(iter):
            tab_line[j].set_data(x[j,:i],y[j,:i])
    else :
        for j in range(iter):
            tab_line[j].set_data(x[j,(i-1)%sizeline:i],y[j,(i-1)%sizeline:i])
    return tab_line

#updatefig is called 30 times, and each iteration of the image is 
#stored as a frame in an animation

ani = animation.FuncAnimation(fig, updatefig, frames=sizeline,interval=900) #900

#probably need to install ffmpeg before
ani.save('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005bigH9timesMLC1_1pointsleftviewofwalllessmarkers'+removestr+'.html')

plt.show()