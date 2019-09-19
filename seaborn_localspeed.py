"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Space-time diagrams of local speed

"""

import numpy as np
import matplotlib.pyplot as plt
import math

import pandas as pd
import ped_utils as putils
import helpful_functions as hf

list_x=np.arange(-4,4,0.05)


date = 2908
timestep = 0.05
already_data = False

number_list = [36,30,24]

list_iteration = np.arange(63,88)


### return local speed through time, for different points in list_x
def local_speed_seaborn(x,y,vx,vy,list_x):
    N=x.shape[0]
    Time=x.shape[1]
    x_full = np.zeros([2,N,Time])
    x_full[0]=x
    x_full[1]=y
    
    v_full = np.zeros([2,N,Time])
    v_full[0]=vx
    v_full[1]=vy
    
    nx = list_x.shape[0]
    data = np.zeros([nx,Time])
    
    for i in range(nx):
        for t in range(Time):
            data[i][t]=putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[list_x[i],0])
            
    return data
    

### Space time diagram
def spacetimediagramlocal(number,iteration,date,timestep,already_data):

    if already_data :
        
        x,z,v=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/data/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1xzvlocalspeed.csv',delimiter = ',')
        occ=hf.occ_2908_oneperson('MLC1',number,iteration)
        
    else : 
        
        x,y,vx,vy,occ = hf.parameters_005(date,'MLC1',number,iteration)
        x=x[:,int(10//timestep):]
        y=y[:,int(10//timestep):]
        vx=vx[:,int(10//timestep):]
        vy=vy[:,int(10//timestep):]
            
        data = local_speed_seaborn(x,y,vx,vy,list_x)
        
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/data/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1datalocalspeedremoved10.csv',data,delimiter = ',')
    
        N=list_x.shape[0]
        Time=x.shape[1]
        z0=np.linspace(0,90.1,num=1802,endpoint=False)
        z0=z0[int(10//timestep):]
        z=z0
        v=data[0]
        x=[]
        for t in range (Time):
            x=np.concatenate((x,[list_x[0]]))
        for i in range(1,N):
            for t in range (Time):
                x=np.concatenate((x,[list_x[i]]))
            
            v=np.concatenate((v,data[i]))
            z=np.concatenate((z,z0))
        
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/data/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1xzvlocalspeedremoved10.csv',(x,z,v),delimiter = ',')
        
    
    index_to_remove = np.where (1 < v) #removing unrelevant speeds
    x=np.delete(x,index_to_remove)
    v=np.delete(v,index_to_remove)
    z=np.delete(z,index_to_remove)
    
    index_to_remove = np.where (0 > v)
    x=np.delete(x,index_to_remove)
    v=np.delete(v,index_to_remove)
    z=np.delete(z,index_to_remove)
    df = pd.DataFrame({'x':x, 'y':z, 'z':v})
    
    plt.figure()
    points=plt.scatter(x,z,s=1,c=v,cmap="Spectral")
    cbar=plt.colorbar(points)
    cbar.set_label(label="Speed (m/s)",size=15)
    plt.xlabel('X',fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    plt.ylabel('Time (s)',fontsize=15)
    plt.tick_params(axis = 'both', labelsize = 15)
    #plt.title(str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", timestep of 0.05")
    plt.xlim([-4,4])
    plt.ylim([10,90])
    plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/'+str(number)+'/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between0and06_size1timex_005_spectral_remove10schangefontsize.png')
    plt.close()
                
#                 
# for i in range(len(number_list)):
#     for j in range(9):
#         iteration = list_iteration[j+9*i]
#         print(number_list[i],iteration)
#         spacetimediagramlocal(number_list[i],iteration,date,timestep,already_data) 
# 

spacetimediagramlocal(75,26,2908,0.05,True)

        
