"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Code in relation with transmission rate, section 4.11

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils
import helpful_functions as hf 

        
    
### computation of data for plot_infectionrate_density_all(), data of the 29/08
def local_density_2908_allperson():
    #computed local density for all person, doing the average
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    time_plot=np.arange(0,90.1,0.05)
    dens_list_meantot = []
    
    for i in range(108):
        iteration = i
        number = number_list[i//9]
        
        print(number,iteration)
        
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        
        timestep = 0.05
        N = vx.shape[0]
        Time = vx.shape[1]
        
        x_full = np.zeros([2,N,Time])
        x_full[0]=x
        x_full[1]=y
        
        v_full = np.zeros([2,N,Time])
        v_full[0]=vx
        v_full[1]=vy
        for j in range(number):
            dens_list = []
            for t in range(Time):
                dens = putils.local_density_current_transmission(x_full[:,:,t],v_full[:,:,t],[x[j,t],y[j,t]])
                dens_list.append(dens)
            dens_list_meantot.append(np.mean(dens_list))
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densityallpedestriansRequals1_5.csv',dens_list_meantot,delimiter=',')
    
    
    
### Plot of the figure 4.28
def plot_infectionrate_density_all():
    R=1.5
    dens_list_meantot=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densityallpedestriansRequals1_5.csv',delimiter=',')
    occall2908 = hf.occlistforeachperson(2908)
    # plt.xlabel('Occupancy')
    # plt.title('Average density around an infected individual over time')
    # plt.ylabel('Average density')
    # plt.plot(occall2908,dens_list_meantot,'o',markersize=1)
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    dens_list_allmean=np.zeros(108)
    std_list_allmean=np.zeros(108)
    begin = 0
    end = 0
    for i in range(108):
        number = number_list[i//9]
        begin=end
        end+=number
        dens_list_allmean[i]=np.mean(dens_list_meantot[begin:end])
        std_list_allmean[i]=np.std(dens_list_meantot[begin:end])
    occ2908 = hf.occlist2908()    
    dens2908 = hf.denslist2908()
    avgvx=hf.avgvxlist2908()
    plt.axhline(0.01,linestyle = ':',color='r',markersize=1, label =  "infections per second, constant")
    plt.plot(dens2908,dens_list_allmean*0.1,'o',color='b',markersize=2,label = "infections per second, density dependent")
    plt.plot(dens2908,dens_list_allmean*0.1/(10*avgvx),'o',color='g',markersize=2,label = "infections per 0.1 metre walked, density dependent")
    plt.legend()
    plt.xlabel('Density')
    plt.ylabel('Infection rate')
    title = "Infection rate through occupancy"
    plt.title(title)
    plt.show()
    
    
    
#list_avoir = [37,115,372,686,1697,1741] #density around 0


#x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',90,1)