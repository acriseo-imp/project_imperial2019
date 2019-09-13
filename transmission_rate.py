import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils
import helpful_functions as hf 


def std_local_density_2908():
    
    dens_list_tot=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0.csv',delimiter=',')
    
    dens_list_std = np.zeros(108)
    for i in range(108):
        dens_list_std[i]=np.std(dens_list_tot[i,:])
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0std.csv',dens_list_std,delimiter=',')
    
    
    
def local_density_2908():
    #first test, supposed that infected is person 0
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    dens_list_tot=np.zeros([108,1802])
    
    time_plot=np.arange(0,90.1,0.05)
    
    
    for i in range(108):
        plt.figure()
        dens_list = []
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
        for t in range(Time):
            dens = putils.local_density_current_transmission(x_full[:,:,t],v_full[:,:,t],[x[0,t],y[0,t]])
            dens_list.append(dens)
        dens_list_tot[i,:]=dens_list
        plt.plot(time_plot,dens_list,'o',color='b',markersize=0.5)
        plt.xlabel("Time")
        plt.ylabel("Local density")
        title = "Local density through time, "+str(number)+" pedestrians, iteration "+str(iteration)+", occupancy of "+str(occ)[0:4]
        plt.title(title)
        plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/'+str(number)+'/'+title+'.png')
        plt.close()
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0.csv',dens_list_tot,delimiter=',')
    
    dens_list_mean = np.zeros(108)
    for i in range(108):
        dens_list_mean[i]=np.mean(dens_list_tot[i,:])
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0mean.csv',dens_list_mean,delimiter=',')
    
def plotmeandensity(): #mean density through occupancy
    
    std_list = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0std.csv',delimiter=',')
    dens_list_mean = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0mean.csv',delimiter=',')
    occ2908 = hf.occlist2908()
    plt.errorbar(occ2908,dens_list_mean,yerr=std_list,fmt='o',color='b',label = "infections per second, density dependent")
    plt.xlabel('Occupancy')
    plt.ylabel('Mean density')
    title = "Mean density through occupancy, pedestrian 0 infected"
    plt.title(title)
    plt.show()
    
def plotcontactrate(): #contact rate through occupancy
    
    R = 0.7
    pi = np.pi
    std_list = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0std.csv',delimiter=',')
    dens_list_mean = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/transmission rate/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_densitytestat0mean.csv',delimiter=',')
    occ2908 = hf.occlist2908()
    avgvx=hf.avgvxlist2908()
    plt.axhline(0.01,linestyle = ':',color='r',markersize=1, label =  "infections per second, constant")
    plt.plot(occ2908,dens_list_mean*0.1,'o',color='b',markersize=2,label = "infections per second, density dependent")
    plt.plot(occ2908,dens_list_mean*0.1/(avgvx),'o',color='g',markersize=2,label = "infections per 0.1 metre walked, density dependent")
    plt.legend()
    plt.xlabel('Occupancy')
    plt.ylabel('Infection rate')
    title = "Infection rate through occupancy"
    plt.title(title)
    plt.show()
    
        
    
    
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
    
def plotdensityall():
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
    # plt.errorbar(occ2908,dens_list_allmean,yerr=std_list_allmean,fmt='o')
    # plt.xlabel('Occupancy')
    # plt.title('Average density around an infected individual over time')
    # plt.ylabel('Average density')
    # plt.show()
        
    #return dens_list_allmean
    dens2908 = hf.denslist2908()
    avgvx=hf.avgvxlist2908()
    plt.axhline(0.01,linestyle = ':',color='r',markersize=1, label =  "infections per second, constant")
    plt.plot(dens2908,dens_list_allmean*0.1,'o',color='b',markersize=2,label = "infections per second, density dependent")
    plt.plot(dens2908,dens_list_allmean*0.1/(10*avgvx),'o',color='g',markersize=2,label = "infections per 0.1 metre walked, density dependent")
    plt.legend()
    plt.xlabel('Density')
    plt.ylabel('Infection rate')
    #title = "Infection rate through occupancy"
    #plt.title(title)
    plt.show()
    
    
    
#list_avoir = [37,115,372,686,1697,1741] #density around 0


#x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',90,1)