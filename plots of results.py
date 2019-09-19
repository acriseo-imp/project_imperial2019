"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Plots of speed-occupancy relations.

"""

import matplotlib.pyplot as plt
import numpy as np
import ped_utils as putils
import helpful_functions as hf



def speedbellomo(density,gamma):
    return (1-np.exp(-gamma*(1/density - 1)))
   
### Figure 4.3

def comparizon_weidmann():
     #serie de 10 simulations du 27/08
     
    number75,occ75_18,avg75_18,time75_18,iter75_18,avgvx75_18=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    number90,occ90,avg90,time90,iter90,avgvx90=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    plt.plot(number75/24,avg75_18,'o',color='k',markersize=2,label='Average speed')
        
    plt.plot(number90/24,avg90,'o',color='k',markersize=2)
        

    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    plt.plot(number/24,avg90_18,'o',color='k',markersize=2)
    
    number,speed=hf.numberspeedlist1908()
    
    plt.plot(np.array(number)/24,speed,'o',color='k',markersize=2)
   
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.xlabel("Density")
    plt.ylabel("Average speed")
    plt.xlim([0,4.5])
    
    data_agent=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/Texts/Density, utility function/An Agent-Based Microscopic Pedestrian Flow 2.csv',delimiter=',')
    
    data_agent=data_agent[:,:3]
    
    plt.plot(data_agent[:,0],data_agent[:,1],'o',markersize=3, marker ='s',label="Weidmann data")
    plt.legend()
    plt.grid()
    plt.show()
   
### Figure 4.4
def compar05025():
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    plt.plot(occ90_18,avg90_18,'o',color='k',markersize=3,label='Average speed for a time step of 0.05')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='burlywood',markersize=3,label='Average speed for a time step of 0.05, only x component')
    
    number,occ90_75,avg90_75,time90_75,iter90_75,avgvx90_75=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    
    plt.plot(occ90_75,avg90_75,'o',color='b',markersize=3,label='Average speed for a time step of 0.025')
   
    plt.plot(occ90_75,avgvx90_75,'o',color='magenta',markersize=3, label = 'Average speed for a time step of 0.025, only x component')
    
    number,occ67_12,avg67_12,time67_12,iter67_12,avgvx67_12=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    
    plt.plot(occ67_12,avg67_12,'o',color='b',markersize=3)
   
    plt.plot(occ67_12,avgvx67_12,'o',color='magenta',markersize=3)
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    plt.title("Average speed = f(occupancy), big H, comparison between 0.05 and 0.025")

    plt.legend()
    plt.show()
    
  
### Figure 4.5
def pkrcomparperiodic():
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    plt.plot(occ90_18,avg90_18,'o',color='k',markersize=3,label='Average speed for a time step of 0.05')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='burlywood',markersize=3,label='Average speed for a time step of 0.05, only x component')
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesnoperiodicMLC4.csv',delimiter=',')
    
    plt.plot(occ90_18,avg90_18,'o',color='b',markersize=3,label='Average speed for a time step of 0.05, no periodic condition')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='magenta',markersize=3,label='Average speed for a time step of 0.05, only x component, no periodic condition')
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    plt.title("Average speed = f(occupancy), big H, comparison between periodic condition or not")

    plt.legend()
    plt.show()
 
 
### Figure 4.1
def time_simulation2908():
    
    liste_number = []
    liste_time = []
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    for i in range(108):
        iteration = i
        number = number_list[i//9]
        time = hf.time_2908_005anim('MLC1',number,iteration)
        liste_number.append(number)
        liste_time.append(time)
        
    for i in range(9):
        iteration = i
        number = 90
        time = hf.time_2708anim(number,iteration)
        liste_number.append(number)
        liste_time.append(time)
    
    number_list =[75,60,45,36,30,24,18]
        
    for i in range(63):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        time = hf.time_2708anim(number,iteration)
        liste_number.append(number)
        liste_time.append(time)
    
    liste_number.sort()
    liste_time.sort()
    
    plt.plot(liste_number,np.array(liste_time),'o',markersize=2)
    plt.grid(True, which="both")
    plt.ylabel("Time of computation")
    plt.xlabel("Number of pedestrians")
    plt.xticks(np.arange(0,100,10))
    plt.yticks(np.arange(0,650,50))
    plt.xlim([0,100])
    plt.ylim([0,650])


    plt.show()




    

