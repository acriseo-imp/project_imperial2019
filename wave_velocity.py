import numpy as np
import matplotlib.pyplot as plt
import ped_utils as putils
import helpful_functions as hf

gamma=0.45
Vm=1.3
rhom=(100/24)

def f_prime(gamma,rho):
    return -(Vm*rhom*gamma/(rho**2))*(np.exp(-gamma*(rhom/rho - 1)))
    
    
def wave_velocity(gamma):
    
    number75,occ75_18,avg75_18,time75_18,iter75_18,avgvx75_18=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    number90,occ90,avg90,time90,iter90,avgvx90=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    rho=number75/24
    u=avgvx75_18
    
    plt.plot(occ75_18,rho*f_prime(gamma,rho)+u,'o',color='b',markersize=3,label='Average speed, only x component')
    
    rho=number90/24
    u=avgvx90
    
    plt.plot(occ90,rho*f_prime(gamma,rho)+u,'o',color='b',markersize=3)
    
    rho=number/24
    u=avgvx90_18
    plt.xlabel("Occupancy")
    plt.ylabel("w/k")
   
    plt.plot(occ90_18,rho*f_prime(gamma,rho)+u,'o',color='b',markersize=3)
    
    plt.show()