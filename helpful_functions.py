"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Functions used in other codes. Extracting wanted simulations data.

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils

import operator

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

### returns occupancies for each person for data of 29/08 or 27/08
def occlistforeachperson(date):
    if date == 2908 : 
    
        number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
        
        number,occ67,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
        
        occ_list=np.concatenate([occ90,occ67])
    
        number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
        
    if date == 2708 : 
    
        number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
        number,occ75,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
        occ_list= np.concatenate([occ90,occ75])
    
        number_list =[90,75,60,45,36,30,24,18]
    
    N = len(occ_list)
    occ_list_person = []
    
    for i in range(N):
        for j in range(number_list[i//9]):
            occ_list_person.append(occ_list[i])
            
    return occ_list_person
            
   
### returns occupancy of 10/09 data
def occlist1009():
    
    occ=[]
    for i in range(6):
        number,occ96,avg,time,iter,avgvx = np.genfromtxt('D:/Project/pkr/96/1009_'+str(i)+'_occupancy96time90timestep005resultsbigH6timesMLC1.csv',delimiter=",")
        occ.append(occ96)
        
    number_list=[18,12,6]
    for i in range(27):
        number = number_list[i//9]
        number,occ18,avg,time,iter,avgvx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(i)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=",")
        occ.append(occ18)
        
    return occ
    
### returns number of pedestrians and average speed for 19/08 data
def numberspeedlist1908():
    
    numberlist=[]
    speed=[]
    for i in range(6):
        number,occ96,avg,time,iter,avgvx = np.genfromtxt('D:/Project/pkr/96/1009_'+str(i)+'_occupancy96time90timestep005resultsbigH6timesMLC1.csv',delimiter=",")
        numberlist.append(number)
        speed.append(avg)
        
    number_list=[18,12,6]
    for i in range(27):
        number = number_list[i//9]
        number,occ18,avg,time,iter,avgvx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(i)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=",")
        numberlist.append(number)
        speed.append(avg)
        
    return numberlist,speed
 
 
### returns density of 27/08 data
def denslist2708():
    
    number90,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    number75,occ75,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    return np.concatenate([number90/24,number75/24])
 
### returns density of 29/08 data
def denslist2908():
    
    number90,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number67,occ67,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([number90/24,number67/24])
 
### returns matrix_contact of one simulation, 29/08 data
def matrix2908(pc,number,iteration):
    return np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005matrixbigH9times'+pc+'.csv', delimiter=',')
    
###returns different occupancies for 29/08 data
def occlist2908():
    
    
    number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number,occ67,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([occ90,occ67])
 
###returns average x-speed for 29/08 data   
def avgvxlist2908():
    
    number,occ90,avg,time,iter,avgvx90=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number,occ67,avg,time,iter,avgvx67=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([avgvx90,avgvx67])
    
### returns occupancy of 27/08 data  
def occlist2708():
    
    number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    number,occ75,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    return np.concatenate([occ90,occ75])


### returns x,y,vx,vy,occ of 27/08 data
def parameters_2708(number,iteration): 
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9timesMLC1.csv', delimiter=',') #x_full[1]
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9timesMLC1.csv', delimiter=',') #v_full[0]
    vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH9timesMLC1.csv', delimiter=',')#v_full[1]
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return x,y,vx,vy,occ
    
### returns x,y,vx,vy,occ of 29/08 data
def parameters_2908_005(pc,number,iteration): 
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9times'+pc+'.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9times'+pc+'.csv', delimiter=',') #x_full[1]
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9times'+pc+'.csv', delimiter=',') #v_full[0]
    vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH9times'+pc+'.csv', delimiter=',')#v_full[1]
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return x,y,vx,vy,occ
    
### returns x,y,vx,vy,occ of chosen data
def parameters_005(date,pc,number,iteration):
    
    if date == 2708:
        return parameters_2708(number,iteration)
    
    if date == 2908:
        return parameters_2908_005(pc,number,iteration)
        
### returns occupancy of one simulation, 29/08 data
def occ_2908_oneperson(pc,number,iteration):

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return occ
    
   
### returns x,y,r,occ of 27/08 data for one simulation
def parameters_2708anim(number,iteration): 
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1.csv', delimiter=',') 
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9timesMLC1.csv', delimiter=',') 
    r = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005rbigH9timesMLC1.csv', delimiter=',') 

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return x,y,r,occ  
    
### returns x,y,vx,r,occ of 29/08 data
def parameters_2908_005anim(pc,number,iteration): 
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9times'+pc+'.csv', delimiter=',') 
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9times'+pc+'.csv', delimiter=',')
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9times'+pc+'.csv', delimiter=',')
    
    r = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005rbigH9times'+pc+'.csv', delimiter=',') 
     
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return x,y,vx,r,occ
    

### returns x,y,vx,r,occ of chosen data           
def parameters_005anim(date,pc,number,iteration):
    
    if date == 2708:
        return parameters_2708anim(number,iteration)
    
    if date == 2908:
        return parameters_2908_005anim(pc,number,iteration)
    
### returns time of one simulation, 29/08 data
def time_2908_005anim(pc,number,iteration): 

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return time
    
### returns time of one simulation, 27/08 data
def time_2708anim(number,iteration): 
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return time  
 
 
### returns time of chosen data
def time_005anim(date,pc,number,iteration):
    
    if date == 2708:
        return time_2708anim(number,iteration)
    
    if date == 2908:
        return time_2908_005anim(pc,number,iteration)


        
### returns x_full and v_full of chosen data
def parameters_xvfull(date,number,iteration):
    if date == 2708 : 
        x,y,vx,vy,occ = parameters_2708(number,iteration)
    if date == 2908 : 
        x,y,vx,vy,occ = parameters_2908_005('MLC1',number,iteration)
    
    people = vx.shape[0]
    Time = vx.shape[1]
    x_full = np.zeros([2,people,Time])
    x_full[0]=x
    x_full[1]=y
    v_full = np.zeros([2,people,Time])
    v_full[0]=vx
    v_full[1]=vy
    return x_full, v_full

    