import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils

import operator

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


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
            
            
def occlist1908():
    
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
 
   
def denslist2708():
    
    number90,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    number75,occ75,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    return np.concatenate([number90/24,number75/24])
 
def denslist2908():
    
    number90,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number67,occ67,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([number90/24,number67/24])
    
def matrix2908(pc,number,iteration):
    return np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005matrixbigH9times'+pc+'.csv', delimiter=',')
    
def occlist2908():
    
    
    number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number,occ67,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([occ90,occ67])
    
def avgvxlist2908():
    
    number,occ90,avg,time,iter,avgvx90=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    number,occ67,avg,time,iter,avgvx67=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    return np.concatenate([avgvx90,avgvx67])
    
    

def occlist2708():
    
    number,occ90,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    number,occ75,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    return np.concatenate([occ90,occ75])


def parameters_005(date,pc,number,iteration):
    
    if date == 2708:
        return parameters_2708(number,iteration)
    
    if date == 2908:
        return parameters_2908_005(pc,number,iteration)

def parameters_2708(number,iteration): #have to put number = a.b, even for O -> O.O
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9timesMLC1.csv', delimiter=',') #x_full[1]
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9timesMLC1.csv', delimiter=',') #v_full[0]
    vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH9timesMLC1.csv', delimiter=',')#v_full[1]
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return x,y,vx,vy,occ
    

def parameters_2908_005(pc,number,iteration): #have to put number = a.b, even for O -> O.O
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9times'+pc+'.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9times'+pc+'.csv', delimiter=',') #x_full[1]
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9times'+pc+'.csv', delimiter=',') #v_full[0]
    vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH9times'+pc+'.csv', delimiter=',')#v_full[1]
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return x,y,vx,vy,occ
    
def occ_2908_oneperson(pc,number,iteration): #have to put number = a.b, even for O -> O.O:

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return occ
    
    
def parameters_2708anim(number,iteration): #have to put number = a.b, even for O -> O.O
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9timesMLC1.csv', delimiter=',') #x_full[1]
    
    r = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005rbigH9timesMLC1.csv', delimiter=',') 

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return x,y,r,occ  
    
def parameters_2908_005anim(pc,number,iteration): #have to put number = a.b, even for O -> O.O
        
    x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9times'+pc+'.csv', delimiter=',') #x_full[0]
    y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9times'+pc+'.csv', delimiter=',') #x_full[1]
    
    vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9times'+pc+'.csv', delimiter=',') #v_full[0]
    
    r = np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005rbigH9times'+pc+'.csv', delimiter=',') 
     
    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return x,y,vx,r,occ
    
    
def time_2908_005anim(pc,number,iteration): #have to put number = a.b, even for O -> O.O
        

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9times'+pc+'.csv',delimiter=',')
    return time
    
def time_2708anim(number,iteration): #have to put number = a.b, even for O -> O.O
        

    number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    return time  
 
def time_005anim(date,pc,number,iteration):
    
    if date == 2708:
        return time_2708anim(number,iteration)
    
    if date == 2908:
        return time_2908_005anim(pc,number,iteration)
           
def parameters_005anim(date,pc,number,iteration):
    
    if date == 2708:
        return parameters_2708anim(number,iteration)
    
    if date == 2908:
        return parameters_2908_005anim(pc,number,iteration)
        
        
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
    
def regression(x,y,value_degree,toplot):
    #transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    
    polynomial_features= PolynomialFeatures(degree=value_degree)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    print(rmse)
    print(r2)
    if toplot : 
        
        plt.scatter(x, y, s=10)
        # sort the values of x before line plot
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
        x, y_poly_pred = zip(*sorted_zip)
        plt.plot(x, y_poly_pred, color='m')
        plt.show()
    else:
        return x,y_poly_pred 
    