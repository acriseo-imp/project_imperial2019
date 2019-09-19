"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Code in relation with correlation coefficient. How correlation coefficient is computed + different plots.

"""
import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils
import helpful_functions as hf 
from matplotlib import ticker, cm
import pandas as pd


###Computation of correlation coefficient for one x and y, removing first 20s of data
def corrTxy(T,x,y,x_full,v_full,timestep): #in our x and y, removing first 20s

    Time = x_full.shape[2]
    data = np.zeros([2,Time-int(T/timestep)])
    for t in range(Time-int(T/timestep)):
        data[0][t]=putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[x,y])
        data[1][t]=putils.local_speed_current(x_full[:,:,t+int(T/timestep)],v_full[:,:,t+int(T/timestep)],[x-2,y])
    data_corr=np.corrcoef(data[:,int(20//timestep):])[0,1]
    return data_corr 

###gives correlation coefficient on one point through time, for only one simulation   
def corrthroughtimeforone(date,number,iteration,x,y,begin,end,repet,toplot,savefig):
    
    corr_list=[]
    timestep=0.05
    x_full,v_full = hf.parameters_xvfull(date,number,iteration)
    T=np.arange(begin,end,1/repet)
    for t in T :
        corr_list.append(corrTxy(t,x,y,x_full,v_full,timestep))
    if toplot :
        plt.plot(T,corr_list,'*')
        title = "Correlation coefficient through time, for simulation with "+str(number)+" people, number "+str(iteration)+", occupancy of "+str(occ)[:4]+", from "+str(begin)+" to "+str(end)
        plt.title(title)
        if savefig :
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/'+title+'.png')
        plt.show()
    else:
        return corr_list

###gives mean of correlation coefficient on one point through  time, for all iteration with same number    

def corrthroughtimeforalliterationmean(date,number,list_iteration,liste_x,liste_y,begin,end,repet,toplot,savefig):
    
    X=len(liste_x)
    Y=len(liste_y)
    T=np.arange(begin,end,1/repet)
    corr_meanlist=np.zeros([9,T.size])
    corr_overall_mean=np.zeros([X*Y,T.size])
    timestep=0.05
    for x in range(X):
        for y in range(Y):
            corr_meanlist=np.zeros([9,T.size])
            for i in range(9):
                corr_meanlist[i,:] = corrthroughtimeforone(date,number,list_iteration[i],liste_x[x],liste_y[y],begin,end,repet,False,False)
            np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/xy/moreT/corr_list'+str(number)+" people_"+str(begin)+"to"+str(end)+"_"+ str(repet) + "repet_xequals"+str(liste_x[x])+"_yequals"+str(liste_y[y])+'secondtry.csv',corr_meanlist,delimiter=',')
            
    if toplot :
        plt.figure()
        plt.plot(T,np.mean(corr_overall_mean,axis=1),'*')
        title = "Mean correlation coefficient through time, for simulation with "+str(number)+" people, from "+str(begin)+" to "+str(end)+", "+ str(repet) + "repetitions, in a rectangle"
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Correlation coefficient")
        print(title)
        if savefig :
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/xy/overall/'+title+'secondtry.png')
        plt.show()


  
###Plot of the phase speed = X/T through occupancy
def phasespeedcorrelation():
    
    timestep=0.05
    begin=1
    end=10
    repet=8
    number_list =[90,82,75,67,60,52,45,36,30]
    liste_x = [-1,0,1]
    T=np.arange(begin,end,1/repet)
    corr_mean_all=np.zeros([81,T.size])
    for i in range(9):
        
        number = number_list[i]
        corr_mean_list=np.zeros([3,9,T.size])
        for x in range(len(liste_x)):
            corr_mean_list[x,:] = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/xy/moreT/corr_list'+str(number)+" people_"+str(begin)+"to"+str(end)+"_"+ str(repet) + "repet_xequals"+str(liste_x[x])+"_yequals"+str(0)+'secondtry.csv',delimiter=',')
        for j in range(9):
            corr_mean_all[i*9+j,:]=np.mean(corr_mean_list[:,j,:],axis=0)
            
    liste_x=hf.occlist2908()[:81]
    
    for j in range(81):
        plt.plot(liste_x[j],-2/T[np.where(corr_mean_all[j]==np.amax(corr_mean_all[j]))],'o',markersize=3,color='b')
    plt.xlabel("Occupancy")
    plt.ylabel("Phase speed")
    plt.show()


###Values of=correlation coeﬃcient as a function of time lag T and occupancy, data of the 29/08, y=0, x = {-1,0,1}

def corrploty0differentx():
    
    timestep=0.05
    begin=1
    end=10
    repet=8
    number_list =[90,82,75,67,60,52,45,36,30]
    liste_x = [-1,0,1]
    T=np.arange(begin,end,1/repet)
    corr_mean_all=np.zeros([81,T.size])
    for i in range(9):
        number = number_list[i]
        corr_mean_list=np.zeros([3,9,T.size])
        for x in range(len(liste_x)):
            corr_mean_list[x,:] = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/xy/moreT/corr_list'+str(number)+" people_"+str(begin)+"to"+str(end)+"_"+ str(repet) + "repet_xequals"+str(liste_x[x])+"_yequals"+str(0)+'secondtry.csv',delimiter=',')
        for j in range(9):
            corr_mean_all[i*9+j,:]=np.mean(corr_mean_list[:,j,:],axis=0)
    

    liste_x=hf.occlist2908()[:81]
    z0=T
    x=np.ones(T.size)*liste_x[0]
    z=z0
    v=corr_mean_all[0]
    for i in range(1,81):
        x=np.concatenate((x,np.ones(T.size)*liste_x[i]))
        v=np.concatenate((v,corr_mean_all[i]))
        z=np.concatenate((z,z0))

    df = pd.DataFrame({'x':x, 'y':z, 'z':v})
    

    points=plt.scatter(x,z,s=200,c=v,marker='s',cmap="jet")
    plt.colorbar(points,label="Correlation ")
    plt.xlabel('Occupancy')
    plt.ylabel('Time lag T (s)')
    plt.title(str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", timestep of 0.05")
    plt.xlim([min(liste_x),max(liste_x)])
    plt.ylim([T[0],T[-1]])
    plt.show()

###Plots of correlation coefficient through time, for different number of pedestrians
def corrploty0xbougeplot():
    
    begin=1
    end=10
    repet=4
    number_list =[90,82,75,67,60,52,45,36,30]
    liste_x = [-1,0,1]
    T=np.arange(begin,end,1/repet)
    corr_meanlist=np.zeros([9,T.size])
    
    for i in range(9):
        corr_mean_all=np.zeros([len(liste_x),T.size])
        number = number_list[i]
        corr_mean_list=np.zeros([3,9,T.size])
        for x in range(len(liste_x)):
            corr_mean_list[x,:] = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/xy/corr_list'+str(number)+" people_"+str(begin)+"to"+str(end)+"_"+ str(repet) + "repet_xequals"+str(liste_x[x])+"_yequals"+str(0)+'secondtry.csv',delimiter=',')
            
        for j in range(9):
            plt.plot(T,np.mean(corr_mean_list[:,j,:],axis=0),'o',markersize=2)
            title = "Correlation coefficient through time, for simulation with "+str(number)+" people, simulation number " +str(j)
            plt.title(title)
            # plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/'+str(number)+'/'+title+'y0threex.png')
            plt.show()