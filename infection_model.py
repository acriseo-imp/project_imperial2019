"""

CRISEO Alexandre
CID 01604586
Imperial College, 2018-2019, MSC Applied Mathematics


Referring to section 4.12

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy.spatial.distance as scidist
import helpful_functions as hf


### returns average distance between pedestrians for one simulation
def matrix_distanceandcontact(x_full):
  
    people = x_full.shape[1]
    Time=x_full.shape[2]
    time_step=0.05
    matrix_distance=np.zeros([people,people])
    matrix_compteur=np.zeros([people,people])
    matrix_moyenne=np.zeros([people,people])
    for t in range(Time):
        x=x_full[:,:,t]
        gap = scidist.squareform(scidist.pdist(x.T)) #distance between i and j
        matrix_distance[gap<1.5]+=gap[gap<1.5]
        matrix_compteur[gap<1.5]+=1
       
       
    matrix_moyenne[matrix_compteur!=0]=matrix_distance[matrix_compteur!=0]/matrix_compteur[matrix_compteur!=0]
    return matrix_moyenne,matrix_compteur*time_step
 
### returns lambda for a d0 value  
def closest_rate(value,proba):
    
    proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/proba3.csv',delimiter=',')
    
    idx = (np.abs(proba[:,0]-value)).argmin()
    return proba[:,1][idx]
    
   
### returns matrix of lambda values for all pedestrians, one simulation
def matrix_lambda(x_full,matrix_moyenne):
    
    people = x_full.shape[1]
    matrix_l=np.zeros([people,people])
    proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/proba3.csv',delimiter=',')
    
    for i in range(people):
        for j in range(i+1,people):
            value = matrix_moyenne[i,j]
            if value!=0:
                matrix_l[i,j]=closest_rate(value,proba)

    return matrix_l
    
    
### computes pij for all pedestrians in one simulation
def pij_onesimulation(number,iteration):
    
    coeff = 0.01/0.2 
    x_full,v_full=hf.parameters_xvfull(2908,number,iteration)
    matrix_moyenne,matrix_tij=matrix_distanceandcontact(x_full)
    matrix_l = coeff*matrix_lambda(x_full,matrix_moyenne)
    people = x_full.shape[1]
    matrix_pij=1-np.exp(-matrix_l*matrix_tij)
    return matrix_pij
  
### returns average and median probability for one simulation
def mean_plot_pijfunctionofpedestrians_onesimulation(number,iteration):
    date=2908
    matrix_pij=pij_onesimulation(number,iteration)
    people=matrix_pij.shape[0]
    list_proba = np.zeros(people)
    list_medproba = np.zeros(people)
    for i in range(people):
        list_proba[i]=(np.sum(matrix_pij[i])+np.sum(matrix_pij[:,i]))/(people-1)
        list_medproba[i]=np.median(np.concatenate([matrix_pij[:,i][:i],matrix_pij[i][i+1:]]))
        

    return list_proba, list_medproba
    
### plots mean and median probability of infection for one simulation, with errorbar figures 4.30 and 4.31    
def meanstd_plot_pijfunctionofpedestrians_onesimulation(number,iteration):
    
    date=2908
    matrix_pij=pij_onesimulation(number,iteration)
    people=matrix_pij.shape[0]
    list_proba = np.zeros(people)
    list_stdproba = np.zeros(people)
    list_medproba = np.zeros(people)
    list_people=np.arange(people)
    for i in range(people):
        list_proba[i]=(np.sum(matrix_pij[i])+np.sum(matrix_pij[:,i]))/(people-1)
        list_stdproba[i]=np.std(np.concatenate([matrix_pij[:,i][:i],matrix_pij[i][i+1:]]))
        list_medproba[i]=np.median(np.concatenate([matrix_pij[:,i][:i],matrix_pij[i][i+1:]]))
    
    
    plt.errorbar(list_people,list_proba,yerr=list_stdproba,fmt='o',markersize=3,color='k', label = "Mean values")
    plt.plot(list_people,list_medproba,'o',markersize=3,color='r', label = "Median values")
    plt.xlabel("Pedestrians")
    plt.ylabel("Median and mean probability of infection")
    plt.legend()
    plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanpijoverpedestrians/'+str(number)+'/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_meanmedianstdprobabilityinfectionthroughpedestrianstest.png')
    plt.close()

    
### plots probability of infection for each person for one simulation, 29/08 data    
def mean_plot_pijfunctionofpedestrians_onesimulation_oneperson(number,iteration):
    
    date=2908
    matrix_pij=pij_onesimulation(number,iteration)
    people=matrix_pij.shape[0]
    list_people=np.arange(people)
    list_proba = np.zeros(people)
    for i in range(people):
        plt.plot(list_people[i+1:],matrix_pij[i][i+1:],'o',color='b',markersize=3)
        plt.plot(list_people[:i],matrix_pij[:,i][:i],'o',color='b',markersize=3)
   
        plt.xlabel("Pedestrians")
        plt.ylabel("Probability of infection")
        plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanpijoverpedestrians/'+str(number)+'/'+str(iteration)+'/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'_'+str(i)+'time90timestep005xfullbigH9timesMLC1_probabilityinfectionthroughpedestrians.png')
        plt.close()
    
    
### Computes data for figure 4.32    
def mean_list_pijpedestrians_allsimulation():
    
    number_list = [90,82,75,67,60,52,45,36,30,24,18,12]
    list_iteration = np.arange(108)
    mean_all_proba=[]
    var_all_proba=[]
    std_all_proba=[]
    med_all_proba=[]
    for i in range(108):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        print(number,iteration)
        list_proba,list_medproba=mean_plot_pijfunctionofpedestrians_onesimulation(number,iteration)
        mean_all_proba.append(np.mean(list_proba))
        med_all_proba.append(np.mean(list_medproba))
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanallproba.csv',mean_all_proba,delimiter=',')
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/medallproba.csv',med_all_proba,delimiter=',')
    

    
### Figure 4.32
def mean_plot_pij():
    
    occ2908=hf.occlist2908()
    mean_all_proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanallproba.csv',delimiter=',')
    med_all_proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/medallproba.csv',delimiter=',')
    plt.plot(occ2908,mean_all_proba,'o',markersize=3, label = "Mean values")
    plt.plot(occ2908,med_all_proba,'o',markersize=3, label = "Median values")
    plt.xlabel("Occupancy")
    plt.ylabel("Average of probability over all pedestrians")
    plt.legend()
    plt.show()
    

