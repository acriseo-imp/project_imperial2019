import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy.spatial.distance as scidist
import helpful_functions as hf


# N=10 #number of elements
# 
# X=np.zeros((N,N)) #matrix such as X[i,j] is the time of contact between i-j (t_ij)
# """which values can have X ? I put an interval [0,75*1/lambda, 1?25*1/lambda], but maybe something else ? We don't have the values used in the paper"""
# 
# #values of matrix X, which is symmetric
# for i in range(N):
#     for j in range(N):
#         if i!=j:
#             X[i,j]=np.random.rand()
#             X[j,i]=X[i,j]
# 
# 
# #l= 1.5*10**(-4) #lambda
# l=0.0015

# 
# #putting the values in the interval [0,75*1/lambda, 1?25*1/lambda]
# a=0.75*1/l
# b=1.25*1/l
# X=(b-a)*X+a

# matrix1 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 10/2007_3_occupancy60time90timestep01matrixfullMLC10.csv', delimiter=',')
# 
# matrix2 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 1/results/new_occupancy/0708_1_occupancy60time90timestep01positionrandomradiusrandommatrixfullMLC1.csv', delimiter=',')
# 
# x_full = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 1/results/new_occupancy/0708_1_occupancy60time90timestep01positionrandomradiusrandomxfullMLC1.csv', delimiter=',')
# y_full = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 1/results/new_occupancy/0708_1_occupancy60time90timestep01positionrandomradiusrandomyfullMLC1.csv', delimiter=',')
# time_step=0.1
# 


def transmission_disease(matrix,l,calculate_DG):
    N=matrix.shape[0]
    number_infected=np.zeros(N)
    list_infected=[[] * N for i in range(N)]
    
    DG = nx.DiGraph()
    
    n=x_full.shape[0]
    m=x_full.shape[1]
    x=np.zeros([2,n,m])
    x[0]=x_full[:n,:m]
    x[1]=y_full[:n,:m]
    matrix_contact= np.zeros((n,n))
    for k in range(m):
        gap = scidist.squareform(scidist.pdist(x[:,:,k].T)) #distance between i and j
        matrix_contact[gap<1]+=time_step
    
    for i in range(N):
        for j in range(N):
            if i==j:
                matrix_contact[i,j]=0
            else : 
                tij=matrix_contact[i,j]
                pij=1-math.exp(-tij*l)
                u=np.random.uniform(0,1)
                if u<pij :
                    number_infected[i]+=1
                    list_infected[i].append(j)
    i=0
    if calculate_DG : 
        for list in list_infected :
            for element in list :
                DG.add_edge(i,element)
            i+=1
    return list_infected,number_infected,DG

# a,b,DG = transmission_disease(matrix2,l,True)
# nx.draw(DG,with_labels=True)
# plt.show()

def evolution_time_l(matrix,rep,start,end,step):
    list_l=np.arange(start,end,step)
    n=len(list_l)
    mean_result=[[]]*len(list_l)
    std_result=[[]]*len(list_l)
    for i in range(n) :
        result = [[] * rep for k in range(rep)]
        for j in range(rep):
            a,b,DG = transmission_disease(matrix,list_l[i],False)
            result[j].append(b)
        mean_result[i] = np.mean(result)
        std_result[i] = np.std(result)
    plt.figure()
    plt.plot(list_l,mean_result)
    plt.grid()
    plt.xlabel("values of lambda")
    plt.ylabel("Mean of the number of people infected")
    plt.xlim([start,end])
    plt.title("Mean of the number of people infected = f(lambda), "+ str(rep) + " repetitions, step of "+str(step))
    plt.figure()
    plt.plot(list_l,std_result)
    plt.grid()
    plt.xlabel("values of lambda")
    plt.ylabel("Steady value of the number of people infected")
    plt.xlim([start,end])
    plt.title("Steady value of the number of people infected = f(lambda), "+ str(rep) + " repetitions, step of "+str(step))
    plt.figure()
    plt.loglog(list_l,mean_result)
    plt.grid()
    plt.xlabel("values of lambda")
    plt.ylabel("Mean of the number of people infected, log plot")
    plt.xlim([start,end])
    plt.title("Mean of the number of people infected = f(lambda), "+ str(rep) + " repetitions, step of "+str(step))
    plt.figure()
    plt.loglog(list_l,std_result)
    plt.grid()
    plt.xlabel("values of lambda")
    plt.ylabel("Steady value of the number of people infected, log plot")
    plt.xlim([start,end])
    plt.title("Steady value of the number of people infected = f(lambda), "+ str(rep) + " repetitions, step of "+str(step))
    plt.show()

def transmission_disease_mean(list_matrix,l,rep): #mean of number infected par person
    
    index=0
    for matrix in  list_matrix:
        index+=1
        result = [[] * rep for i in range(rep)]
        for j in range(rep):
            a,b,DG = transmission_disease(matrix,l,False)
            result[j].append(b)
        mean_result = np.mean(result,axis=0)
        std_result=np.std(result,axis=0)
        fig = plt.figure(0)
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 61, 1))
        plt.plot(mean_result[0],label="matrix"+str(index))
        plt.xlabel("People")
        plt.ylabel("Number of infected people")
        plt.title("Mean of number of infected people, "+ str(rep) + " repetitions, l = "+ str(l))
        plt.xlim([0,60])
        plt.legend()
        plt.grid()
        fig = plt.figure(1)
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 61, 1))
        plt.plot(std_result[0],label="matrix"+str(index))
        plt.xlabel("People")
        plt.ylabel("Number of infected people")
        plt.title("Standard deviation of number of infected people, "+ str(rep) + " repetitions, l = "+ str(l))
        plt.xlim([0,60])
        plt.legend()
        plt.grid()
    
    plt.show()
    
#transmission_disease_mean([matrix1,matrix2],l,10000)
# 
# l=1/5000
# time_step=0.1
# x_full = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 10/2007_3_occupancy60time90timestep01xfullMLC10.csv', delimiter=',')
# y_full = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 10/2007_3_occupancy60time90timestep01yfullMLC10.csv', delimiter=',')

def transmission_disease_time(x_full,y_full,time_step,l,infected=0,draw_network=True):
    n=x_full.shape[0]
    m=x_full.shape[1]
    x=np.zeros([2,n,m])
    x[0]=x_full[:n,:m]
    x[1]=y_full[:n,:m]
    matrix_contact= np.zeros((n,n))
    list_who_infected = [] #[i] = infected number i 
    list_who_infected.append(infected) 
    list_time_infected=np.zeros(n) #[i] = time where i was infected
    list_infected=[[] * n for i in range(n)] #[i] = people infected by i
    for k in range(m):
        gap = scidist.squareform(scidist.pdist(x[:,:,k].T)) #distance between i and j
        matrix_contact[gap<1]+=time_step
        #print(list_who_infected)
        #print(matrix_contact)
        for i in list_who_infected:
            for j in range(n):
                if j not in list_who_infected : 
                    tij=matrix_contact[i,j]
                    if tij != 0 :
                        pij=1-math.exp(-tij*l)
                        u=np.random.uniform(0,1)
                        if u<pij :
                            list_who_infected.append(j)
                            list_infected[i].append(j)
                            list_time_infected[j]=k
     
     
    
    DG = nx.DiGraph()
    i=0
    for list in list_infected :
        for element in list :
            DG.add_edge(i,element)
        i+=1
     
    if draw_network : 
        carac = pd.DataFrame({ 'ID':np.arange(0,60), 'myvalue':list_time_infected/100})   
        pos = nx.spectral_layout(DG)     
        nc=nx.draw(DG,with_labels=True,node_color=carac['myvalue'],cmap=plt.cm.jet)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        sm.set_array([])
        #cbar = plt.colorbar(sm)
        #plt.colorbar(nc)
        #nx.draw_networkx_nodes(DG,pos, node_color=result_time)
        plt.show()
    
    
    return matrix_contact,list_who_infected,list_infected,list_time_infected,DG
   
def number_infected_time_mean(x_full,y_full,time_step,l,list_choice,rep):
    number = np.arange(0,60)
    
    for i in list_choice:
        result = [[] * rep for i in range(rep)]
        for j in range(rep):
            matrix_contact,result_who,result_infected,result_time,DG = transmission_disease_time(x_full,y_full,time_step,l,infected=i,draw_network = False)
            result_time.sort()
            result[j].append(result_time)
        mean_result = np.mean(result,axis=0)
        plt.plot(mean_result[0],number,label=i)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("number of infected people")
    plt.title("number of infected people = f(time), 50 repetitions")
    plt.grid()
    plt.show()
 
 
#matrix_contact,result_who,result_infected,result_time,DG = transmission_disease_time(x_full,y_full,time_step,l,infected=20,draw_network = True)
#number_infected_time_mean(x_full,y_full,time_step,l,[0,5,10,15,20,25,30,35,40,45,50,55],50)


def function_f(x):
    return -np.log2(x/10) 
def probadistance():
    
    distance = np.arange(0.18,3,0.05)

    proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/proba3.csv',delimiter=',')
    plt.plot(proba[:,0],proba[:,1],'o')
    plt.xlabel("Distance (m)")
    plt.ylabel("Risk of cross infection")
    #plt.plot(distance,function_f(distance))
    plt.show()


def matrix_distanceandcontact(x_full):
    #distance moyenne entre populations à moins de 1.5m. Compte les distances entre deux centres d'individus, pas leurs bords
    
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
    
def closest_rate(value,proba):
    
    proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/proba3.csv',delimiter=',')
    
    idx = (np.abs(proba[:,0]-value)).argmin()
    return proba[:,1][idx]
    
    
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
    
    
def pij_onesimulation(number,iteration):
    
    coeff = 0.01/0.2 #valeur arbitraire pour avoir un bon lambda
    x_full,v_full=hf.parameters_xvfull(2908,number,iteration)
    matrix_moyenne,matrix_tij=matrix_distanceandcontact(x_full)
    matrix_l = coeff*matrix_lambda(x_full,matrix_moyenne)
    people = x_full.shape[1]
    matrix_pij=1-np.exp(-matrix_l*matrix_tij)
    return matrix_pij
    
def mean_plot_pijfunctionofpedestrians_onesimulation(number,iteration):
    #plot mean probability of infection for one simulation
    date=2908
    matrix_pij=pij_onesimulation(number,iteration)
    people=matrix_pij.shape[0]
    list_proba = np.zeros(people)
    list_medproba = np.zeros(people)
    for i in range(people):
        list_proba[i]=(np.sum(matrix_pij[i])+np.sum(matrix_pij[:,i]))/(people-1)
        list_medproba[i]=np.median(np.concatenate([matrix_pij[:,i][:i],matrix_pij[i][i+1:]]))
        
    # plt.plot(list_proba,'o',markersize=3)
    # plt.xlabel("Pedestrians")
    # plt.ylabel("Mean probability of infection")
    # plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanpijoverpedestrians/'+str(number)+'/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_meanprobabilityinfectionthroughpedestrians.png')
    # plt.close()
    return list_proba, list_medproba
    
def meanstd_plot_pijfunctionofpedestrians_onesimulation(number,iteration):
    #plot mean and std probability of infection for one simulation
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

def meanstd_plotallsimulations():
    
    number_list = [90,82,75,67,60,52,45,36,30,24,18,12]
    list_iteration = np.arange(108)
    mean_all_proba=[]
    var_all_proba=[]
    std_all_proba=[]
    for i in range(108):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        print(number,iteration)
        meanstd_plot_pijfunctionofpedestrians_onesimulation(number,iteration)
    
    
def mean_plot_pijfunctionofpedestrians_onesimulation_oneperson(number,iteration):
    #plot  probability of infection for each person for one simulation
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
    
    
    
def mean_list_pijpedestrians_allsimulation():
    #plot mean probability of infection for all simulations and pedestrians-average

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
        # var_all_proba.append(np.var(list_proba))
        # std_all_proba.append(np.std(list_proba))
        #med_all_proba.append(np.median(list_proba))
    
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanallproba.csv',mean_all_proba,delimiter=',')
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/medallproba.csv',med_all_proba,delimiter=',')
    
    # np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/varallproba.csv',var_all_proba,delimiter=',')
    # 
    # np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/stdallproba.csv',std_all_proba,delimiter=',')
    
    
def mean_plot_pij():
    
    occ2908=hf.occlist2908()
    mean_all_proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/meanallproba.csv',delimiter=',')
    #std_all_proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/stdallproba.csv',delimiter=',')
    med_all_proba=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/proba/medallproba.csv',delimiter=',')
    plt.plot(occ2908,mean_all_proba,'o',markersize=3, label = "Mean values")
    plt.plot(occ2908,med_all_proba,'o',markersize=3, label = "Median values")
    plt.xlabel("Occupancy")
    plt.ylabel("Average of probability over all pedestrians")
    plt.legend()
    #plt.errorbar(occ2908,mean_all_proba,yerr=std_all_proba,fmt='o',markersize=3,color='b')
    plt.show()
    

