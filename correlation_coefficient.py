import numpy as np
import matplotlib.pyplot as plt
import math
import ped_utils as putils
import helpful_functions as hf 
from matplotlib import ticker, cm
import pandas as pd


x = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 11/new_occupancy/1008_1_occupancy72time90timestep01positionrandomradiusrandomxfullbigHMLC11.csv', delimiter=',') #x_full[0]
y = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 11/new_occupancy/1008_1_occupancy72time90timestep01positionrandomradiusrandomyfullbigHMLC11.csv', delimiter=',') #x_full[1]

vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 11/new_occupancy/1008_1_occupancy72time90timestep01positionrandomradiusrandomvxfullbigHMLC11.csv', delimiter=',') #v_full[0]
vy = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 11/new_occupancy/1008_1_occupancy72time90timestep01positionrandomradiusrandomvyfullbigHMLC11.csv', delimiter=',')#v_full[1]


    
    
def corrpkr_2708list(T,x):
    number_list =np.array([75,60,45,36,30,24,18])
    mean_corrlist=[]
    timestep = 0.05
    corr_list =np.zeros(72)
    occ_list=np.zeros(72)
    for i in range(9):
        x_full,v_full = hf.parameters_xvfull(2708,90,i)
        corr_list[i]=corrT(T,[x],x_full,v_full,timestep)
        
    for i in range(63):
        
        iteration = i
        number = number_list[i//9]
        x_full,v_full = hf.parameters_xvfull(2708,number,i)
        corr_list[i+9]=corrT(T,[x],x_full,v_full,timestep)
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_corrlist_T'+str(T)+'_x'+str(x)+'.csv',corr_list,delimiter=',')
    
def corrpkr_2908list(T,x1,x2,y1,y2,repetx,repety):
    
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    timestep = 0.05
    
    X=np.arange(x1,x2,1/repetx)
    Y=np.arange(y1,y2,1/repety)
    for xi in X:
        for yj in Y:
            corr_list =np.zeros(108)
            for i in range(108):   
                iteration = i
                number = number_list[i//9]
                x_full,v_full = hf.parameters_xvfull(2908,number,iteration)
                corr_list[i]=corrTxy(T,xi,yj,x_full,v_full,timestep)
            
            np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/data_corr_2908_'+str(T)+'/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrlist_T'+str(T)+'_x'+str(xi)+'_y'+str(yj)+'.csv',corr_list,delimiter=',')
            
            
def corrpkr_2908plot(T):
    
    x1,x2,y1,y2,repetx,repety= -2,2,-1,1,2,4
    corr_list=np.zeros([64,108])
    X=np.arange(x1,x2,1/repetx)
    Y=np.arange(y1,y2,1/repety)
    i=0
    for xi in X:
        for yj in Y:
            corr_list[i,:]=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/data_corr_2908_'+str(T)+'/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrlist_T'+str(T)+'_x'+str(xi)+'_y'+str(yj)+'.csv',delimiter=',')
            i+=1
    
    occ_list = hf.occlist2908()
    
    plt.plot(occ_list,np.mean(corr_list,axis=0),'o',markersize=3)
    plt.title("Mean of correlation coefficient in a rectangle through occupancy, T=3")
    plt.show()
            
    
def corrpkr(T,save):
    
    liste_x = [-1,1]
    n=len(liste_x)
    
    corr2908_0 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrlist_T3_x0.csv',delimiter=',')
    corr2708_0 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_corrlist_T3_x0.csv',delimiter=',')
    
    
    corr2908 = np.zeros([n+1,corr2908_0.size])
    corr2708 = np.zeros([n+1,corr2708_0.size])
    
    for i in range(n):
        x = liste_x[i]
        corr2908[i,:] = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrlist_T'+str(T)+'_x'+str(x)+'.csv',delimiter=',')
        corr2708[i,:] = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_corrlist_T'+str(T)+'_x'+str(x)+'.csv',delimiter=',')
        
    corr2908[-1,:]=corr2908_0
    corr2708[-1,:]=corr2708_0
        
    corrmean2908 = np.mean(corr2908,axis=0)
    corrmean2708 = np.mean(corr2708,axis=0)
    occ2908 = hf.occlist2908()
    
    occ2708 = hf.occlist2708()
    
    corrmeantot = np.concatenate([corrmean2908,corrmean2708])
    occtot = np.concatenate([occ2908,occ2708])
    
    if save : 
        
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrmean_T'+str(T)+'_xto-1to1'+'.csv',corrmeantot,delimiter=',')
        
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_occtot_T'+str(T)+'_xto-1to1'+'.csv',occtot,delimiter=',')
    
    plt.plot(occ2908,corrmean2908,'o',color='b',markersize=3)
    plt.plot(occ2708,corrmean2708,'o',color='b',markersize=3)
    
    plt.show()
    
def regressioncorrpkr(T,degree):
    
    corrmeantot = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_corrmean_T'+str(T)+'_xto-1to1'+'.csv',delimiter=',')
    
    occtot = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_occtot_T'+str(T)+'_xto-1to1'+'.csv',delimiter=',')
    
    hf.regression(occtot,corrmeantot,degree,True)
    
def corrthroughtimeforone(date,number,iteration,x,y,begin,end,repet,toplot,savefig):
    #gives correlation coefficient on one point through time, for only one simulation
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
        
        
def corrthroughtimeforalliterationmean90to18(toplot,savefig) :
    liste_y = [0]
    liste_x = [-1,0,1]
    begin=1
    end=10
    repet = 8
    date = 2908
    number_list =[90,82,75,67,60,52,45,36,30]
    for i in range(1,9):
        number = number_list[i]
        print(number)
        list_iteration = np.arange(9*i,9*(i+1),1)
        corrthroughtimeforalliterationmean(date,number,list_iteration,liste_x,liste_y,begin,end,repet,toplot,savefig)
        
        
        
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
            #corr_mean_all[x]=np.mean(corr_mean_list,axis=0)
        for j in range(9):
            plt.plot(T,np.mean(corr_mean_list[:,j,:],axis=0),'o',markersize=2)
            title = "Correlation coefficient through time, for simulation with "+str(number)+" people, simulation number " +str(j)
            plt.title(title)
            # plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/correlation/pkr/through_time/'+str(number)+'/'+title+'y0threex.png')
            plt.show()
   
def repetsize(value,size):
    x=np.array([value])
    sol=np.array([value])
    for i in range(1,size):
        sol=np.concatenate([sol,x])
    return sol
    
     
def corrploty0xbougediagseaborn():
    
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
    x=repetsize(liste_x[0],T.size)
    z=z0
    v=corr_mean_all[0]
    for i in range(1,81):
        x=np.concatenate((x,repetsize(liste_x[i],T.size)))
        v=np.concatenate((v,corr_mean_all[i]))
        z=np.concatenate((z,z0))

    df = pd.DataFrame({'x':x, 'y':z, 'z':v})
    

    points=plt.scatter(x,z,s=200,c=v,marker='s',cmap="jet")
    plt.colorbar(points,label="Correlation ")
    plt.xlabel('Occupancy')
    plt.ylabel('Time lag T (s)')
   # plt.title(str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", timestep of 0.05")
    plt.xlim([min(liste_x),max(liste_x)])
    plt.ylim([T[0],T[-1]])
    plt.show()




#     plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/speedandsimulationsplot/localspeed/'+str(number)+'/'+str(date)+'_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between0and1_size1timex_005_spectral_remove10sdensitymoinsque4.png')
#     plt.close()
#     
    
    
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
        plt.plot(number_list[j//9],-2/T[np.where(corr_mean_all[j]==np.amax(corr_mean_all[j]))],'o',markersize=3,color='b')
    plt.xlabel("Occupancy")
    plt.ylabel("Phase speed")
    plt.show()
    
            
             
def corrthroughtimeforalliterationmean(date,number,list_iteration,liste_x,liste_y,begin,end,repet,toplot,savefig):
    #gives mean of corr coeff on one point through  time, for all iteration with same number
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
        #corr_overall_mean[x*Y+y,:]=np.mean(corr_meanlist,axis=0)
        
            
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
                
    
    
        
def corrthroughrectangleforonesimulation(number,iteration,x1,x2,y1,y2,repetx,repety,T,toplot): #repetx = 4, repety=8 x1x2 -2 2 y1y2 -1 1
    
    corr_list=[]
    timestep=0.05
    x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
    people = vx.shape[0]
    Time = vx.shape[1]
    x_full = np.zeros([2,people,Time])
    x_full[0]=x
    x_full[1]=y
    v_full = np.zeros([2,people,Time])
    v_full[0]=vx
    v_full[1]=vy
    X=np.arange(x1,x2,1/repetx)
    Y=np.arange(y1,y2,1/repety)
    for xi in X:
        for yj in Y:
            corr_list.append(corrTxy(T,xi,yj,x_full,v_full,timestep))
            
    if toplot :
        plt.plot(corr_list,'o',markersize=3)
        N=X.size
        M=Y.size
        plt.xticks(range(0,M*N,M),[X[i] for i in range(N)])
        plt.axhline(np.mean(corr_list),color='r',label = "Mean of correlation coefficient")
        plt.legend()
        plt.ylabel("Correlation coefficient")
        plt.xlabel("Values of x, y changes through each cell")
        plt.grid()
        plt.show()
    else :
        return corr_list
        
def corrthroughrectanglemean(number,list_iteration,x1,x2,y1,y2,repetx,repety,T,toplot):
    
    X=np.arange(x1,x2,1/repetx)
    Y=np.arange(y1,y2,1/repety)
    N=X.size
    M=Y.size
    corr_meanlist=np.zeros([9,N*M])
    timestep=0.05
    for i in range(9):
        corr_meanlist[i,:] = corrthroughrectangleforonesimulation(number,list_iteration[i],x1,x2,y1,y2,repetx,repety,T,False)
        
    if toplot :
        plt.plot(np.mean(corr_meanlist,axis=0),'o',markersize=3)
        plt.xticks(range(0,M*N,M),[X[i] for i in range(N)])
        plt.axhline(np.mean(corr_meanlist),color='r',label = "Mean of correlation coefficient")
        plt.legend()
        plt.ylabel("Mean of correlation coefficient")
        plt.xlabel("Values of x, y changes through each cell")
        title = "Mean correlation coefficient through a rectangle, for simulation with "+str(number)+" people"
        plt.title(title)
        plt.grid()
        plt.show()
        plt.title()
    
    return corr_meanlist
    

    
# figure = plt.figure()
# axes = figure.add_subplot(111)
# X=np.arange(-2,2+1/2,1/2)
# Y=np.arange(-1,1+1/2,1/2)
# N=X.size
# M=Y.size
# plt.xticks(range(0,M*N+1,M),[X[i] for i in range(N)])
# plt.plot(aa,'*')
# plt.grid()
# plt.show()
   
def corrTxy(T,x,y,x_full,v_full,timestep): #in our x and y, removing first 20s

    Time = x_full.shape[2]

    data = np.zeros([2,Time-int(T/timestep)])
    for t in range(Time-int(T/timestep)):
        data[0][t]=putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[x,y])
        data[1][t]=putils.local_speed_current(x_full[:,:,t+int(T/timestep)],v_full[:,:,t+int(T/timestep)],[x-2,y])
    data_corr=np.corrcoef(data[:,int(20//timestep):])[0,1]
    return data_corr 
    
    

def corrT(T,liste,x_full,v_full,timestep): #in one x, y = 0, removing first 20s
    data_corr=[0]
    Time = x_full.shape[2]
    for i in liste:
        data = np.zeros([2,Time-int(T/timestep)])
        for t in range(Time-int(T/timestep)):
            data[0][t]=putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[i,0])
            data[1][t]=putils.local_speed_current(x_full[:,:,t+int(T/timestep)],v_full[:,:,t+int(T/timestep)],[i-2,0])
        data_corr[0]=np.corrcoef(data[:,int(20//timestep):])[0,1]
    return data_corr[0]
    
def printV(T,liste,date,number,iteration,timestep):
    
    x_full,v_full = hf.parameters_xvfull(date,number,iteration)
    
    Time = x_full.shape[2]
    for i in liste:
        data = np.zeros([2,Time-int(T/timestep)])
        for t in range(Time-int(T/timestep)):
            data[0][t]=putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[i,0])
            data[1][t]=putils.local_speed_current(x_full[:,:,t+int(T/timestep)],v_full[:,:,t+int(T/timestep)],[i-2,0])
    plt.plot(data[0][int(10//timestep):],label = "V(x,t)")
    plt.plot(data[1][int(10//timestep):],label = "V(x-X,t+T)")
    plt.legend()
    plt.title("T= "+str(T))
    plt.show()
    
# print(np.mean(corrT(3,liste)))
# print(np.mean(corrT(5,liste)))
# print(np.mean(corrT(7,liste)))



def plotscorr():
    
    ###Radius mass random 0.1, big H
    plt.figure()
    
    radiusrandom01_3 = [0.2550433,0.162947257,0.681428107,0.470729666,-0.024794767,-0.15273174,3,-0.091391117]
    radiusrandom01_5 = [0.0994030,-0.21134407,-0.565666132,-0.561104454,-0.040633629,-0.034475187,0.059398888]
    radiusrandom01_7= [0.0962396,0.088540952,-0.651118432,-0.167498911,0.114402493,0.196947959,0.017170848]
    radiusrandom01_occ=[0.781,0.693,0.593,0.492,0.377,0.282,0.207]
    
    plt.plot(radiusrandom01_occ,radiusrandom01_3,'*--',label="T=3s")
    plt.plot(radiusrandom01_occ,radiusrandom01_5,'*--',label="T=5s")
    plt.plot(radiusrandom01_occ,radiusrandom01_7,'*--',label="T=7s")
    plt.title("Radius mass random, timestep of 0.1, big H")
    plt.legend()
    
    ###Radius mass random 0.05, new dest, small H
    plt.figure()
    
    new_dest_3=[0.453671206,0.18388491,-0.025454146,-0.009110735,0.022410269,0.040391767]
    new_dest_5 = [0.097877212,-0.364646061,-0.088467883,-0.014202161,0.041986849,0.02480846]
    new_dest_7=[0.007046313,-0.12091088,0.059836426,0.074035982,0.146830982,-0.023666204]
    new_dest_occ = [0.771,0.705,0.616,0.524,0.417,0.292]
    
    plt.plot(new_dest_occ,new_dest_3,'*--',label="T=3s")
    plt.plot(new_dest_occ,new_dest_5,'*--',label="T=5s")
    plt.plot(new_dest_occ,new_dest_7,'*--',label="T=7s")
    plt.title("Radius mass random 0.05, new dest, small H")
    plt.legend()
    
    ###Radius random 0.025, big H
    plt.figure()		 		

    radiusrandom0025_3 = [0.637262125,0.640611157,0.666572555,0.067754158,0.010736152,-0.124892753,-0.153313731]
    radiusrandom0025_5 = [0.191895431,-0.168901077,-0.619128402,-0.047551544,0.018760941,-0.109112533,0.340222984]
    radiusrandom0025_7 = [-0.028248006,-0.38660037,-0.638773354,0.126551464,0.082855054,0.090632702,-0.030716971]
    radiusrandom0025_occ = [0.837,0.691,0.611,0.506,0.409,0.336,0.203]
    
    plt.plot(radiusrandom0025_occ,radiusrandom0025_3, '*--', label = "T=3s")
    plt.plot(radiusrandom0025_occ,radiusrandom0025_5, '*--', label = "T=5s")
    plt.plot(radiusrandom0025_occ,radiusrandom0025_7, '*--', label = "T=7s")
    plt.title("Radius mass random 0.025, big H")
    plt.legend()
    plt.show()



#occ_list,corr_list = corrpkr()
#hf.regression(occ_list,corr_list)
    