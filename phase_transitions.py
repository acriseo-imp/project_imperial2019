import numpy as np
import matplotlib.pyplot as plt
import ped_utils as putils
import helpful_functions as hf

#vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 3/results/new_occupancy/1008_1_occupancy72time90timestep0025positionrandomradiusrandomvxfullbigHMLC3.csv', delimiter=',') 


vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 2/results/new_occupancy/1008_1_occupancy96time90timestep0025positionrandomradiusrandomvxfullbigHMLC2.csv', delimiter=',')


#vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 2/results/new_occupancy/0708_1_occupancy48time90timestep01positionrandomradiusrandomvxfullMLC2.csv', delimiter=',')

#vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 8/new_occupancy/1008_1_occupancy48time90timestep0025positionnotrandomradiusnotrandomvxfullbigHMLC8.csv', delimiter=',')

#vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 4/results/new_occupancy/1008_1_occupancy48time90timestep0025positionrandomradiusrandomvxfullbigHMLC4.csv', delimiter=',')

#vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 9/new_occupancy/1008_3_occupancy72time90timestep005positionrandomradiusrandomvxfullsmallHnewdestMLC9.csv', delimiter=',')

vx = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 2/results/new_occupancy/1008_2_occupancy84time90timestep0025positionrandomradiusrandomvxfullbigHMLC2.csv', delimiter=',')


    
def calculationphase75to18(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[75,60,45,36,30,24,18]
    
    mean_list=np.zeros(63)
    var_list=np.zeros(63)
    std_list=np.zeros(63)
    
    for i in range(0,63):

        iteration = i
        number = number_list[i//9]
        
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]

        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(v_full))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = np.sum(np.abs(vx_t))
                phi = abs_sum_vxt/sum_abs
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_mean75_18'+text_allv+'.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_var75_18'+text_allv+'.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_std75_18'+text_allv+'.csv',std_list,delimiter=',')
        
def calculationphase75to18removingvx(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[75,60,45,36,30,24,18]
    
    mean_list=np.zeros(63)
    var_list=np.zeros(63)
    std_list=np.zeros(63)
    
    for i in range(0,63):

        iteration = i
        number = number_list[i//9]
        
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]

        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        avg_vx = np.average(vx)
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t] - avg_vx
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(v_full))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t] - avg_vx
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = putils.average_speed(np.abs(vx_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean'+text_allv+'removingvx.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_mean75_18'+text_allv+'removingvx.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'removingvx.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_var75_18'+text_allv+'removingvx.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(7):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'removingvx.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_std75_18'+text_allv+'removingvx.csv',std_list,delimiter=',')
            
             
    #return mean_list,var_list,std_list
    
    
def calculationphase90(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    
    mean_list=np.zeros(9)
    var_list=np.zeros(9)
    std_list=np.zeros(9)
    
    
    number = 90
    for i in range(9):
        iteration = i
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        people = vx.shape[0]
        Time = vx.shape[1]

        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        
        timestep = 0.05
        
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(v_full_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = putils.average_speed(np.abs(vx_t))
                phi = abs_sum_vxt/sum_abs
                all_phi.append(phi)
        
        if save_phi: 
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'.png')
            plt.close()
        
        
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(9):
            plt.figure()
            plt.plot(mean_list,'*')
            plt.axhline(np.mean(mean_list),color='r',label = "Mean of all")
            plt.title("Mean of order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean_new'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean90_new'+text_allv+'.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(9):
            plt.figure()
            plt.plot(var_list,'*')
            plt.axhline(np.mean(var_list),color='r',label = "Mean of all")
            plt.title("Var of order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var90'+text_allv+'.csv',var_list,delimiter=',')
    
    if calcul_std:
        for i in range(9):
            plt.figure()
            plt.plot(std_list,'*')
            plt.axhline(np.mean(std_list),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std90'+text_allv+'.csv',std_list,delimiter=',')
            
        
    #return mean_list,var_list,std_list
    
    
#calculationphase75to18(0.0,1.2,True,True,True,True,False)
#calculationphase90(0,0,False,True,True,True,False)

def calculationphase90removingvx(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    
    mean_list=np.zeros(9)
    var_list=np.zeros(9)
    std_list=np.zeros(9)
    
    
    number = 90
    for i in range(9):
        iteration = i
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        people = vx.shape[0]
        Time = vx.shape[1]

        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        avg_vx=np.average(vx)
        timestep = 0.05
        
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(v_full_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = np.mean(np.abs(vx[:,t]))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
        
        if save_phi: 
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'.png')
            plt.close()
        
        
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(9):
            plt.figure()
            plt.plot(mean_list,'*')
            plt.axhline(np.mean(mean_list),color='r',label = "Mean of all")
            plt.title("Mean of order Parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean_new'+text_allv+'removingvxnotforvo.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean90_new'+text_allv+'removingvxnotforvo.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(9):
            plt.figure()
            plt.plot(var_list,'*')
            plt.axhline(np.mean(var_list),color='r',label = "Mean of all")
            plt.title("Var of order Parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'removingvxnotforvo.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var90'+text_allv+'removingvxnotforvo.csv',var_list,delimiter=',')
    
    if calcul_std:
        for i in range(9):
            plt.figure()
            plt.plot(std_list,'*')
            plt.axhline(np.mean(std_list),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing avg vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'removingvxnotforvo.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2708_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std90'+text_allv+'removingvxnotforvo.csv',std_list,delimiter=',')
        
        

def plotmean2908_list(allv):
    
    if allv : 
        text_allv = 'allv'
    else :
        text_allv = 'onlyvx'
        
    mean_list = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_new'+text_allv+'.csv',delimiter=',')
    

    
    return mean_list

def plotmean75to18mean_list(allv):
    
    if allv : 
        text_allv = 'allv'
    else :
        text_allv = 'onlyvx'
        
    mean_list75 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_mean75_18'+text_allv+'.csv',delimiter=',')
    
    mean_list90 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/90/2708_occupancy90time90timestep005xfullbigH9timesMLC1_mean90'+text_allv+'.csv',delimiter=',')
    
    mean_list = np.concatenate([mean_list90,mean_list75])
    
    return mean_list
    
    

    
    
    

def plotmean75to18(allv,both): #allv : decide if we look of vx or allv, both : in the same graph
    
    number_list =[90,75,60,45,36,30,24,18]
    plt.figure()
    mean_meanlist = np.zeros([2,8])
    occ_list = hf.occlist2708()
    occ_list_2908 = hf.occlist2908()
    
    
    if both :
        
        text_allv = "both"
        
        mean_list_true = plotmean75to18mean_list(True)
        
        mean_list_false = plotmean75to18mean_list(False)
        
        mean_list2908_true = plotmean2908_list(True)
        mean_list2908_false = plotmean2908_list(False)
        
        
       # plt.plot(occ_list,mean_list_true,'o',color='g',markersize=3,label = "allv")
        # x,y=hf.regression(occ_list,mean_list_true,4,False)
        # plt.plot(x,y,color='m')
        #plt.plot(occ_list,mean_list_false,'x',color='orange',markersize=3, label = "only vx")
        # x,y=hf.regression(occ_list,mean_list_false,4,False)
        # plt.plot(x,y,color='m')
        # g
        
        #plt.plot(occ_list_2908,mean_list2908_true,'o',color='g',markersize=5, label = "All velocity")
        # x,y=hf.regression(occ_list,mean_list_true,4,False)
        # plt.plot(x,y,color='m')
        
        plt.plot(occ_list_2908,mean_list2908_false,'x',color='b',markersize=5, label = "Only x-component of velocity") 
        plt.tick_params(axis = 'both', labelsize = 15)       
        plt.xlabel("Occupancy",fontsize=15)
        plt.ylabel("Order parameter",fontsize=14,)
        # for j in range(8):
        #     mean_meanlist[0,j]=np.mean(mean_list_true[9*j:9*(1+j)])
        #     mean_meanlist[1,j]=np.mean(mean_list_false[9*j:9*(1+j)])
        #     
        # plt.plot(number_list,mean_meanlist[0],'*--',color='k',label = "Mean of all, allv")
        # plt.plot(number_list,mean_meanlist[1],'*--',color='r',label = "Mean of all, onlyvx")
        plt.legend()
        plt.grid()
        plt.xticks(np.arange(0,0.9,0.1))
        plt.yticks(np.arange(0.75,1.05,0.05))
        #plt.title("Mean of order parameters from 12 to 90 pedestrians")
        plt.show()
        
        
    
    else : 
    
        if allv : 
            text_allv = 'allv'
        else :
            text_allv = 'onlyvx'
        
        mean_list = plotmean75to18mean_list(allv)
        
        number_list =[90,75,60,45,36,30,24,18]
        for i in range(0,72):
            number = number_list[i//9]
            plt.plot(number,mean_list[i],'o',markersize=3)
        
        mean_meanlist = np.zeros(8)
        
        for j in range(8):
            mean_meanlist[j]=np.mean(mean_list[9*j:9*(1+j)])
            
        plt.plot(number_list,mean_meanlist,'*--',color='k',label = "Mean of all")
            
        plt.legend()
        plt.title("Mean of order parameters from 12 to 90 pedestrians")
        
        plt.show()
        

def calculationphase90to12_2908(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    mean_list=np.zeros(108)
    var_list=np.zeros(108)
    std_list=np.zeros(108)
    
    
    for i in range(108):

        iteration = i
        number = number_list[i//9]
        
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]


        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(vfull_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = np.average(vx) #np.mean(np.abs(vx_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean_new'+text_allv+'voconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_new'+text_allv+'voconstant.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'voconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_var90_112'+text_allv+'voconstant.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'voconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_std90_112'+text_allv+'voconstant.csv',std_list,delimiter=',')
        
def calculationphase90to12_2908_removingvx(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    mean_list=np.zeros(108)
    var_list=np.zeros(108)
    std_list=np.zeros(108)
    
    
    for i in range(108):

        iteration = i
        number = number_list[i//9]
        
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        avg_vx = np.average(vx)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]


        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(vfull_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = avg_vx #np.mean(np.abs(vx[:,t]))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'removingvxvoconstant.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean_new'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_new'+text_allv+'removingvxvoconstant.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_var90_112'+text_allv+'removingvxvoconstant.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(12):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/2908_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_std90_112'+text_allv+'removingvxvoconstant.csv',std_list,delimiter=',')
 
 
def calculationphase96_1009_removingvx(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[96]
    
    mean_list=np.zeros(6)
    var_list=np.zeros(6)
    std_list=np.zeros(6)
    
    
    for i in range(6):

        iteration = i
        number = number_list[i//9]
        
        x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH6timesMLC1.csv', delimiter=',') #x_full[0]
        y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH6timesMLC1.csv', delimiter=',') #x_full[1]
        
        vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH6timesMLC1.csv', delimiter=',') #v_full[0]
        vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH6timesMLC1.csv', delimiter=',')#v_full[1]
        number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH6timesMLC1.csv',delimiter=',')
    

        avg_vx = np.average(vx)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]


        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(vfull_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = np.mean(np.abs(vx_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH6timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'removingvx.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH6timesMLC1_mean_new'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH6timesMLC1_mean90_112_new'+text_allv+'removingvx.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH6timesMLC1_var'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH6timesMLC1_var90_112'+text_allv+'removingvx.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH6timesMLC1_std'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH6timesMLC1_std90_112'+text_allv+'removingvx.csv',std_list,delimiter=',')     
        
 
def calculationphase18_6_1009_removingvx(begin,end,save_phi,calcul_mean,calcul_var,calcul_std,allv):
    number_list =[18,12,6]
    
    mean_list=np.zeros(27)
    var_list=np.zeros(27)
    std_list=np.zeros(27)
    
    
    for i in range(27):

        iteration = i
        number = number_list[i//9]
        
        x = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1.csv', delimiter=',') #x_full[0]
        y = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005yfullbigH9timesMLC1.csv', delimiter=',') #x_full[1]
        
        vx = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vxfullbigH9timesMLC1.csv', delimiter=',') #v_full[0]
        vy = np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005vyfullbigH9timesMLC1.csv', delimiter=',')#v_full[1]
        number,occ,avg,time,iter,avgvx=np.genfromtxt('D:/Project/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005resultsbigH9timesMLC1.csv',delimiter=',')
    

        avg_vx = np.average(vx)
        
        timestep = 0.05
        people = vx.shape[0]
        Time = vx.shape[1]


        v_full = np.zeros([2,people,Time])
        v_full[0]=vx
        v_full[1]=vy
        t_list = np.arange(0,(Time+1)*timestep,timestep)
        t_test = np.arange(0,Time)
        all_phi = []
        
        if allv:
            text_allv = 'allv' 
        
            for t in range (Time):
                vfull_t = v_full[:,:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vfull_t))
                sum_abs = putils.average_speed(np.abs(vfull_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
                
        else :
            text_allv = 'onlyvx'
            for t in range (Time):
                vx_t = vx[:,t]-avg_vx
                abs_sum_vxt = np.abs(np.sum(vx_t))
                sum_abs = np.mean(np.abs(vx_t))
                phi = abs_sum_vxt/(people*sum_abs)
                all_phi.append(phi)
            
        
        
        if save_phi:
            plt.figure(i)
            plt.plot(t_test[int(10/timestep):]*timestep,all_phi[int(10/timestep):])
            plt.xlim([int(10/timestep)*timestep,Time*timestep])
            plt.ylim([begin,end])
            plt.title("Order Parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_'+str(iteration)+'_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_between'+str(int(begin*10))+'and'+str(int(end*10))+text_allv+'removingvx.png')
            plt.close()
            
        if calcul_mean: 
            mean_list[i]=np.mean(all_phi[int(10/timestep):])
            
        if calcul_var: 
            var_list[i]=np.var(all_phi[int(10/timestep):])
            
        if calcul_std:
            std_list[i]=np.std(all_phi[int(10/timestep):])
    
    if calcul_mean:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(mean_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(mean_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Mean of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_mean_new'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_new'+text_allv+'removingvx.csv',mean_list,delimiter=',')
            
    if calcul_var:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(var_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(var_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Var of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_var'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH9timesMLC1_var90_112'+text_allv+'removingvx.csv',var_list,delimiter=',')
            
    if calcul_std:
        for i in range(1):
            number = number_list[i]
            plt.figure()
            plt.plot(std_list[9*i:9*(i+1)],'*')
            plt.axhline(np.mean(std_list[9*i:9*(i+1)]),color='r',label = "Mean of all")
            plt.title("Standard deviation of order parameter, "+str(number)+" people, timestep of 0.05, big H, removing vx")
            plt.legend()
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/'+str(number)+'/1009_occupancy'+str(number)+'time90timestep005xfullbigH9timesMLC1_std'+text_allv+'removingvxvoconstant.png')
            plt.close()
        np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH9timesMLC1_std90_112'+text_allv+'removingvx.csv',std_list,delimiter=',')       
#        
# 




mean_list1009_90=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH6timesMLC1_mean90_112_newonlyvxremovingvx.csv',delimiter=',')

mean_list1009_18=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/1009_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_newonlyvxremovingvx.csv',delimiter=',')

mean_list2908=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2908_occupancy90to12time90timestep005xfullbigH9timesMLC1_mean90_112_newonlyvxremovingvx.csv',delimiter=',')

mean_list2708=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/2708_occupancy75to18time90timestep005xfullbigH9timesMLC1_mean75_18onlyvxremovingvx.csv',delimiter=',')

mean_list2708_90=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/phase parameter/pkr/90/2708_occupancy90time90timestep005xfullbigH9timesMLC1_mean90_newonlyvxremovingvx.csv',delimiter=',')

# mean_list2708=np.concatenate([mean_list2708_90,mean_list2708])
# 
# mean_list1009=np.concatenate([mean_list1009_90,mean_list1009_18])
# 
# occ1009 = hf.occlist1908()
# occ2708= hf.occlist2708()
# occ2908 = hf.occlist2908()
# plt.plot(occ2908,mean_list2908,'o',color='b',markersize=3)
# plt.plot(occ1009,mean_list1009,'o',color='b',markersize=3)
# plt.plot(occ2708,mean_list2708,'o',color='b',markersize=3)
# plt.xlabel("Occupancy")
# plt.ylabel("Order parameter")
# plt.title("Mean of order parameter, from 6 to 96 people, removing vx")
# plt.show()
