import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scsig
import helpful_functions as hf

vx180=np.genfromtxt('D:/Project/new_occupancy/72/1908_1_occupancy72time180timestep01positionrandomradiusrandomvxfullbigHMLC1.csv', delimiter=',')


    
def mean_bestfft_list2708(welch):


    str_welch = ""
    if welch :
        str_welch = 'welchpsd'
        
    number_list =[75,60,45,36,30,24,18]
    result_list=[]
    result_mean_list=[]

    for i in range(9):
        iteration = i
        number = 90
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        for j in range(90):
            if welch :
                freq,spectre = welch_psd_calc(vx[j])
                result=freq[spectre.argsort()[::-1][0]]
            else : 
                freq,spectre = bestfft_calc(vx[j])
                result=freq[spectre.argsort()[::-1][1]]
            if result > 0.5 :
                result=1-result
            result_list.append(result)
    result_mean_list.append(np.mean(result_list))
    result_mean_list7518=[]
        
    for i in range(63):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        x,y,vx,vy,occ = hf.parameters_2708(number,iteration)
        for j in range(number):
            if welch :
                freq,spectre = welch_psd_calc(vx[j])
                result=freq[spectre.argsort()[::-1][0]]
            else : 
                freq,spectre = bestfft_calc(vx[j])
                result=freq[spectre.argsort()[::-1][1]]
            if result > 0.5 :
                result=1-result
            result_list.append(result)
            result_list_person.append(result)
        result_mean_list7518.append(np.mean(result_list_person))
    for i in range(7):
        result_mean_list.append(np.mean(result_mean_list7518[9*i:9*i+1]))
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_frequencyresult'+str_welch+'.csv',result_list,delimiter=',')
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_meanresult'+str_welch+'.csv',result_mean_list,delimiter=',')

    
def mean_bestfft_list2908(welch):
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    result_mean_list=[]
    
    result_mean_list2908=[]
    result_list=[]
    
    str_welch = ""
    if welch :
        str_welch = 'welchpsd'
    
    for i in range(108):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        for j in range(number):
            if welch :
                freq,spectre = welch_psd_calc(vx[j])
                result=freq[spectre.argsort()[::-1][0]]
            else : 
                freq,spectre = bestfft_calc(vx[j])
                result=freq[spectre.argsort()[::-1][1]]
            if result > 0.5 :
                result=1-result
            result_list.append(result)
            result_list_person.append(result)
        result_mean_list2908.append(np.mean(result_list_person))
    for i in range(12):
        result_mean_list.append(np.mean(result_mean_list2908[9*i:9*i+1]))

    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_frequencyresult'+str_welch+'.csv',result_list,delimiter=',')
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_meanfrequencyresult'+str_welch+'.csv',result_mean_list,delimiter=',')
    
def mean_bestfft_list2908_maxvaluespectrum(welch):
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    result_mean_list=[]
    
    result_mean_list2908=[]
    result_list=[]
    
    str_welch = ""
    if welch :
        str_welch = 'welchpsd'
    for i in range(108):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        for j in range(number):
            if welch :
                freq,spectre = welch_psd_calc(vx[j])
                result=spectre[spectre.argsort()[::-1][0]]
            else : 
                freq,spectre = bestfft_calc(vx[j])
                result=spectre[spectre.argsort()[::-1][1]]
            result_list.append(result)
            result_list_person.append(result)
        result_mean_list2908.append(np.mean(result_list_person))
    for i in range(12):
        result_mean_list.append(np.mean(result_mean_list2908[9*i:9*i+1]))
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_maxvaluelistresult'+str_welch+'.csv',result_list,delimiter=',')
    
    np.savetxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_maxvaluemeanresult'+str_welch+'.csv',result_mean_list,delimiter=',')



def mean_bestfft_plot():
    
    result2708 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_listresult.csv',delimiter=',')
    
    result2908 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_listresult.csv',delimiter=',')
    
    occ2708=hf.occlistforeachperson(2708)
    occ2908=hf.occlistforeachperson(2908)
    
    plt.plot(occ2708,result2708/0.05,'o',color='b',markersize=1)

    
    plt.plot(occ2908,result2908/0.05,'o',color='b',markersize=1)
    plt.xlabel("Occupancy")
    plt.ylabel("Frequency")
    plt.show()
    
def plotallwelch():
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    
    for i in range(108):
        iteration = i
        number = number_list[i//9]
        print(number, iteration)
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        for j in range(number):
            time=len(vx[j])
            f, Pxx_den = scsig.welch(vx[j,time//10:])
            plt.figure()
            plt.loglog(f, Pxx_den)
            plt.xlabel('frequency [Hz]',fontsize=14)
            plt.ylabel('PSD [V**2/Hz]',fontsize=14)
            plt.tick_params(axis = 'both', labelsize = 10)  
            title = "Welch spectrum, "+str(number)+" pedestrians, number "+str(j)+", occupancy " + str(occ)[0:4]
            #plt.title(title)
            plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/'+str(number)+'/'+str(iteration%9)+'/'+title+'semilogplotfontsize.png')
            plt.close()
            
    
    
    
def fft():
    
    time=vx180.shape[1]
    t = np.arange(200,time)
    aa=np.fft.fft(vx180[0,200:])
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq,np.abs(aa))
    plt.show()


def welch_psd_plot(vx): 

    time=len(vx)
    f, Pxx_den = scsig.welch(vx[time//10:])
    plt.loglog(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
def welch_psd_calc(vx): 

    time=len(vx)
    f, Pxx_den = scsig.welch(vx[time//10:])
    return f,Pxx_den
    
def welch_linearspectrum(vx):
        
    time=len(vx)
    f, Pxx_spec = scsig.welch(vx[time//10:],window='flattop', scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()
    
    
def bestfft_plot():
    
    time=vx180.shape[1]
    fe=10
    
    echantillons = vx180[0,200:]
    tfd = np.fft.fft(echantillons)
    N=len(echantillons)
    spectre = np.absolute(tfd)*2/N
    freq=np.arange(N)*1.0/(time-200)
        
    plt.figure(figsize=(10,4))
    plt.plot(freq,spectre,'r')
    plt.xlabel('f')
    plt.ylabel('A')
    plt.axis([-0.1,1/2,0,spectre.max()])
    plt.grid()
    
    plt.show()
    
def bestfft_calc(vx):
    
    time=len(vx)
    echantillons = vx[time//10:]
    tfd = np.fft.fft(echantillons)
    N=len(echantillons)
    spectre = np.absolute(tfd)*2/N
    freq=np.arange(N)*1.0/(time-time//10)
    return freq,spectre
    
    


def trouversimulations():
    occ2708=hf.occlistforeachperson(2708)
    occ2908=hf.occlistforeachperson(2908)
    
    # 
    # 
    # resultamplitude2908 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_maxvaluelistresult.csv',delimiter=',')
    # 
    # resultamplitude2708 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_maxvaluelistresult.csv',delimiter=',')
    
    # resultfreq2908 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_listresult.csv',delimiter=',')
    # 
    # resultfreq2708 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2708_occupancy90to18time90timestep005xfullbigH9timesMLC1_listresult.csv',delimiter=',')
    # 
    # 
    # resultperiod2908 = np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/spectrum/2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_periodlistresult.csv',delimiter=',')
    
    resultperiod2908 = np.genfromtxt('2908_occupancy90to18time90timestep005xfullbigH9timesMLC1_periodlistresult.csv',delimiter=',')
    
    liste_i=[]
    for i in range(len(resultperiod2908)):
        if resultperiod2908[i]*0.05<4:
            if 0.3<occ2908[i]<0.45:
                liste_i.append(i)
       
    list_occ=[]
    for i in liste_i:
        list_occ.append(occ2908[i])
            
    list_iter = liste_iteration()
    
    list_same40=[]
    
    for i in liste_i:
        list_same40.append(list_iter[i]) 
    
    return liste_i,list_occ,list_same40

def liste_iteration():
    
    list_iter=[]
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    for i in range(108):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        for j in range(number):
            list_iter.append((number,iteration,j))
            
    return list_iter
    
#     
# liste_i,list_occ,list_same40= trouversimulations()

def plotbon():
    for i in list_same40:
        number,iteration,j=i
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        print(i)
        plt.plot(vx[j])
        #plt.plot(x[j])
        plt.show()

# freq,spectre=bestfft_calc(vx[10])

def bonamplitude():
    
    result=[]
    best_amplitude = resultamplitude2908.argsort()[::-1][:100]
    list_iter=liste_iteration()
    for i in best_amplitude:
        (number,iteration,j) = list_iter[i]
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        if 0.3<occ<0.45:
            result.append(list_iter[i])
            
    return result
        
        
#resultamp = bonamplitude()

def plotbonamplitude():
    
    for i in resultamp:
        number,iteration,j=i
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        print(i)
        plt.plot(vx[j])
        #plt.plot(x[j])
        plt.show()
        



# number,iteration,j=45,59,26
# x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
# 
# freq,spectre=bestfft_calc(vx[j])
# plt.plot(freq,spectre)
# plt.xlabel("Discrete frequency")
# plt.ylabel("Module of spectrum")
# plt.show()

# time = np.arange(0,90.1,0.05)
# 
# fig, ax1 = plt.subplots()
# 
# major_ticks = np.arange(0, 90, 1)    
# ax1.set_xticks(major_ticks)     
# ax1.grid(which='major', alpha=1) 
# ax2 = ax1.twinx()
# ax1.plot(time,vx[j], 'g-')
# ax2.plot(time,x[j], 'b--')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Vx (m/s)', color='g')
# ax2.set_ylabel('X (m)', color='b')

# 
# plt.show()