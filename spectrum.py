#CRISEO Alexandre
#CID 01604586
#Imperial College, 2018-2019, MSC Applied Mathematics


"""
Computations and plots of spectra

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scsig
import helpful_functions as hf


    
###Plots and saves all welch spectrum through occupancy, for all simulations, removing first 10s
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