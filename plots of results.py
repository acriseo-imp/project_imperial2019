

import matplotlib.pyplot as plt
import numpy as np
import ped_utils as putils
import helpful_functions as hf
### Comparizon of num_cores

def compar_num_cores():

    MLC_01_8c=np.array([138,294,569,828,1522,2387,3538]) #value of occupancy 12 not correct
    
    Person = [6,12,18,24,36,48,60]
    
    
    MLC_01_6c = np.array([137,312,614,859,1553,2474,3587])
    
    
    MLC_01_4c = np.array([151,467,680,947,1685,2690,4022])
    
    MLC_01_2c = np.array([259,569,948,1410,2581,4306,9586])
    
    MLC_01_1c =np.array([541,1318,2111,4131,7603,11359,16013])
    
    
    MLC_0025_8c = [547.5,1239,2284,3289,5995,9468,999999999]
    
    MLC_001_8c = [1378,3314,5612,8161,999999999,999999999,999999999]
    
    plt.figure(1)
    
    plt.plot(Person,MLC_01_2c/MLC_01_8c,'*',label = "2 cores / 8 cores")
    plt.plot(Person,MLC_01_4c/MLC_01_8c,'*',label = "4 cores / 8 cores")
    plt.plot(Person,MLC_01_6c/MLC_01_8c,'*',label = "6 cores / 8 cores")
    plt.plot(Person,MLC_01_1c/MLC_01_8c,'*',label = "1 core / 8 cores")
    plt.xlabel("Number of people")
    plt.ylabel("Time taken (s)")
    plt.title("time taken = f(num_cores,people)")
    
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(Person,MLC_01_2c/MLC_01_1c,'*',label = "2 cores / 1 core")
    plt.plot(Person,MLC_01_4c/MLC_01_1c,'*',label = "4 cores / 1 core")
    plt.plot(Person,MLC_01_6c/MLC_01_1c,'*',label = "6 cores / 1 core")
    plt.plot(Person,MLC_01_8c/MLC_01_1c,'*',label = "8 cores / 1 core")
    #plt.plot(Person,MLC_01_2c,'*',label = "2 cores")
    
    # plt.loglog(Person,MLC_01_8c,'*',label = "8 cores")
    # plt.loglog(Person,MLC_01_6c,'*',label = "6 cores")
    # plt.loglog(Person,MLC_01_4c,'*',label = "4 cores")
    # plt.loglog(Person,MLC_01_2c,'*',label = "2 cores")
    
    plt.xlabel("Number of people")
    plt.ylabel("Time taken (s)")
    plt.title("time taken = f(num_cores,people)")
    
    plt.legend(loc='best')
    
    plt.figure(3)
    plt.plot(Person,MLC_01_1c/MLC_01_2c,'*',label = "1 core / 2 core")
    plt.plot(Person,MLC_01_1c/MLC_01_4c,'*',label = "1 core / 4 core")
    plt.plot(Person,MLC_01_1c/MLC_01_6c,'*',label = "1 core / 6 core")
    plt.plot(Person,MLC_01_1c/MLC_01_8c,'*',label = "1 core / 8 core")
    #plt.plot(Person,MLC_01_2c,'*',label = "2 cores")
    
    # plt.loglog(Person,MLC_01_8c,'*',label = "8 cores")
    # plt.loglog(Person,MLC_01_6c,'*',label = "6 cores")
    # plt.loglog(Person,MLC_01_4c,'*',label = "4 cores")
    # plt.loglog(Person,MLC_01_2c,'*',label = "2 cores")
    
    plt.xlabel("Number of people")
    plt.ylabel("Time taken (s)")
    plt.title("time taken = f(num_cores,people)")
    
    plt.legend(loc='best')
    
    #plt.ylim([0,2])
    #plt.xlim([5,50])
    plt.show()

### Comparizon of timestep

def compar_timestep():
    plt.plot(Person,MLC_01_8c,'*',label = "timestep 0.1")
    plt.plot(Person,MLC_0025_8c,'*',label = "timestep 0.025")
    plt.plot(Person,MLC_001_8c,'*',label = "timestep 0.01")
    
    plt.legend()
    
    plt.title("time taken = f(timestep,people)")
    
    plt.ylim([0,10000])
    plt.show()


### Plots (avg_speed = f(occupancy)
MLC_avgspeed_0025_8c = [1.133,1.287,1.241,1.21,1.17,1.103,0.822]

MLC_occupancy_0025_8c = [0.062,0.105,0.168,0.201,0.303,0.382,0.503]

##new occupancy, new H, mass and radius not random

def newoccnewhnorandom():
    MLC_occupancy_01_newH = [0.196,0.294,0.393,0.49,0.589,0.687,0.785]
    
    MLC_avg_01_newH = [1.270,1.15,1.056,0.797,0.456,0.221,0.10]
    
    MLC_occupancy_0025_newH=[0.196,0.295,0.393,0.49] #60 stopped at 72
    MLC_avg_0025_newH=[1.247,1.141,1.103,0.795]	

    
    MLC_occupancy_001_newH= [0.196,0.295,0.393] #48 stopped at 30 sec, 36 at 39
    MLC_avg_001_newH =[1.267,1.158,0.927]
    
    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH,'o',markersize=3,label="timestep = 0.025")
    plt.plot(MLC_occupancy_001_newH,MLC_avg_001_newH,'o',markersize=3,label="timestep = 0.01")
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH,'o',markersize=3,label="timestep = 0.1")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.6])
    plt.title("New calculation of occupancy, big H, mass and radius not random")
    plt.show()


def newoccnewhallrandom(): #A REPLOT c'est fait
    
    MLC_occupancy_01_newH=[0.207,0.282,0.377,0.492,0.593,0.693,0.781]
    
    MLC_avg_01_newH=[1.256,1.175,1.171,0.814,0.458,0.266,0.184]
    
    plt.figure()
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH,'o',markersize=3,label="mass and radius random")
    
    MLC_occupancy_01_newH = [0.196,0.294,0.393,0.49,0.589,0.687,0.785]
    
    MLC_avg_01_newH = [1.270,1.15,1.056,0.797,0.456,0.221,0.10]
    
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH,'o',markersize=3,label="mass and radius not random")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.85])
    plt.title("New calculation of occupancy, big H, timestep of 0.1")
    
    plt.figure()
    MLC_occupancy_0025_newH = [0.203,0.336,0.409,0.506,0.611,0.691,0.837]
    MLC_avg_0025_newH = [1.224,1.124,1.135,0.757,0.487,0.220,0.180]
    
    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH,'o',markersize=3,label="mass and radius random")
    
    MLC_occupancy_0025_newH=[0.196,0.295,0.393,0.49] 
    MLC_avg_0025_newH=[1.247,1.141,1.103,0.795]	
    
    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH,'o',markersize=3,label="mass and radius not random")
    
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.85])
    plt.title("New calculation of occupancy, big H, timestep of 0.025")
    
    plt.show()

def newoccoldhnotrandom():

    MLC_occupancy=[0.049,0.098,0.147,0.196,0.294,0.393,0.49]
    MLC_avg_01_oldH=[1.336,1.248,1.266,1.171,1.210,1.110,1.06]
    MLC_avg_0025_oldH =[1.286,1.193,1.255,1.243,1.14,1.140,1.030]
    MLC_avg_001_oldH = [1.345,1.255,1.218,1.236,-1,-1,-1]
    
    plt.plot(MLC_occupancy,MLC_avg_0025_oldH,'o',markersize=3,label="timestep = 0.025")
    plt.plot(MLC_occupancy,MLC_avg_001_oldH,'o',markersize=3,label="timestep = 0.01")
    plt.plot(MLC_occupancy,MLC_avg_01_oldH,'o',markersize=3,label="timestep = 0.1")
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0,0.6])
    plt.ylim([0,1.5])
    plt.title("New calculation of occupancy, small H, mass and radius not random")
    plt.show()

def oldoccbighrandom():
    
    MLC_01_8c_occ=[0.047,-1,0.15,0.194,0.306,0.397,0.523]
    MLC_01_8c_avg=[1.284,-1,1.30,1.18,1.192,1.099,0.778]
    
    MLC_01_6c_occ=[0.053,0.099,0.138,0.194,0.291,0.421,0.500]
    MLC_01_6c_avg=[1.392,1.246,1.226,1.192,1.198,1.007,0.84]

    MLC_01_4c_occ=[0.05,0.09,0.161,0.224,0.314,0.393,0.48]
    MLC_01_4c_avg=[1.283,1.266,1.254,1.257,1.168,1.117,0.873]
    
    MLC_01_2c_occ=[0.049,0.106,0.146,0.206,0.298,0.403,0.519]
    MLC_01_2c_avg=[1.249,1.199,1.329,1.245,1.199,1.059,0.783]
    
    MLC_01_1c_occ=[0.048,0.099,0.152,0.202,0.3,0.374,0.514]
    MLC_01_1c_avg=[1.242,1.245,1.373,1.232,1.170,1.104,0.799]
    
    plt.plot(MLC_01_8c_occ,MLC_01_8c_avg,'o',markersize=3,label="timestep = 8 cores")
    plt.plot(MLC_01_6c_occ,MLC_01_6c_avg,'o',markersize=3,label="timestep = 6 cores")
    plt.plot(MLC_01_4c_occ,MLC_01_4c_avg,'o',markersize=3,label="timestep = 4 cores")
    plt.plot(MLC_01_2c_occ,MLC_01_2c_avg,'o',markersize=3,label="timestep = 2 cores")
    plt.plot(MLC_01_1c_occ,MLC_01_1c_avg,'o',markersize=3,label="timestep = 1 core")
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0,0.6])
    plt.ylim([0,1.5])
    plt.title("Old calculation of occupancy, big H, mass and radius random")
    plt.show()
    
 
def comparizon_old_new(): #A REPLOT c'est fait
    
    MLC_01_8c_occ=[0.194,0.306,0.397,0.523]
    MLC_01_8c_avg=[1.18,1.192,1.099,0.778]
    
    "[0.05,0.09,0.161,"
    "[1.283,1.266,1.254,"
    MLC_01_4c_occ=[0.224,0.314,0.393,0.48,0.626,0.700,0.807]
    MLC_01_4c_avg=[1.257,1.168,1.117,0.873,0.583,0.442,0.415]

    MLC_occupancy_01_newH=[0.207,0.282,0.377,0.492,0.593,0.693,0.781]

    MLC_avg_01_newH=[1.256,1.175,1.171,0.814,0.458,0.266,0.184]
    
    MLC_avg_01_newH_vx= [999,1.16,1.16,0.8,0.39,0.19,0.09]
    
    plt.plot(MLC_01_4c_occ,MLC_01_4c_avg,'o',markersize=3,label="old occupancy")
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH,'o',markersize=3,label="new occupancy")
    
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH_vx,'o',markersize=3,label="new occupancy, only vx")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.15,0.85])
    plt.ylim([0,1.5])
    plt.title("Comparizon of the two methods, mass and radius random, timestep = 0.1")
    plt.show()
    
def comparizon_big_small_h(): #A REPLOT
    
    MLC_occupancy=[0.196,0.294,0.393,0.49,0.589,0.687,0.785]
    MLC_avg_01_oldH=[1.171,1.210,1.110,1.06,0.794,0.635,0.491]
    MLC_avg_01_newH = [1.270,1.15,1.056,0.797,0.456,0.221,0.10]
    
    MLC_avg_01_newH_vx= [999,1.13,1.06,0.77,0.4,0.16,0.03]
    MLC_avg_01_oldH_vx= [999,1.2,1.13,0.78,0.74,0.53,0.36]
    
    plt.plot(MLC_occupancy,MLC_avg_01_oldH,'o',markersize=3,label="small H")
    plt.plot(MLC_occupancy,MLC_avg_01_newH,'o',markersize=3,label="big H")
    plt.plot(MLC_occupancy,MLC_avg_01_oldH_vx,'o',markersize=3,label="small H vx")
    plt.plot(MLC_occupancy,MLC_avg_01_newH_vx,'o',markersize=3,label="big H vx")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.legend()
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.8])
    plt.ylim([0,1.5])
    plt.title("Comparizon of new method with different H, mass and radius not random, timestep = 0.1")
    plt.show()
    
def avgspeed_test():

    
    datav1 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 3/results/occupancy/1907_2_occupancy6time90timestep0025vxfullMLC3.csv', delimiter=',')
    
    sizedata=datav1.shape[1]
    plt.plot(0,np.average(datav1),'*')
    for i in range (1,6):
        plt.plot(i,np.average(datav1[:,(sizedata//(10*i)):]),'*')
        
    datav2 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 4/results/occupancy/1907_1_occupancy12time90timestep025vxfullMLC4.csv', delimiter=',')
    
    sizedata=datav2.shape[1]
    plt.plot(0,np.average(datav2),'*')
    for i in range (1,6):
        plt.plot(i,np.average(datav2[:,(sizedata//(10*i)):]),'*')
        
    datav3 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 4/results/occupancy/1907_2_occupancy18time90timestep025vxfullMLC4.csv', delimiter=',')
    
    sizedata=datav3.shape[1]
    plt.plot(0,np.average(datav3),'*')
    for i in range (1,6):
        plt.plot(i,np.average(datav3[:,(sizedata//(10*i)):]),'*')
        
    datav4 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 3/results/occupancy/1907_3_occupancy24time90timestep0025vxfullMLC3.csv', delimiter=',')
    
    sizedata=datav4.shape[1]
    plt.plot(0,np.average(datav4),'*')
    for i in range (1,6):
        plt.plot(i,np.average(datav4[:,(sizedata//(10*i)):]),'*')
    
    plt.legend()
    
    #plt.plot(MLC_occupancy_0025_8c,MLC_avgspeed_0025_8c)
    
    v_full_60 = np.genfromtxt('D:/Project/Sauvegarde 20_07/Project MLC 1/results/occupancy/2007_1_occupancy60time90timestep01vxfullMLC1.csv',delimiter=',')
    
    plot_pub = plt.plot( data[:,0], data[:,1], 'r', label = 'Published Results')
    plt.plot(MLC_occupancy_0025_8c,MLC_avgspeed_0025_8c)
    plt.grid()
    plt.show()


def comparizon010025newdest():
    
    MLC_occupancy_01_newH=[0.207,0.282,0.377,0.492,0.593,0.693,0.781]
    
    MLC_avg_01_newH=[1.256,1.175,1.171,0.814,0.458,0.266,0.184]
    
    MLC_avg_01_newH_vx= [999,1.16,1.16,0.8,0.39,0.19,0.09]
    
    MLC_occupancy_0025_newH = [0.203,0.336,0.409,0.506,0.611,0.691,0.837]
    
    MLC_avg_0025_newH = [1.224,1.124,1.135,0.757,0.487,0.220,0.180]
    
    MLC_occupancy_005_smallH = [0.292,0.417,0.524,0.616,0.705,0.771]
    
    MLC_avg_005_smallH_vx= [1.06,999,0.62,0.52,0.41,0.33]
    
    MLC_avg_005_smallH = [1.092,0.837,0.685,0.594,0.486,0.443]
    
    MLC_avg_0025_newH_vx= [999,1.11,1.12,0.72,0.42,0.16,0.09]
    
    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH,'o',color='y',markersize=3,label="timestep of 0.025")
    
    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH_vx,'o',color='y',markersize=3,label="timestep of 0.025 vx")
    
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH,'o',color='c',markersize=3,label="timestep of 0.1")
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH_vx,'o',color='c',markersize=3,label="timestep of 0.1 vx")
    
    plt.plot(MLC_occupancy_005_smallH,MLC_avg_005_smallH,'o',color='m',markersize=3,label="timestep of 0.05, new dest")
    
    plt.plot(MLC_occupancy_005_smallH,MLC_avg_005_smallH_vx,'o',color='m',markersize=3,label="timestep of 0.05, new dest vx")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.85])
    plt.ylim([0,1.5])
    plt.title("New calculation of occupancy, big H, random")
    
    data_agent=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/Texts/Density, utility function/An Agent-Based Microscopic Pedestrian Flow 2.csv',delimiter=',')
    
    data_agent=data_agent[:,:3]
    
    plt.plot(data_agent[:,0]*0.15,data_agent[:,1],label="square data")
    plt.plot(data_agent[:,0]*0.15,data_agent[:,2],label="triangle data")
    plt.legend()
    plt.show()
    
def onlyvx():
    
    MLC_occupancy_005_smallH = [0.292,0.417,0.524,0.616,0.705,0.771]
    
    MLC_avg_005_smallH_vx= [1.06,999,0.62,0.52,0.41,0.33]

    MLC_occupancy_01_newH=[0.207,0.282,0.377,0.492,0.593,0.693,0.781]

    MLC_avg_01_newH_vx= [999,1.16,1.16,0.8,0.39,0.19,0.09]
    
    MLC_occupancy_0025_newH = [0.203,0.336,0.409,0.506,0.611,0.691,0.837]
    
    MLC_avg_0025_newH_vx= [999,1.11,1.12,0.72,0.42,0.16,0.09]

    plt.plot(MLC_occupancy_0025_newH,MLC_avg_0025_newH_vx,'o',color='y',markersize=3,label="timestep of 0.025 vx")
    
    plt.plot(MLC_occupancy_005_smallH,MLC_avg_005_smallH_vx,'o',color='c',markersize=3,label="timestep of 0.05, small H vx")
    
    plt.plot(MLC_occupancy_01_newH,MLC_avg_01_newH_vx,'o',color='m',markersize=3,label="timestep of 0.1 vx")
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.85])
    plt.ylim([0,1.5])
    plt.title("New calculation of occupancy, big H except new dest, random, only vx")

    plt.legend()
    plt.show()
    
def newdestbigH():
    occ = [0.31,0.382,0.507,0.582,0.693,0.724]
    avg=[0.968,0.799,0.526,0.336,0.142,0.107]
    vx = [0.94,0.77,0.48,0.29,0.09,0.06]
    
    plt.plot(occ,avg,'o',markersize=3,label="classic avg speed")
    plt.plot(occ,vx,'o',markersize=3,label="vx avg speed")
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    plt.xlim([0.1,0.85])
    plt.ylim([0,1.5])
    plt.title("New calculation of occupancy, big H new dest, random, with vx")

    plt.legend()
    plt.show()
    
def newpkr():
    
    occ_1 = [0.0520123228120834, 0.1021468991414307, 0.14311147083048306, 0.20286070213085394, 0.25406077051435788, 0.30136695704223065, 0.37553862186182219, 0.47734962328895675, 0.62887135971265984, 0.75587368777968356]
    
    avg_1 = [1.1450660934283894, 1.1551836987739839, 1.1729045652746706, 0.98311956874541362, 0.92816017861657174, 0.78701769927434662, 0.59101205342237528, 0.40600004866571787, 0.24285662852514381, 0.21272028875118987]
    
    avg_vx_1 = [1.15731114693,1.17564678668, 1.18476241176,0.985276259541,0.919913452542,0.760767733129,0.565209392354,0.371333861839,0.192538681653,0.13786920245]
    
    occ_2 = [0.053591475542012157, 0.094831211845895588, 0.15050481973101348, 0.19796063930532418, 0.22859158257496448, 0.31242068924218064, 0.36390918784361709, 0.49580288126688038, 0.62748534737696804, 0.75796079843622544]
    
    avg_2 = [1.3080824795387656, 1.1728062840694498, 1.1934406932932751, 1.0539831298221309, 0.92831854545846504, 0.7357854336087517, 0.59972401186114588, 0.3833585956356495, 0.24730175028598708, 0.20734104948960108]


    
    plt.plot(occ_1,avg_1,'o',markersize=3,label="avg speed")
    
    plt.plot(occ_2,avg_2,'o',markersize=3)
    
    plt.plot(occ_1,avg_vx_1,'x',label="avg speed, only vx")
    
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    plt.title("New calculation of occupancy, big H new dest, random, with vx")

    plt.legend()
    plt.show()
    
def speedbellomo(density,gamma):
    return (1-np.exp(-gamma*(1/density - 1)))
   
def totalpkrtotal():
     #serie de 10 simulations du 27/08
     
    number75,occ75_18,avg75_18,time75_18,iter75_18,avgvx75_18=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    number90,occ90,avg90,time90,iter90,avgvx90=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    plt.plot(number75/24,avg75_18,'o',color='k',markersize=2,label='Average speed')
        
    plt.plot(number90/24,avg90,'o',color='k',markersize=2)
        
   # plt.plot(number75/24,avgvx75_18,'o',color='burlywood',markersize=3,label='Average speed, only x component')
    
    #plt.plot(number90/24,avgvx90,'o',color='burlywood',markersize=3)
    
    
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    plt.plot(number/24,avg90_18,'o',color='k',markersize=2)
   
    #plt.plot(number/24,avgvx90_18,'o',color='burlywood',markersize=3)
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    # plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Density")
    plt.ylabel("Average speed")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    #plt.title("Average speed = f(density), big H, with vx")
    
    data_agent=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/Texts/Density, utility function/An Agent-Based Microscopic Pedestrian Flow 2.csv',delimiter=',')
    
    data_agent=data_agent[:,:3]
    
    plt.plot(data_agent[:,0],data_agent[:,1],'o',markersize=3, marker ='s',label="Weidmann data")
   # plt.plot(data_agent[:,0],data_agent[:,2],'s',markersize=3,marker='^',label="triangle data")
    plt.legend()
    plt.grid()
   # plt.legend()
    plt.show()
    
def totalpkr():
     #serie de 10 simulations du 27/08
     
    number75,occ75_18,avg75_18,time75_18,iter75_18,avgvx75_18=np.genfromtxt('D:/Project/pkr/total/27089fois75a18_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    number90,occ90,avg90,time90,iter90,avgvx90=np.genfromtxt('D:/Project/pkr/total/27089fois90_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
   # plt.plot(number75/24,avg75_18,'o',color='k',markersize=3,label='Average speed')
    
   # plt.plot(number90/24,avg90,'o',color='k',markersize=3)
    
    plt.plot(number75/24/(100/24),avgvx75_18/1.3,'o',color='burlywood',markersize=3,label='Average speed, only x component')
    
    plt.plot(number90/24/(100/24),avgvx90/1.3,'o',color='burlywood',markersize=3)
    
    
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
   # plt.plot(number/24,avg90_18,'o',color='k',markersize=3)
   
    plt.plot(number/24/(100/24),avgvx90_18/1.3,'o',color='burlywood',markersize=3)
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    # density = np.arange(0,1,0.01)
    # gamma = np.array([0.45])
    # for i in range(gamma.size):
    #     plt.plot(density,speedbellomo(density,gamma[i]),'o-',markersize=1,label="ve = f(rho), gamma="+str(gamma[i]))
    # plt.legend()
    
    #plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Density dimensionless")
    plt.ylabel("Average speed dimensionless")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    plt.title("Average speed = f(density), big H, with vx")
    
    data_agent=np.genfromtxt('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/Texts/Density, utility function/An Agent-Based Microscopic Pedestrian Flow 2.csv',delimiter=',')
    
    data_agent=data_agent[:,:3]
    
    plt.plot(data_agent[:,0]*0.15,data_agent[:,1],'o',markersize=1,label="square data")
    plt.plot(data_agent[:,0]*0.15,data_agent[:,2],'s',markersize=1,label="triangle data")
    plt.legend()

   # plt.legend()
    plt.show()
    
def pkrcompar05025():
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    
    plt.plot(occ90_18,avg90_18,'o',color='k',markersize=3,label='Average speed for a time step of 0.05')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='burlywood',markersize=3,label='Average speed for a time step of 0.05, only x component')
    
    number,occ90_75,avg90_75,time90_75,iter90_75,avgvx90_75=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    
    plt.plot(occ90_75,avg90_75,'o',color='b',markersize=3,label='Average speed for a time step of 0.025')
   
    plt.plot(occ90_75,avgvx90_75,'o',color='magenta',markersize=3, label = 'Average speed for a time step of 0.025, only x component')
    
    number,occ67_12,avg67_12,time67_12,iter67_12,avgvx67_12=np.genfromtxt('D:/Project/pkr/total/29089fois67a12_time90timestep0025resultsfullbigH9timesMLC2.csv',delimiter=',')
    
    
    plt.plot(occ67_12,avg67_12,'o',color='b',markersize=3)
   
    plt.plot(occ67_12,avgvx67_12,'o',color='magenta',markersize=3)
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    #plt.title("Average speed = f(occupancy), big H, comparison between 0.05 and 0.025")

    plt.legend()
    plt.show()
    
    
def pkrcomparperiodic():
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesMLC1.csv',delimiter=',')
    
    plt.plot(occ90_18,avg90_18,'o',color='k',markersize=3,label='Average speed for a time step of 0.05')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='burlywood',markersize=3,label='Average speed for a time step of 0.05, only x component')
    
    number,occ90_18,avg90_18,time90_18,iter90_18,avgvx90_18=np.genfromtxt('D:/Project/pkr/total/29089fois90a12_time90timestep005resultsfullbigH9timesnoperiodicMLC4.csv',delimiter=',')
    
    plt.plot(occ90_18,avg90_18,'o',color='b',markersize=3,label='Average speed for a time step of 0.05, no periodic condition')
   
    plt.plot(occ90_18,avgvx90_18,'o',color='magenta',markersize=3,label='Average speed for a time step of 0.05, only x component, no periodic condition')
    
    data = np.genfromtxt('data/data_pub_occupancy.csv', delimiter=',')
    
    
    plt.plot( data[:,0], data[:,1], 'r',label = 'Published Results')
    plt.xlabel("Occupancy")
    plt.ylabel("Average speed (m/s)")
    #plt.xlim([0.1,0.85])
    #plt.ylim([0,1.5])
    plt.title("Average speed = f(occupancy), big H, comparison between periodic condition or not")

    plt.legend()
    plt.show()
    
def time_simulation2908():
    
    liste_number = []
    liste_time = []
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    for i in range(108):
        iteration = i
        number = number_list[i//9]
        time = hf.time_2908_005anim('MLC1',number,iteration)
        liste_number.append(number)
        liste_time.append(time)
        
    for i in range(9):
        iteration = i
        number = 90
        time = hf.time_2708anim(number,iteration)
        liste_number.append(number)
        liste_time.append(time)
    
    number_list =[75,60,45,36,30,24,18]
        
    for i in range(63):
        result_list_person=[]
        iteration = i
        number = number_list[i//9]
        time = hf.time_2708anim(number,iteration)
        liste_number.append(number)
        liste_time.append(time)
    
    liste_number.sort()
    liste_time.sort()
    
    plt.plot(liste_number,np.array(liste_time),'o',markersize=2)
    #plt.plot(liste_number,0.07*np.array(liste_number)**2+7.95,'o',markersize=2)
    #plt.loglog(liste_number,np.array(liste_number)**2,'o',markersize=2, label = 2)
    #plt.plot(liste_number,np.array(liste_number)**3,'o',markersize=2, label = 3)
    #plt.legend()
    plt.grid(True, which="both")
    plt.ylabel("Time of computation")
    plt.xlabel("Number of pedestrians")
    plt.xticks(np.arange(0,100,10))
    plt.yticks(np.arange(0,650,50))
    plt.xlim([0,100])
    plt.ylim([0,650])


    plt.show()




    

