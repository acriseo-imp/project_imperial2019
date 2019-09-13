import numpy as np
import matplotlib.pyplot as plt
import ped_utils as putils
import helpful_functions as hf 

x72_180=np.genfromtxt('D:/Project/new_occupancy/72/1908_1_occupancy72time180timestep01positionrandomradiusrandomxfullbigHMLC1.csv', delimiter=',') #x_full[0]
y72_180=np.genfromtxt('D:/Project/new_occupancy/72/1908_1_occupancy72time180timestep01positionrandomradiusrandomyfullbigHMLC1.csv', delimiter=',') #x_full[1]

vx72_180=np.genfromtxt('D:/Project/new_occupancy/72/1908_1_occupancy72time180timestep01positionrandomradiusrandomvxfullbigHMLC1.csv', delimiter=',') #v_full[0]
vy72_180=np.genfromtxt('D:/Project/new_occupancy/72/1908_1_occupancy72time180timestep01positionrandomradiusrandomvyfullbigHMLC1.csv', delimiter=',')#v_full[1]

x60_180=np.genfromtxt('D:/Project/new_occupancy/60/1908_2_occupancy60time180timestep01positionrandomradiusrandomxfullbigHMLC1.csv', delimiter=',') #x_full[0]
y60_180=np.genfromtxt('D:/Project/new_occupancy/60/1908_2_occupancy60time180timestep01positionrandomradiusrandomyfullbigHMLC1.csv', delimiter=',') #x_full[1]

vx60_180=np.genfromtxt('D:/Project/new_occupancy/60/1908_2_occupancy60time180timestep01positionrandomradiusrandomxfullbigHMLC1.csv', delimiter=',') #v_full[0]
vy60_180=np.genfromtxt('D:/Project/new_occupancy/60/1908_2_occupancy60time180timestep01positionrandomradiusrandomxfullbigHMLC1.csv', delimiter=',')#v_full[1]




def number_area(x1,x2,y1,y2,x_full,y_full): #here x_full is (people,1), only one element of time
    #return number of people under area and that list of people
    area = (y2-y1)*(x2-x1)
    counter = 0
    list = []
    if x1>x2 or y1>y2 :
        return ("wrong values")
    number_people = x_full.shape[0]
    for i in range(number_people):
        x = x_full[i]
        y = y_full[i]
        if x1<x<x2 and y1<y<y2:
            counter += 1/area
            list.append(i)
    return counter,list
    
def number_area_number(x1,x2,y1,y2,x_full,y_full): #here x_full is (people,1), only one element of time
    #return number of people under area
    area = (y2-y1)*(x2-x1)
    counter = 0
    list = []
    if x1>x2 or y1>y2 :
        return ("wrong values")
    number_people = x_full.shape[0]
    for i in range(number_people):
        x = x_full[i]
        y = y_full[i]
        if x1<x<x2 and y1<y<y2:
            counter += 1/area
            list.append(i)
    return counter
  
def densitythroughtime(x1,x2,y1,y2,x_full,y_full):
    
    Time = x_full.shape[1]    
    all_counter = np.zeros(Time)
    for t in range(Time):
        all_counter[t]=number_area_number(x1,x2,y1,y2,x_full[:,t],y_full[:,t])
    return all_counter
    
def densitythroughtimeplot(x1,x2,y1,y2,x_full,y_full):
    
    plt.plot(densitythroughtime(x1,x2,y1,y2,x_full,y_full))
    plt.show()


def isInArea(x1,x2,y1,y2,x,y):
    
    if x1<x<x2 and y1<y<y2:
        return True
    else : 
        return False
    
          
def velocity(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full):
    
    area = (y2-y1)*(x2-x1)
    Time = vx_full.shape[1]
    all_avgspeed = np.zeros(Time)
    for t in range(Time):
        sum = 0
        N,list = number_area(x1,x2,y1,y2,x_full[:,t],y_full[:,t])
        all_avgspeed[t]=np.sum(vx_full[list,t])/(area*N)
    
    return all_avgspeed
    
def velocitythroughtimeplot(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full):

    plt.plot(velocity(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full))
    plt.show()
    
def velocity_through_density(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full):
    
    speed = velocity(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full)
    dens = densitythroughtime(x1,x2,y1,y2,x_full,y_full)
    plt.plot(dens,speed,'*')
    plt.show()
    
# -1,1,-1,1,x72_180,y72_180,vx72_180,vy72_180
# -1,1,-1,1,x60_180,y60_180,vx60_180,vy60_180


def continuity_equation(x1,x2,y1,y2,x,y,vx,vy,x_0=0,h=2,deltax=0.5):
    
    #value of continuity equation for one x, through time. We decide of h and delta x
    people = vx.shape[0]
    Time = vx.shape[1]
    x_full = np.zeros([2,people,Time])
    x_full[0]=x
    x_full[1]=y
    v_full = np.zeros([2,people,Time])
    v_full[0]=vx
    v_full[1]=vy
    dens = densitythroughtime(x_0-deltax,x_0+deltax,y1,y2,x,y)
    div = np.zeros(Time-1)
    dens_xh=densitythroughtime(x_0+h-deltax,x_0+h+deltax,y1,y2,x,y) #p(x+h)
   
    dens_x_h=densitythroughtime(x_0-deltax-h,x_0+deltax,y1,y2,x,y) #p(x-h)
    dens_x=densitythroughtime(x_0-deltax,x_0+deltax,y1,y2,x,y)

    
    for t in range(Time-1) : 
        ddens=dens[t+1]-dens[t]
        U_x = putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[x_0,0])
        U_x_h = putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[x_0-h,0]) #velocity(x_0-deltax,x_0+deltax,y1,y2,x,y,vx,vy)
        U_xh = putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[x_0+h,0]) #U(x+h) #velocity(x_0+h-deltax,x_0+h+deltax,y1,y2,x,y,vx,vy)
        div[t] = (U_xh*dens_x[t]+U_x*dens_xh[t]-U_x*dens_x_h[t]-dens_x[t]*U_x_h)/(2*h) +ddens #(U_xh*dens_xh-U_x*dens_x)/h
    return div
    
def continuity_equation_mean_x(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,h=2,deltax=0.5):
    #value of mean of continuity equation for all x, through time. We decide of h and delta x
    mean_continuity = []
    value_x = np.arange(-1,1,0.1)
    for x in value_x :
        mean_continuity.append(np.mean(continuity_equation(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,x,h,deltax)))
    return mean_continuity
    
def continuity_equation_mean_x_plot(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,h=2,deltax=0.5):
    
    plt.plot(np.arange(-1,1,0.1),continuity_equation_mean_x(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,h=2,deltax=0.5),'*')
    plt.title("Average of Equation of continuity, h = " + str(h)+", deltax = "+str(deltax) + " 60 people 180 time 01 timestep")
    plt.show()



def continuity_equation_mean_time(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,h=2,deltax=0.5):
    #value of mean of continuity equation for all time, through x. We decide of h and delta x
    mean_continuity = []
    value_x = np.arange(-1,1,0.1)
    continuity_0 = continuity_equation(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,value_x[0],h,deltax)
    mean_continuity = np.zeros([value_x.size,continuity_0.size])
    mean_continuity[0,:]=continuity_0
    for x in range(1,value_x.size) :
        mean_continuity[x,:]=continuity_equation(x1,x2,y1,y2,x_full,y_full,vx_full,vy_full,value_x[x],h,deltax)
    return np.mean(mean_continuity,axis=0)
    
    
def continuity_equation_mean_time_plot_2908(x1,x2,y1,y2,h=0.1,deltax=0.2):
    
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    time_plot=np.arange(0,90.1,0.05)
    
    for i in range(0,108,9):
        plt.figure()
        dens_list = []
        iteration = 63
        number = 36
        print(number,iteration)
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        plt.plot(time_plot[200:-1],continuity_equation_mean_time(x1,x2,y1,y2,x,y,vx,vy,h=2,deltax=0.5)[200:],'o', markersize=1)
        title = "h = " + str(h)+", deltax = "+str(deltax) + ", "+str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", number " +str(iteration)
        #plt.title(title)
        plt.tick_params(axis = 'both', labelsize = 10)  
        plt.xlabel("Time",fontsize=14)
        plt.ylabel("Average value of density equation through x",fontsize=14)
        plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/density/average value through time/'+title+'newcalculationofdiv10sremovedlocalspeedfontsize.png')
        plt.close()

    
def continuity_equation_mean_x_plot_2908(x1,x2,y1,y2,h=2,deltax=0.5):
    
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    time_plot=np.arange(0,90.1,0.05)
    value_x = np.arange(-1,1,0.1)
    for i in range(0,108,9):
        plt.figure()
        dens_list = []
        iteration = i
        number = number_list[i//9]
        print(number,iteration)
        x,y,vx,vy,occ = hf.parameters_2908_005('MLC1',number,iteration)
        plt.plot(value_x,continuity_equation_mean_x(x1,x2,y1,y2,x,y,vx,vy,h=2,deltax=0.5),'o', markersize=3)
        title = "h = " + str(h)+", deltax = "+str(deltax) + ", "+str(number)+" pedestrians, occupancy of "+str(occ)[0:4]+", number " +str(iteration)
        plt.title(title)
        plt.xlabel("X values")
        plt.ylabel("Average value of density equation through x")
        plt.savefig('C:/Users/alexa/Documents/Alexandre/Imperial/Cours/Project/model19/results/occupancy/density/average value through x/'+title+'newcalculationofdivlocalspeed.png')
        plt.close()


def localspeedthroughtime():
    
    #plot local speed for one simulation through time
    
    number_list =[90,82,75,67,60,52,45,36,30,24,18,12]
    localspeed=[]
    # for i in range(0,108,9):
    #     plt.figure()
    #     dens_list = []
    #     iteration = i
    #     number = number_list[i//9]
    #     print(number,iteration)
    
    x_full,v_full = hf.parameters_xvfull(2908,67,30)
    Time = x_full.shape[2]
    for t in range(Time):
        localspeed.append(putils.local_speed_current(x_full[:,:,t],v_full[:,:,t],[0,0]))
    plt.plot(np.arange(0,90.1,0.05),localspeed,'o',markersize=1)
    plt.show()
    
    
    
    
        
    
        
        
        