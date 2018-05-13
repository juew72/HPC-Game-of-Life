#Second Part of Game of Life 

import numpy
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
stat = MPI.Status()

fig = plt.figure()

prob = 0.2 # and try 0.4; 0.5; 0.75; 0.9
COLS = 400
ROWS = 198
generations = 100

if size > ROWS:
        print("Not enough ROWS")
        exit()

subROWS=ROWS//size+2

N = numpy.random.binomial(1,prob,size=subROWS*COLS)
M = numpy.reshape(N,(subROWS,COLS))

def msgUp(M):
        comm.send(M[subROWS-2,:],dest=rank+1)
        M[subROWS-1,:]=comm.recv(source=rank+1)
        return 0
    
def msgDn(M):
        comm.send(M[1,:],dest=rank-1)
        M[0,:] = comm.recv(source=rank-1)
        return 0

def computeGridPoints(M):
    for ROWelem in range(1,subROWS-1):
        for COLelem in range(1,COLS-1):
            sum = (M[ROWelem-1,COLelem-1]+M[ROWelem-1,COLelem]+M[ROWelem-1,COLelem+1]
                                    +M[ROWelem,COLelem-1]+M[ROWelem,COLelem+1]
                                    +M[ROWelem+1,COLelem-1]+M[ROWelem+1,COLelem]+M[ROWelem+1,COLelem+1])
            if M[ROWelem,COLelem] == 1: # if cell is alive
                if sum < 2:
                    intermediateM[ROWelem,COLelem] = 0 # cell < 2 neighbors dies 
                elif sum > 3:
                    intermediateM[ROWelem,COLelem] = 0 # cell > 3 neighbors dies 
                else:
                    intermediateM[ROWelem,COLelem] = 1 # cell = 2|3 neighbors lives on 
            if M[ROWelem,COLelem] == 0: # if cell is dead 
                if sum == 3:
                    intermediateM[ROWelem,COLelem] = 1 # cell = 3 neighbors becomes alive
                else:
                    intermediateM[ROWelem,COLelem] = 0 # cell < 3 < neighbors stays dead
    return 0


generation = 0
ims=[]
for i in range(generations):
    generation = generation + 1
    intermediateM = numpy.copy(M)
    computeGridPoints(M)
    M = numpy.copy(intermediateM) 
    finalGrid = comm.gather(M[1:subROWS-1,:],root=0)
    if rank == 0:
        result= numpy.vstack(finalGrid)
        im=plt.imshow(result, animated=True,interpolation='None')
        ims.append([im])

if rank ==0:
    print("Present Generation = %d" %(generation))
    ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True,repeat_delay=500)
    ani.save('animate_life_Part2.mp4')

plt.show()
