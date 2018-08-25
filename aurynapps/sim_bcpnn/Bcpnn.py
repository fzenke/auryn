import sys, os, select
import numpy as np
import random
import string
import copy

import math
import numpy as np
import copy
import itertools
from scipy.spatial import distance

def setseed(seed) :
    np.random.seed(seed)

class BcpnnConnection :

    maxfq = 200

    def __init__(self,Ni,Nj) :

        self.Ni = Ni
        self.Nj = Nj

        Zi = np.zeros(Ni)
        Zj = np.zeros(Nj)
        Pi = np.zeros(Ni)
        Pj = np.zeros(Nj)
        Pij = np.zeros((Ni,Nj))

        Bj = np.zeros(Nj)
        Wij = np.zeros((Ni,Nj))
        
    @classmethod
    def mexpand(self,N,spikes,mintime = None,maxtime = None,timestep = 0.0001) :
        # Spikes are represented as a Tx2-dimensional numpy array
        # [time,uidx] Output is a matrix with N columns respectively
        # and each row representing spikes during a timestep
        if np.min(spikes[:,1])<0 : raise AssertionError("Bcpnn.mexpand","Illegal uidx<0")
        if N<=np.max(spikes[:,1]) : raise AssertionError("Bcpnn.mexpand","Illegal N<=uidx")

        X = np.zeros(N)
        XX = [X]
        
        if mintime==None : mintime = np.min(spikes[:,0])
        if maxtime==None : maxtime = np.max(spikes[:,0])
        if mintime>maxtime : raise AssertionError("Bcpnn.mexpand","mintime>maxtime")

        TT = [mintime - timestep]

        nstep = int((maxtime - mintime)/timestep)

        sidx = 0
        while spikes[int(sidx),0]<mintime : sidx += 1

        for step in range(0,nstep) :

            t1 = mintime + step * timestep
            t2 = mintime + (step+1) * timestep
            X = np.zeros(N)
            while spikes[int(sidx),0]<t2 :
                X[int(spikes[int(sidx),1])] += 1
                sidx += 1

            TT.append(t2)
            XX.append(X)

        XX = np.array(XX)

        return XX,TT


    @classmethod
    def computespiketrace(self,N,spikes,tau,maxfq = 200,mintime = None,maxtime = None,timestep = 0.0010) :
        # Spikes are represented as a Tx2-dimensional numpy array
        # [time,uidx]. Output is a matrix with N columns respectively
        # and each row representing spikes during a timestep
        if np.min(spikes[:,1])<0 : raise AssertionError("Bcpnn.computespiketrace","Illegal uidx<0")
        if N<=np.max(spikes[:,1]) : raise AssertionError("Bcpnn.computespiketrace","Illegal N<=uidx")
        X = np.zeros(N)
        Q = np.zeros(N)
        QQ = [Q]
        TT = [0]

        ktau = math.exp(-timestep/tau)
        if (maxfq==None) :
            kinc = 1
        else :
            kinc = 1/(tau*maxfq)

        print "ktau = %.4f kinc = %.4f" % (ktau,kinc)

        if mintime==None : mintime = np.min(spikes[:,0])
        if maxtime==None : maxtime = np.max(spikes[:,0])
        if mintime>maxtime : raise AssertionError("Bcpnn.mexpand","mintime>maxtime")

        TT = [mintime - timestep]

        nstep = int((maxtime - mintime)/timestep)

        sidx = 0
        while len(spikes)<sidx and spikes[int(sidx),0]<mintime : sidx += 1

        for step in range(0,nstep) :

            t1 = mintime + step * timestep
            t2 = mintime + (step+1) * timestep

            X = np.zeros(N)
            while len(spikes)>sidx and spikes[int(sidx),0]<t2 :
                X[int(spikes[int(sidx),1])] += 1
                sidx += 1

            Q *= ktau
            Q += X * kinc

            TT.append(t2)
            QQ.append(copy.copy(Q))

        QQ = np.array(QQ)

        return QQ,TT
            
    @classmethod
    def computetracetrace1(self,trace,tau,timestep = 0.0010,sample = 1) :
        # Traces are represented as a TxN-dimensional numpy array.
        # Output is a matrix with N columns respectively
        # and each row representing spikes during a timestep

        N = trace.shape[1]
        Q = np.zeros(N)
        QQ = [Q]
        TT = [0]

        ktau = math.exp(-timestep/tau)
        kinc = 1 - ktau

        print "ktau = %.4f kinc = %.4f" % (ktau,kinc)

        for step in range(1,len(trace)) :

            Q *= ktau
            Q += kinc * trace[step]

            if step%sample==0 :
                TT.append(step * timestep)
                QQ.append(copy.copy(Q))
        

        QQ = np.array(QQ)
        TT = np.array(TT)

        return QQ,TT


    def computetracetrace2(self,tracei,tracej,tau,timestep = 0.0010,sample = 1) :
        # Traces are represented as a TxN-dimensional numpy array.
        # Output is a matrix with N columns respectively
        # and each row representing spikes during a timestep
        if tracei.shape[1]!=self.Ni :
            raise AssertionError("Bcpnn.computetracetrace2","Illegal tracei width != Ni")
        if tracej.shape[1]!=self.Nj :
            raise AssertionError("Bcpnn.computetracetrace2","Illegal tracej width != Nj")
        if len(tracei)!=len(tracej) :
            raise AssertionError("Bcpnn.computetracetrace2","trace len mismatch")

        Q = np.zeros((self.Ni,self.Nj))
        QQ = [Q]
        TT = [0]

        ktau = math.exp(-timestep/tau)
        kinc = 1 - ktau

        print "ktau = %.4f kinc = %.4f" % (ktau,kinc)

        for step in range(1,len(tracei)) :

            Q *= ktau
            Q += kinc * np.outer(tracei[step],tracej[step])

            if step%sample==0 :
                TT.append(step * timestep)
                QQ.append(copy.copy(Q))

        QQ = np.array(QQ)
        TT = np.array(TT)

        return QQ,TT

    def computeZZi(self,spikes,tauzi,mintime = 0,maxtime = None,timestep = 0.0010) :
        ZZi,TT = self.computespiketrace(self.Ni,spikes,tauzi,mintime = mintime,maxtime = maxtime,
                                        timestep = timestep)
        return ZZi,TT
    

    def computeZZj(self,spikes,tauzj,mintime = 0,maxtime = None,timestep = 0.0010) :
        ZZj,TT = self.computespiketrace(self.Nj,spikes,tauzj,mintime = mintime,maxtime = maxtime,
                                        timestep = timestep)
        return ZZj,TT
    

    def computePPi(self,spikes,tauzi,taup,maxfq = 200,mintime = 0,maxtime = None,timestep = 0.0010,
                   sample = 1) :
        ZZi,TT = self.computespiketrace(self.Ni,spikes,tauzi,maxfq = maxfq,mintime = mintime,
                                        maxtime = maxtime,timestep = timestep)

        PPi,TT = self.computetracetrace1(ZZi,taup,sample = sample)

        return ZZi,PPi,TT


    def computePPj(self,spikes,tauzj,taup,maxfq = 200,mintime = 0,maxtime = None,timestep = 0.0010,
                   sample = 1) :
        ZZj,TT = self.computespiketrace(self.Nj,spikes,tauzj,maxfq = maxfq,mintime = mintime,
                                        maxtime = maxtime,timestep = timestep)

        PPj,TT = self.computetracetrace1(ZZj,taup,sample = sample)
        return ZZj,PPj,TT

    def computeBBWW(self,prspikes,pospikes,tauzi,tauzj,taup,maxfq = 200,mintime = 0,maxtime = None,
                    timestep = 0.0010,sample = 1) :
        if mintime==None : mintime = min(prspikes[-1,0],pospikes[-1,0])
        if maxtime==None : maxtime = max(prspikes[-1,0],pospikes[-1,0])
        ZZi,PPi,TT = self.computePPi(prspikes,tauzi,taup,maxfq = maxfq,mintime = mintime,maxtime = maxtime,
                                     timestep = timestep,sample = sample)
        ZZj,PPj,TT = self.computePPj(pospikes,tauzj,taup,maxfq = maxfq,mintime = mintime,maxtime = maxtime,
                                     timestep = timestep,sample = sample)

        PPij,TT = self.computetracetrace2(ZZi,ZZj,taup,sample = sample)

        PPii = []
        for Pi in PPi :
            PPii.append(np.repeat([Pi],self.Nj,0).T)
        PPii = np.array(PPii)
        
        PPjj = []
        for Pj in PPj :
            PPjj.append(np.repeat([Pj],self.Ni,0))
        PPjj = np.array(PPjj)

        BBj = []
        for Pj in PPj :
            BBj.append(np.log(Pj+1e-4))

        BBj = np.array(BBj)

        WWij = []
        for Pij,Pii,Pjj in zip(PPij,PPii,PPjj) :
            Wij = np.log(np.divide(Pij+1e-8,np.multiply(Pii+1e-4,Pjj+1e-4)))
            WWij.append(Wij)

        WWij = np.array(WWij)            

        return ZZi,ZZj,PPi,PPj,PPij,BBj,WWij,TT

def test1(mintime = 0,maxtime = 1) :

    Ni = 1000
    Nj = 10
    timestep = 0.001
    tauzi = 0.010
    tauzj = 0.010
    taup = 1.100

    bcpnn1 = BcpnnConnection(Ni,Nj)

    prspikes = np.loadtxt("prspike.txt")
    pospikes = np.loadtxt("pospike.txt")

    maxtime = np.max(prspikes[-1,0],pospikes[-1,0])

    ZZi,TT = bcpnn1.computespiketrace(Ni,prspikes,tauzi,mintime = 0,maxtime = maxtime,timestep = timestep)
    PPi,TT = bcpnn1.computetracetrace1(ZZi,taup)
    ZZj,TT = bcpnn1.computespiketrace(Nj,pospikes,tauzj,mintime = 0,maxtime = maxtime,timestep = timestep)
    PPj,TT = bcpnn1.computetracetrace1(ZZj,taup)

    PPij,TT = bcpnn1.computetracetrace2(ZZi,ZZj,taup)

    return ZZi,ZZj,PPi,PPj,TT


def test2(tau,maxfq = 200) :

    ffq = []
    yymax = []
    
    for fq in range(1,501,10) :

        f = min(fq,maxfq)

        ymax = 1/(1 - math.exp(-1./f/tau))

        ffq.append(fq)

        yymax.append(ymax)

    return np.array(ffq),np.array(yymax)/(tau*maxfq) # yymax/tau = firing frequency [1/s]


import matplotlib.pyplot as plt

def plot1(fig = 1,files = True) :

    Ni = 100
    Nj = 25
    ri = 99
    rj = 17
    tauzi = 0.025
    tauzj = 0.010
    taup = 0.2

    invbwgain = 1/1e-4
    t1 = 4
    t2 = 6

    bcpnn1 = BcpnnConnection(Ni,Nj)

    figh = plt.figure(fig)
    plt.clf()
    figh.subplots_adjust(hspace=.5)
    
    pospikes = np.loadtxt("bcpnn.pospikes") # If >1 core: sort -g -m sim_bcpnn.*.prspikes > bcpnn.prspikes
    prspikes = np.loadtxt("bcpnn.prspikes") # If >1 core: sort -g -m sim_bcpnn.*.pospikes > bcpnn.pospikes

    pridx = np.where(prspikes[:,1]<=5)[0]
    poidx = np.where(pospikes[:,1]<=5)[0]

    ax1 = plt.subplot(5,2,1)
    ax1.plot(prspikes[pridx,0],prspikes[pridx,1],'.')
    ax1.set_xlim(t1,t2)
    ax1.set_ylim(5.5,-0.5)
    ax1.set_title("pre-spikes[0,5]")

    ax2 = plt.subplot(5,2,2)
    ax2.plot(pospikes[poidx,0],pospikes[poidx,1],'.')
    ax2.set_xlim(t1,t2)
    ax2.set_ylim(5.5,-0.5)
    ax2.set_title('post-spikes[0,5]')

    ZZi,TT = bcpnn1.computespiketrace(Ni,prspikes,tauzi,mintime=0,maxtime=10.0,timestep = 0.0001)    
    ax3 = plt.subplot(5,2,3)
    ax3.plot(TT,ZZi[:,ri])
    ax3.set_xlim(t1,t2)
    ax3.set_title('zi-trace in [0,1]')
    if files :
        zi = np.loadtxt("sim_bcpnn.0.zi")
        ax3.plot(zi[:,0],zi[:,1])
    
    ZZj,TT = bcpnn1.computespiketrace(Nj,pospikes,tauzj,mintime=0,maxtime=10.0,timestep = 0.0001)    
    ax4 = plt.subplot(5,2,4)
    ax4.plot(TT,ZZj[:,rj])
    ax4.set_xlim(t1,t2)
    ax4.set_title('zj-trace in [0,1]')
    if files :
        zj = np.loadtxt("sim_bcpnn.0.zj")
        ax4.plot(zj[:,0],zj[:,1])

    PPi,TT = bcpnn1.computetracetrace1(ZZi,taup,timestep = 0.0001)
    ax5 = plt.subplot(5,2,5)
    ax5.plot(TT,PPi[:,rj])
    ax5.set_xlim(t1,t2)
    ax5.set_title('pi-trace in [0,1]')
    if files :
        pi = np.loadtxt("sim_bcpnn.0.pi")
        ax5.plot(pi[:,0],pi[:,1])

    PPj,TT = bcpnn1.computetracetrace1(ZZj,taup,timestep = 0.0001)
    ax6 = plt.subplot(5,2,6)
    ax6.plot(TT,PPj[:,rj])
    ax6.set_xlim(t1,t2)
    ax6.set_title('pj-trace in [0,1]')
    if files :
        pj = np.loadtxt("sim_bcpnn.0.pj")
        ax6.plot(pj[:,0],pj[:,1])

    ZZi,ZZj,PPi,PPj,PPij,BBj,WWij,TT = bcpnn1.computeBBWW(prspikes,pospikes,tauzi,tauzj,taup,timestep = 0.001,
                                                          sample = 20)    
    ax7 = plt.subplot(5,2,7)
    ax7.plot(TT,PPij[:,ri,rj])
    ax7.set_xlim(t1,t2)
    ax7.set_title('pij-trace in [0,1]')
    if files :
        pij = np.loadtxt("sim_bcpnn.0.pij")
        ax7.plot(pij[:,0],pij[:,1])

    ax9 = plt.subplot(5,2,9)
    ax9.plot(TT,WWij[:,ri,rj])
    ax9.set_xlim(t1,t2)
    ax9.set_ylim(-5,5)
    ax9.set_title('wij-trace')
    if files :
        wij = np.loadtxt("sim_bcpnn.0.wij")
        ax9.plot(wij[:,0],wij[:,1]*invbwgain)

    ax10 = plt.subplot(5,2,10)
    ax10.plot(TT,BBj[:,rj])
    ax10.set_xlim(t1,t2)
    ax10.set_ylim(-5,0)
    ax10.set_title('bj-trace in [0,...]')
    if files :
        bj = np.loadtxt("sim_bcpnn.0.bj")
        ax10.plot(bj[:,0],bj[:,1]*invbwgain)

    ax9.set_xlabel('time (s)')
    ax10.set_xlabel('time (s)')
    
