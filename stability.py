#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:36:09 2021

@author: markov
"""

import numpy as np
import pylab as py
from time import time
from numba import jit

start = time()



dt = 0.0001 
sqrtdt = np.sqrt(dt)
m =  0.006242*22   #12e-3 #eV/nm**2/Ghz**2

# masa 0.006242*22

Q = 10000
Omega = 1.69 #GHz
gamma = Omega*m/Q #eV/nm2/GHz

k = 0.1248 #eV/nm OSCILLATOR CONSTANT
k = m*Omega**2

g0 = 1e-2 #eV/nm #1.8e6 #eV/m

# ESTOS DATOS FUNCIONAN, NO LOS TOQUES
# Gamma_L = 300
# Gamma_R = 20*Omega
# lambda_L = 20 #200*1e-1
# lambda_R = 0

Gamma_L = 400
Gamma_R = 15
lambda_L = 20
lambda_R = 0



epsilon_01 = 0#  g0_eV * 1e-11

mu_R = - 0.0025


Kb = 8.6e-5 #eV/K
T = 1e-13 #K de los lead

T_diff = 1e-13

D = T_diff#Kb*T #Diffusion coef


@jit
def fermi(epsilon, mu, T):
    
    return 1 /  ( np.exp((epsilon - mu)/(Kb*T)) + 1)



@jit
def force1(x1,v1,n1):
    
    return -k * x1 - gamma * v1 - g0 * n1



@jit
def epsilon_x1(x):
   
    return epsilon_01 + g0 * x


iteraciones = 2*np.pi / Omega / dt


As = np.linspace(0, 0.003/g0, 20)
mus = np.linspace(-0.001, 0.001, 30)

@jit
def energy(A, mu_L):
    
    n = 0
    de0 = []
    
    for it in range(int(iteraciones)):
        
        x = A * np.cos(Omega*it*dt)
        v = -Omega* A * np.sin(Omega*it*dt)
        epsilon1 = epsilon_x1(x)
        
        Gamma_out_R =  Gamma_R * np.exp(lambda_R*x) * \
            (1-fermi(epsilon1, mu_R, T ))
            
        Gamma_in_R = Gamma_R * np.exp(lambda_R*x) * \
            (fermi(epsilon1, mu_R, T ))
            
        Gamma_out_L =  Gamma_L * np.exp(lambda_L*x) * \
            (1-fermi(epsilon1, mu_L, T ))
            
        Gamma_out = Gamma_out_L + Gamma_out_R
        
        Gamma_in = Gamma_L * np.exp(lambda_L*x) * \
            (fermi(epsilon1, mu_L, T )) + Gamma_in_R
            
        n = n + ( Gamma_in * (1-n) - Gamma_out*n )*dt


        dE = -gamma* v **2 - g0* v *n
        de0.append(dE)
        
    return sum(de0)


Es = np.zeros([len(As), len(mus)])

for i in range(len(As)):
    for j in range(len(mus)):
        
        print([i,j])
        
        Es[i,j] = energy(As[i], mus[j])
        
print(time() - start)        
   
        
#%%
        
# # py.pcolormesh(mus, As[:16],Es[:16,:],shading= 'flat')
# py.contourf(mus, As[:],Es[:,:], levels = 30)

# py.colorbar()
# py.xlabel('mu_L (eV)', fontsize = 15)
# py.ylabel('Amplitude (nm)', fontsize = 15)


#%%   
sino = np.zeros([len(As), len(mus)])  

for i in range(len(sino)):
    for j in range(len(sino[0])):
        
        if Es[i][j] <= 0:
            sino[i][j] = -1

py.contourf(mus*1000, As,sino, levels = 1, cmap = 'Blues')

# xe = [str('{:.1e}'.format(mus[k])) for k in range(len(mus))]

# py.colorbar()
py.xticks(fontsize = 13)
py.yticks(fontsize = 13)   
py.xlabel('$Vs$ (mV)', fontsize = 15)
py.ylabel('Amplitude (nm)', fontsize = 15)

#%%   
# sino = np.zeros([len(As), len(mus)])  

# for i in range(len(sino)):
#     for j in range(len(sino[0])):
        
#         if Es[i][j] <= 0:
#             sino[i][j] = -1

# py.pcolormesh(mus*1000, As,sino, shading = 'gouraud', cmap = 'Blues')

# # xe = [str('{:.1e}'.format(mus[k])) for k in range(len(mus))]

# # py.colorbar()
# py.xticks(fontsize = 13)
# py.yticks(fontsize = 13)   
# py.xlabel('Source (meV)', fontsize = 15)
# py.ylabel('Amplitude (nm)', fontsize = 15)
      
      
#%%
# cmap = py.get_cmap('plasma')


# for k in range(3,10):
#     py.plot(As,Es[:,k], '.-', color = cmap( -1000 * mus[k] )\
#             , markersize = 4, label = '$V_{S} = $ ' + str('{:.1e}'.format(mus[k])) )
    
        
# py.legend(loc = 2)

# py.xticks(fontsize = 13)
# py.yticks(fontsize = 13)     
# py.xlabel('Amplitude (nm)', fontsize = 15) 
# py.ylabel('$\Delta E$', fontsize = 15)         
        
   
#%%

np.save('stability_new', Es)


        
        
        
    