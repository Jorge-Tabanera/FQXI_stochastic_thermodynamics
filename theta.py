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



dt = 0.000051 
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

# mu_R = - 0.0025


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


As = np.linspace(0.0006/g0, 0.004/g0, 30)
mus = [0.001]#np.linspace(-0.001, 0.001, 30)

#%%

def ene(A, mu_L):
    
    n = 1
    de0 = []
    X = []
    
    mu_R = -g0*A
    
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

        X.append(x)
        de0.append(n/x)
        
    return X, de0
    
#%%

@jit
def theta_serie(A, mu_L):
    
    n = 1
    de0 = []
    
    mu_R = -g0*A
    
    for it in range(int(iteraciones)):
        
        x = A * np.cos( Omega * it * dt )
        v = -Omega* A * np.sin( Omega * it * dt )
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

        dth =  - Omega - 1 / (1 + v**2 /(Omega*x) **2 ) *  g0/m * n/(Omega*x) 
        # dth = 1/ (1 + ( v / (Omega*x) )**2 ) * ( force1(x,v,n)/(m*Omega*x) - v**2 / (Omega*x**2))
        
        de0.append(dth * dt)
        
    return de0

@jit
def theta_serie(A, mu_L):
    
    n = 1
    de0 = []
    
    mu_R = -g0*A
    
    for it in range(int(iteraciones)):
        
        x = A * np.cos( Omega * it * dt )
        v = -Omega* A * np.sin( Omega * it * dt )
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

        dth =  1 / (1 + v**2 /(Omega*x) **2 )**2 *  g0**2/m**2 * n**2/(Omega**2*x**2) 
        
        de0.append(dth * dt)
        
    return de0

@jit
def conteo(ds):
    
    ds2 = 0
    
    for k in range(len(ds)):
            # print(k)
            
            for l in range(len(ds)):
                
                ds2 = ds2 + ds[k]*ds[l]
    return ds2 



vares = []

for i in range(len(As)):
    for j in range(len(mus)):
        
        print(i)
        
        ds = theta_serie(As[i], mus[j])
            
    # vares.append(conteo(ds) -  sum(ds)**2)
    vares.append(sum(ds) -  sum(ds)**2)
    
    print(np.nansum(ds))       
            
print(time() - start)        
   
#%%
py.plot( As,vares , '.-')
        
        
        
    