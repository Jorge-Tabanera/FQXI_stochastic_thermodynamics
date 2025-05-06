#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:59:13 2022

@author: jorge
"""

import numpy as np
import pylab as py



I = np.loadtxt('/home/jorge/Documentos/BISTABILITY/DATA/I1_89.txt')

py.pcolor(I, cmap = 'inferno')
py.colorbar()