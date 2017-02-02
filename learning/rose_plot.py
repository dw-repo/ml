# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 04:49:40 2016

@author: Darbinyan
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('t0004_SF017677_CustomPeriod_%JFD_Wave_Hs_Mdir_1.csv',
                      dtype=float, delimiter=',',
                      skip_header=2, skip_footer=2)
datars=np.delete(data,([0,1,10,11,12]), axis=1)
N = 8

theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
ax = plt.subplot(111, projection='polar')

for i in range(N):
    indsort=datars[:,i].argsort()
    radii=datars[indsort,i]
    
    print([i, datars[indsort,i]])

#theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
#radii = 10 * np.random.rand(N)
##radii = [20,20,20,20,20,20,np.pi,np.pi]
#width = radii.min() * np.pi / 8 #np.pi / 4 * np.random.rand(N)
#
#ax = plt.subplot(111, projection='polar')
## bars = ax.bar(theta-np.pi/16, radii, width=width, bottom=width)
#bars = ax.bar(theta-width/2, radii, width=width, bottom=radii.min())
#ax.set_xticklabels(['90$^\circ$','45$^\circ$','0$^\circ$',
#              '315$^\circ$','270$^\circ$','225$^\circ$',
#              '180$^\circ$','135$^\circ$'])
## Use custom colors and opacity
#for r, bar in zip(radii, bars):
#    bar.set_facecolor(plt.cm.jet(r / 10.))
#    bar.set_alpha(0.5)

plt.show()