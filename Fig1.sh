#!/home/evgeniia/anaconda3/bin/python3

#python3 << EOF

import numpy as np
import os
import math
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as col
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib import rc
import pprint
from math import cos
import collections
from math import cos, sqrt, exp, pi
from scipy import integrate
import matplotlib.ticker as tick

font = {'family' : 'serif',
        'size'   : 24}
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
rc('text.latex', preamble=r"\usepackage[english]{babel}")
rc('text.latex', preamble=r"\usepackage[T2A]{fontenc}")
plt.rcParams.update({'font.size': 15})



#matplotlib.rc('xtick', labelsize=24) 
#matplotlib.rc('ytick', labelsize=24)

gs =gridspec.GridSpec(1,1)#,height_ratios=[0.1,1])
def SubPlot(x,y,fig):
	return fig.add_subplot(gs[y,x])

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

WorkDir='./Results'

fig = plt.figure(figsize=(7,5.5))
ax = SubPlot(0,0,fig)

alphas, P=np.loadtxt(WorkDir+'/ResPower', skiprows=1, unpack=True,usecols=[0,1])
	
ax.plot(alphas, P, color='black')

ax.set_xlabel(r'$\alpha$, degrees',fontsize=24)
ax.set_ylabel(r'$P$, GW', fontsize=24)	

#ax1.legend(loc=2, prop={'size': 18})

#ax.xaxis.set_major_locator(plt.MaxNLocator(8))
#tick_locator = ticker.MaxNLocator(nbins=5)
#formatter = ticker.ScalarFormatter ()
#formatter.set_powerlimits((-1, 3))
# Установка форматера для оси Y
#ax.yaxis.set_major_formatter (formatter)
#ax.yaxis.set_major_locator(tick_locator)


plt.savefig('./Fig.pdf')
plt.show()
plt.close(fig)
#plt.close(fig2)

