#!/home/evgeniia/anaconda3/bin/python3

#python3 << EOF

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc


rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
rc('text.latex', preamble=r"\usepackage[english]{babel}")
rc('text.latex', preamble=r"\usepackage[T2A]{fontenc}")
plt.rcParams.update({'font.size': 15})


gs = gridspec.GridSpec(1, 1)
def SubPlot(x, y, fig):
	return fig.add_subplot(gs[y, x])

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

WorkDir='./Results'

fig = plt.figure(figsize=(7, 5.5))
ax = SubPlot(0, 0, fig)

alphas, P = np.loadtxt(WorkDir+'/ResPower', skiprows=1, unpack=True,usecols=[0, 1])
	
ax.plot(alphas, P, color='black')

ax.set_xlabel(r'$\alpha$, degrees',fontsize=24)
ax.set_ylabel(r'$P$, GW', fontsize=24)

plt.savefig('./Fig.pdf')
plt.show()
plt.close(fig)


