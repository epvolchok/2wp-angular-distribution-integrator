
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as col
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib import rc
import pprint
import collections
from math import pi


theta, phi, r = np.genfromtxt(r'./Scripts/Data/ResPowerAlpha', unpack=True, usecols=(1,3,11),skip_header=1)
d_theta=pi/400.;
d_phi=2.*pi/200.;


THETA, PHI = np.meshgrid(theta, phi)

x=r*np.cos(phi)*np.sin(theta)
y=r*np.sin(phi)*np.sin(theta)
z=r*np.cos(theta)

X=np.cos(PHI)*np.sin(THETA)
Y=np.sin(PHI)*np.sin(THETA)
Z=np.cos(THETA)



fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

print('plotting')

# Plot the surface.

plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False, alpha=1)                
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
