"""Fig. SI.5"""
__author__ = 'Eric R. Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['text.latex.preamble']= r'\usepackage{{amsmath}}' 
matplotlib.rcParams['axes.linewidth'] = 3.0
matplotlib.rcParams['xtick.major.size'] = 15
matplotlib.rcParams['xtick.minor.size'] = 12
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.minor.width'] = 2
matplotlib.rcParams['xtick.direction'] = "inout"
matplotlib.rcParams['ytick.major.size'] = 15
matplotlib.rcParams['ytick.minor.size'] = 12
matplotlib.rcParams['xtick.labelsize'] = 24
matplotlib.rcParams['ytick.labelsize'] = 24
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.width'] = 2
matplotlib.rcParams['ytick.direction'] = "inout"
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

# Import sizes of critical nuclei for three different driving strengths
n00 = np.genfromtxt("Nuclei00.txt")
h   = n00[:,0]  # Strength of applied field, h 
n00 = n00[:,1]
n25 = np.genfromtxt("Nuclei25.txt")[:,1]
n50 = np.genfromtxt("Nuclei50.txt")[:,1]

fig,ax = plt.subplots(figsize=(8,6))

### Plotting ###
ax.plot(h, n00, '-o', color='C0', label='$\\nu = 0$'   , lw=3.0, zorder=2)	
ax.plot(h, n25, '-o', color='C1', label='$\\nu = 0.25$', lw=3.0, zorder=2)	
ax.plot(h, n50, '-o', color='C2', label='$\\nu = 0.5$' , lw=3.0, zorder=2)	

ax.set_xlabel("$h$",fontsize=24,labelpad=8)
ax.set_ylabel("Critical nucleus",fontsize=24,labelpad=12)

ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

plt.xlim(1.02*np.min(h),1.02*np.max(h))
plt.legend()
plt.tight_layout()
#plt.savefig("CriticalNuclei.pdf",format='pdf')
plt.show()
