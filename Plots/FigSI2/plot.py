"""Fig. SI.3"""
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


# Import rates for different diffusivities
D025 = np.genfromtxt("Rate025.txt")
bias = D025[:,0]    # Parameter b
D025 = D025[:,1:]
D05  = np.genfromtxt("Rate05.txt")[:,1:]
D1   = np.genfromtxt("Rate1.txt")[:,1:]  
D2   = np.genfromtxt("Rate2.txt")[:,1:]
D4   = np.genfromtxt("Rate4.txt")[:,1:]

fig,ax = plt.subplots(figsize=(8,6))

### Kramers–Langer rates ###
ax.plot(bias, D025[:,0], color='C0', label='$D_{11} = \\tfrac{1}{4}$', lw=3.0, zorder=2)	
ax.plot(bias, D05[:,0] , color='C1', label='$D_{11} = \\tfrac{1}{2}$', lw=3.0, zorder=2)	
ax.plot(bias, D1[:,0]  , color='C2', label='$D_{11} = 1$'            , lw=3.0, zorder=2)	
ax.plot(bias, D2[:,0]  , color='C3', label='$D_{11} = 2$'            , lw=3.0, zorder=2)	
ax.plot(bias, D4[:,0]  , color='C4', label='$D_{11} = 4$'            , lw=3.0, zorder=2)	

### Instanton rates ###
ax.scatter(bias, D025[:,1], color='C0', s=30, zorder=3)	
ax.scatter(bias, D05[:,1] , color='C1', s=30, zorder=3)	
ax.scatter(bias, D1[:,1]  , color='C2', s=30, zorder=3)	
ax.scatter(bias, D2[:,1]  , color='C3', s=30, zorder=3)	
ax.scatter(bias, D4[:,1]  , color='C4', s=25, zorder=3)	

ax.set_xlabel("$b$",fontsize=24)
ax.set_ylabel("$k$",fontsize=24)

ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.125))

### Text box ###
text = plt.text(0.35,0.85,r'$\mathbf{D} = \begin{pmatrix} 1 & 0 \\  0 & D_{11}   \end{pmatrix}$',transform=ax.transAxes,fontsize=18)
text.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='k'))

plt.xlim(np.min(bias),1.003*np.max(bias))
plt.legend()
plt.tight_layout()
#plt.savefig("KLTrates.pdf",format='pdf')
plt.show()
