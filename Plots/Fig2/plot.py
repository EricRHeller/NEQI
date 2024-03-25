"Figure 2"
__author__ = 'Eric R. Heller'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator

# Import data files
DVR  = np.genfromtxt("DVRrates.txt")
instanton = np.genfromtxt("instrates.txt")

β = DVR[:,0]
ε = 1/β             # Noise strength
DVR[:,1:] /= 2.0    # DVR data file contains the sum of forward and backward rates. Because of the symmetry of the system, the two rates are identical. Hence, we can divide by two to get the forward rate.

# Rates
k_DVR  = DVR[:,1:]
k_inst = instanton[:,1:]

ν = np.linspace(0.0,6.0,25) # Considered driving strenghts 

# Normalize the rates for each ε by the DVR equilibrium rate (ν = 0)
for indc in range(DVR.shape[0]):
    k_inst[indc,:] /= k_DVR[indc,0]
    k_DVR[indc,:]  /= k_DVR[indc,0]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['xtick.major.size'] = 12
matplotlib.rcParams['xtick.minor.size'] = 9
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.minor.width'] = 1
matplotlib.rcParams['xtick.direction'] = "inout"
matplotlib.rcParams['ytick.major.size'] = 12
matplotlib.rcParams['ytick.minor.size'] = 9
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.minor.width'] = 1
matplotlib.rcParams['ytick.direction'] = "inout"
matplotlib.rcParams['legend.fontsize'] = 14
#matplotlib.rcParams['legend.loc'] = "lower left"
matplotlib.rcParams['xtick.major.pad'] = '6'
matplotlib.rcParams['ytick.major.pad'] = '6'

### Plotting ###
fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=2,figsize=(8.5,6.5))

# Forward
ax1[0].plot(ν, k_DVR[0,:] , '-' , color='k' , lw=2.5, zorder=2, label='DVR')	
ax1[0].plot(ν, k_inst[0,:], '--', color='C0', lw=2.5, zorder=2, label='NEQI')	

ax1[1].plot(ν, k_DVR[1,:] , '-' , color='k' , lw=2.5, zorder=2)	
ax1[1].plot(ν, k_inst[1,:], '--', color='C0', lw=2.5, zorder=2)	

ax2[0].plot(ν, k_DVR[2,:] , '-' , color='k' , lw=2.5, zorder=2)	
ax2[0].plot(ν, k_inst[2,:], '--', color='C0', lw=2.5, zorder=2)	

ax2[1].plot(ν, k_DVR[3,:] , '-' , color='k' , lw=2.5, zorder=2)	
ax2[1].plot(ν, k_inst[3,:], '--', color='C0', lw=2.5, zorder=2)	

ax2[0].set_xlabel('$\\nu$', size=20)
ax2[0].set_ylabel('$k(\\nu)/k_\mathrm{DVR}(\\nu=0)$', size=20, labelpad=8)
ax2[1].set_xlabel('$\\nu$', size=20)
ax1[0].set_ylabel('$k(\\nu)/k_\mathrm{DVR}(\\nu=0)$', size=20, labelpad=8)

ax1[0].set_xticklabels([])
ax1[1].set_xticklabels([])

#ax1[0].set_yscale('log')
ax1[1].set_yscale('log')
ax2[0].set_yscale('log')
ax2[1].set_yscale('log')

ax2[0].xaxis.set_major_locator(MultipleLocator(2.0))
ax2[1].xaxis.set_major_locator(MultipleLocator(2.0))
ax1[0].xaxis.set_major_locator(MultipleLocator(2.0))
ax1[1].xaxis.set_major_locator(MultipleLocator(2.0))

ax2[0].xaxis.set_minor_locator(MultipleLocator(1.0))
ax2[1].xaxis.set_minor_locator(MultipleLocator(1.0))
ax1[0].xaxis.set_minor_locator(MultipleLocator(1.0))
ax1[1].xaxis.set_minor_locator(MultipleLocator(1.0))

ax1[0].legend(ncol=1)

ax1[0].text(0.65,0.45,"$\epsilon=1$"  ,fontsize=20,transform=ax1[0].transAxes)
ax1[1].text(0.65,0.45,"$\epsilon=1/3$",fontsize=20,transform=ax1[1].transAxes)
ax2[0].text(0.65,0.45,"$\epsilon=1/5$",fontsize=20,transform=ax2[0].transAxes)
ax2[1].text(0.65,0.45,"$\epsilon=1/7$",fontsize=20,transform=ax2[1].transAxes)

ax1[0].set_xlim(np.min(ν),np.max(ν))
ax1[1].set_xlim(np.min(ν),np.max(ν))
ax2[0].set_xlim(np.min(ν),np.max(ν))
ax2[1].set_xlim(np.min(ν),np.max(ν))

rect10 = patches.Rectangle((0.62, 0.42), 0.235, 0.125, linewidth=1, edgecolor='grey', facecolor='none',transform=ax1[0].transAxes)
rect11 = patches.Rectangle((0.62, 0.42), 0.32 , 0.125, linewidth=1, edgecolor='grey', facecolor='none',transform=ax1[1].transAxes)
rect20 = patches.Rectangle((0.62, 0.42), 0.32 , 0.125, linewidth=1, edgecolor='grey', facecolor='none',transform=ax2[0].transAxes)
rect21 = patches.Rectangle((0.62, 0.42), 0.32 , 0.125, linewidth=1, edgecolor='grey', facecolor='none',transform=ax2[1].transAxes)
ax1[0].add_patch(rect10)
ax1[1].add_patch(rect11)
ax2[0].add_patch(rect20)
ax2[1].add_patch(rect21)

plt.tight_layout()

plt.subplots_adjust(wspace=0.3,hspace=0.1)

#plt.savefig("rates.pdf", format='pdf')
plt.show()

