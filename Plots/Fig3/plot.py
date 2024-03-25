"""Figure 3"""
__author__ = 'Eric R. Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, LogLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['legend.loc'] = "upper center"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(17,6))

################## Instanton plot ##################
Nx = 65         # Number of space grid points
N = 6400        # Number of time segments
length = 1      # Length of the simulation box
T = 15          # Propagation time

# Import field configurations at the fixed points and at the instanton
minR  = np.genfromtxt("minR_opt.txt")
minP  = np.genfromtxt("minP_opt.txt")
TS    = np.genfromtxt("TS_opt.txt")
instx = np.genfromtxt("instanton.txt")
instp = np.concatenate([np.concatenate([minR.reshape(1,Nx),instx.reshape(-1,Nx)],axis=0),minP.reshape(1,Nx)],axis=0)

# Plot of the instanton trajectory in field space
xplot = np.linspace(0.0,length,Nx)
tplot = np.linspace(0.0,T,N+1)
xplot = np.linspace(0.0,length,Nx)
TIME, X = np.meshgrid(tplot,xplot)

cm = ax1.contourf(TIME,X,instp.T,levels=20)
cbar = fig.colorbar(cm,ticks=[-0.5,0,0.5,1])
cbar.ax.set_yticklabels(['-0.5', '0', '0.5', '1'])

# Indicate location of TS
minarg = 2661   # Index of the instanton bead at which the TS is reached 
ax1.plot(np.array([tplot[minarg],tplot[minarg]]),np.array([-1000,1000]),'--',lw=3.0,color='k')

ax1.set_xlabel("$t$",fontsize=24)
ax1.set_ylabel("$x$",size=24,labelpad=10)
ax1.set_xlim(np.min(tplot),np.max(tplot))
ax1.set_ylim(np.min(xplot),np.max(xplot))

ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.xaxis.set_minor_locator(MultipleLocator(2.5))
ax1.yaxis.set_major_locator(MultipleLocator(0.25))
ax1.yaxis.set_minor_locator(MultipleLocator(0.125))

################## Rate plot ##################
# Import instanton rates for three different driving strenghts
k00 = np.genfromtxt("rate00.txt")
h00 = k00[:,0]      # Field strength h
k00 = k00[:,1]

k25 = np.genfromtxt("rate25.txt")
h25 = k25[:,0]      # Field strength h
k25 = k25[:,1]

k50 = np.genfromtxt("rate50.txt")
h50 = k50[:,0]      # Field strength h
k50 = k50[:,1]

### Plot rates as function of the applied field ###
ax2.plot(h00, k00, '-o', color='C0', label='$\\nu = 0$'   , lw=3.0, zorder=2, markersize=12)	
ax2.plot(h25, k25, '-X', color='C1', label='$\\nu = 0.25$', lw=3.0, zorder=2, markersize=12)	
ax2.plot(h50, k50, '-P', color='C2', label='$\\nu = 0.5$' , lw=3.0, zorder=2, markersize=12)	

ax2.set_xlabel("$h$",fontsize=24,labelpad=8)
ax2.set_ylabel("$k$",fontsize=24,labelpad=12)
ax2.set_yscale('log')

ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

ax2.set_xlim(-0.204,0.204)
ax2.set_ylim(1e-40,1e+5)
ax2.legend(ncol=3)

### Inset: Rate as function of ΔS ###
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['xtick.minor.size'] = 6
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['xtick.minor.width'] = 1
matplotlib.rcParams['xtick.direction'] = "inout"
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['ytick.minor.size'] = 6
matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['ytick.minor.width'] = 1
matplotlib.rcParams['ytick.direction'] = "inout"
matplotlib.rcParams['xtick.major.pad'] = '5'
matplotlib.rcParams['ytick.major.pad'] = '5'


# Import actions of forward and backward instantons
action00 = np.genfromtxt("action00.txt")
action25 = np.genfromtxt("action25.txt")
action50 = np.genfromtxt("action50.txt")

# Action difference
ΔS00 = action00[:,1] - action00[:,2]
ΔS25 = action25[:,1] - action25[:,2]
ΔS50 = action50[:,1] - action50[:,2]

# Plotting
axins = inset_axes(ax2, width="100%", height="100%", loc='lower right',bbox_to_anchor=(0.58,0.14,.4,.4), bbox_transform=ax2.transAxes)

axins.spines['bottom'].set_color('dimgray')
axins.spines['top'].set_color('dimgray')
axins.spines['right'].set_color('dimgray')
axins.spines['left'].set_color('dimgray')

axins.plot(ΔS00, k00, '-o', color='C0', label='$\\nu = 0$'   , lw=2.0, zorder=3, markersize=10)	
axins.plot(ΔS25, k25, '-X', color='C1', label='$\\nu = 0.25$', lw=2.0, zorder=2, markersize=10)	
axins.plot(ΔS50, k50, '-P', color='C2', label='$\\nu = 0.5$' , lw=2.0, zorder=2, markersize=10)	

axins.set_xlabel("$\Delta S$",fontsize=16)
axins.set_ylabel("$k$",fontsize=16)
axins.set_xlim(-0.71,0.13)
axins.set_ylim(1e-24,1e+1)
axins.xaxis.set_major_locator(MultipleLocator(0.3))
axins.xaxis.set_minor_locator(MultipleLocator(0.15))
axins.set_yscale("log")

################## Subplot labels ##################
ax1.text(1.15,0.92,'(a)',transform=ax1.transAxes,fontsize=24)
ax2.text(0.92,0.92,'(b)',transform=ax2.transAxes,fontsize=24)

plt.tight_layout()
fig.subplots_adjust(wspace=0.4)
#plt.savefig("Nucleation.pdf",format='pdf')
plt.show()
