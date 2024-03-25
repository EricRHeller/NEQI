"""Fig. SI.4"""
__author__ = 'Eric R. Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
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
matplotlib.rcParams['legend.loc'] = "upper center"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

################## Instanton plot ##################
Nx = 65         # Number of space grid points
N = 6400        # Number of time segments
length = 1      # Length of the simulation box
T = 30          # Propagation time

# Import field configurations at the fixed points and at the instanton
minR  = np.genfromtxt("minR_opt.txt")
minP  = np.genfromtxt("minP_opt.txt")
TS    = np.genfromtxt("TS_opt.txt")
instx = np.genfromtxt("instanton.txt")
instp = np.concatenate([np.concatenate([minR.reshape(1,Nx),instx.reshape(-1,Nx)],axis=0),minP.reshape(1,Nx)],axis=0)

fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,width_ratios=[1,1.5],figsize=(8.5,6))

############# Kramers pricture #############
xplot = np.linspace(0.0,length,Nx)
ax1.plot(minR, xplot,'-' , label='Reactant', lw=3.0)
ax1.plot(minP, xplot,'-' , label='Product' , lw=3.0)
ax1.plot(TS  , xplot,'--', label='TS'      , lw=3.0)

ax1.set_yticks([0.0,0.5,1.0])
ax1.set_yticklabels([0.0,0.5,1.0])

ax1.set_ylabel("$x$",size=24)
ax1.set_xlabel("$\phi(x)$",size=24)

ax1.set_ylim(np.min(xplot),np.max(xplot))
ax1.legend()

############# Instanton pricture #############
ax2.set_yticks([0.0,0.5,1.0])
ax2.set_yticklabels([])

tplot = np.linspace(0.0,T,N+1)
TIME, X = np.meshgrid(tplot,xplot)
ax2.set_xlabel("$t$",fontsize=24)
cm = ax2.contourf(TIME,X,instp.T,levels=10)
cbar = fig.colorbar(cm,ticks=[-1,0,1])
cbar.ax.set_yticklabels(['-1', '0', '1'])

# Indicate location of TS
minarg = 2418   # Index of the instanton bead at which the TS is reached 
ax2.plot(np.array([tplot[minarg],tplot[minarg]]),np.array([-1000,1000]),'--',lw=3.0,color='k')

ax2.set_xlim(np.min(tplot),np.max(tplot))
ax2.set_ylim(np.min(xplot),np.max(xplot))

plt.tight_layout()
fig.subplots_adjust(wspace=0.15)

#plt.savefig("InstFieldSI.pdf",format='pdf')
plt.show()
