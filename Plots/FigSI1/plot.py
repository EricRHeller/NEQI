"""Fig. SI.1"""
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
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

# Parameters
dof = 2     # Number of degrees of freedom
T   = 20    # Propagation time
N   = 800   # Number of segments
tau = T/N   # Time step

# Import instanton trajectory
instp = np.genfromtxt("instanton.txt").reshape(-1,dof)

##### Calculate cumulative path length ###
def pathlength(x):
    dpl = np.sqrt(np.sum(np.diff(x,axis=0)**2, 1))
    pl = [0] + list(np.cumsum(dpl))
    return np.array(pl)

tpath = np.linspace(0.0,T,N+1)
rpath = pathlength(instp)

### Plot path length ###
fig,ax = plt.subplots(figsize=(8,6))
plt.plot(tpath,rpath,lw=3.0,color='C0',zorder=3)

plt.xlim(np.min(tpath),np.max(tpath))
ax.set_xticks([0.0,tpath[415]])
ax.set_yticks([0.0,rpath[415],rpath[-1]])
ax.set_xticklabels(["$0$","$t_\mathrm{f}$"])
ax.set_yticklabels(["$x_\mathrm{R}$","$x^\ddagger$","$x_\mathrm{P}$"])

plt.xlabel("Time",size=24)
plt.ylabel("Path length",size=24)

### Plot time-translated intantons ###
Ni = 50
Na = 300
rpath = pathlength(instp[Ni:Na+1])
tpath = np.linspace(0.0,tau*(Na-Ni),Na-Ni+1)

plt.plot(tpath+3.5,rpath,'--',lw=3.0,color='C3',zorder=3)
plt.plot(tpath-1.25,rpath,'--',lw=3.0,color='C3',zorder=3)

plt.tight_layout()

#plt.savefig("pathlength.eps",format='eps')
plt.show()

