"""Figure 1"""
__author__ = 'Eric R. Heller'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

### System specification ###
A    = 2.0                                                      # Parameter of potential
B    = 1.8                                                      # Parameter of potential
dof  = 2                                                        # Number of degrees of freedom

### Definition of the potential 
def potential(x):                                               
    V = A/B**4*((x[0]+x[1])**4 + B**4 - 2*(x[0]+x[1])**2*B**2) + (x[0]-x[1])**2
    return V

minR = np.array([-0.9,-0.9])
minP = np.array([ 0.9, 0.9])
TS   = np.array([ 0.0, 0.0])

### Plotting ###
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
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.minor.width'] = 2
matplotlib.rcParams['ytick.direction'] = "inout"
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['legend.loc'] = "lower right"
matplotlib.rcParams['xtick.major.pad'] = '8'
matplotlib.rcParams['ytick.major.pad'] = '8'

### Plot instanton on top of potential and vector field ###
### Vector field ###
fig,ax = plt.subplots(figsize=(8.5,6))
xplot = np.linspace(-1.2,1.2,101)
yplot = np.linspace(-1.2,1.2,101)
X, Y = np.meshgrid(xplot,yplot)
FX = -A/B**4*(4*(X+Y)**3 - 4*B**2*(X+Y)) - 2*(X-Y) 
FY = -A/B**4*(4*(X+Y)**3 - 4*B**2*(X+Y)) + 2*(X-Y) 
ax.set_xlabel("$x_1$",fontsize=24)
ax.set_ylabel("$x_2$",fontsize=24)
qr = ax.streamplot(X,Y,FX,FY,density=1.5,color='C0',linewidth=1.5)

Z = np.zeros((len(xplot),len(yplot)))
for indx, xi in enumerate(xplot):
    for indy, yi in enumerate(yplot):
        Z[indx,indy] = potential(np.array([xi,yi]))
cm = ax.contourf(X,Y,Z,levels=20)
fig.colorbar(cm)

plt.xlim(np.min(xplot),np.max(xplot))
plt.ylim(np.min(yplot),np.max(yplot))

### Forward paths for various driving strenghts ν ###
x_0 = np.genfromtxt("x_0.txt")
x_0 = np.concatenate([np.concatenate([minR.reshape(1,dof),x_0.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)
x_2 = np.genfromtxt("x_2.txt")
x_2 = np.concatenate([np.concatenate([minR.reshape(1,dof),x_2.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)
x_4 = np.genfromtxt("x_4.txt")
x_4 = np.concatenate([np.concatenate([minR.reshape(1,dof),x_4.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)
x_6 = np.genfromtxt("x_6.txt")
x_6 = np.concatenate([np.concatenate([minR.reshape(1,dof),x_6.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)

### Backward paths for various driving strenghts ν ###
xb_2 = np.genfromtxt("xb_2.txt")
xb_2 = np.concatenate([np.concatenate([minR.reshape(1,dof),xb_2.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)
xb_4 = np.genfromtxt("xb_4.txt")
xb_4 = np.concatenate([np.concatenate([minR.reshape(1,dof),xb_4.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)
xb_6 = np.genfromtxt("xb_6.txt")
xb_6 = np.concatenate([np.concatenate([minR.reshape(1,dof),xb_6.reshape(-1,dof)],axis=0),minP.reshape(1,dof)],axis=0)

## Plot fixed points of the force ###
plt.scatter(minR[0],minR[1],s=50,color='k',zorder=7,label='Minima')
plt.scatter(minP[0],minP[1],s=50,color='k',zorder=7)
plt.scatter(TS[0],TS[1],s=50,color='k',marker='X',zorder=7,label='TS')

## Plot paths ###
plt.plot(x_0[:,0], x_0[:,1], "--", lw=2.5, color='k' , zorder=6, label='$\\nu=0$')
plt.plot(x_2[:,0], x_2[:,1], '-' , lw=2.5, color='C1', zorder=5, label='$\\nu=2$')
plt.plot(x_4[:,0], x_4[:,1], "-" , lw=2.5, color='C2', zorder=5, label='$\\nu=4$')
plt.plot(x_6[:,0], x_6[:,1], "-" , lw=2.5, color='C3', zorder=5, label='$\\nu=6$')

plt.plot(xb_2[:,0], xb_2[:,1], "-", lw=2.5, color='C1', zorder=4, alpha=0.5)
plt.plot(xb_4[:,0], xb_4[:,1], "-", lw=2.5, color='C2', zorder=4, alpha=0.5)
plt.plot(xb_6[:,0], xb_6[:,1], "-", lw=2.5, color='C3', zorder=4, alpha=0.5)

### Arrows ###
plt.arrow(x=-0.633,y=-0.247, dx=0.001,dy=0.0008,width=0.0,head_width=0.08, head_length=0.10, shape='full', overhang=0.3,color='k',zorder=8)
plt.arrow(x=0.633,y=0.247, dx=-0.001,dy=-0.0008,width=0.0,head_width=0.08, head_length=0.10, shape='full', overhang=0.3,color='k',zorder=8,alpha=0.5)

### Axis ticks ###
ax.xaxis.set_major_locator(MultipleLocator(0.6))
ax.xaxis.set_minor_locator(MultipleLocator(0.3))
ax.yaxis.set_major_locator(MultipleLocator(0.6))
ax.yaxis.set_minor_locator(MultipleLocator(0.3))

plt.legend(ncol=2)

plt.tight_layout()
#plt.savefig("Pathways.pdf",format='pdf')

plt.show()
