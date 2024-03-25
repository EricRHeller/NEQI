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

#System specification
dof = 2     # Number of degrees of freedom
#Potential
A = 2.0
B = 1.8
Dmat = np.array([[1.0,0.5],[0.5,4.0]])  # Diffusion matrix
μ = Dmat

# Potential 
def potential(x):
    V = A/B**4*((x[0]+x[1])**4 + B**4 - 2*(x[0]+x[1])**2*B**2) + (x[0]-x[1])**2 
    return V

# Hessian
def d2Vdx2(x):                                                  
    h = np.zeros((dof,dof))
    h[0,0] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) + 2    
    h[0,1] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) - 2   
    h[1,0] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) - 2   
    h[1,1] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) + 2  
    return h

# Mobility-scaled conservative force 
def force(x):    
    f = np.zeros(dof)
    f[0] = -A/B**4*(4*(x[0]+x[1])**3 - 4*B**2*(x[0]+x[1])) - 2*(x[0]-x[1]) 
    f[1] = -A/B**4*(4*(x[0]+x[1])**3 - 4*B**2*(x[0]+x[1])) + 2*(x[0]-x[1]) 
    f = np.matmul(μ,f)
    return f

### Plot vector field ###
fig,ax = plt.subplots(figsize=(8,6))

xplot = np.linspace(-1.8,1.8,101)
yplot = np.linspace(-1.8,1.8,101)
X, Y = np.meshgrid(yplot,xplot)
Fxrot = np.zeros(X.shape)
Fyrot = np.zeros(Y.shape)

# Plot force field
for i in range(len(xplot)):
        for j in range(len(yplot)):
            Fxrot[i,j], Fyrot[i,j] = force(np.array([X[i,j],Y[i,j]]))

ax.set_xlabel("$x_1$",fontsize=24)
ax.set_ylabel("$x_2$",fontsize=24)
qr = ax.streamplot(X,Y,Fxrot,Fyrot,density=1.5,color='C0',linewidth=1.5)

# Plot potential
Z = np.zeros((len(xplot),len(yplot)))
for indx, xi in enumerate(xplot):
    for indy, yi in enumerate(yplot):
        Z[indx,indy] = potential(np.array([xi,yi]))
cm = ax.contourf(X,Y,Z,levels=20)
cbar = fig.colorbar(cm, ticks=[0,6,12,18])

# Import and plot instanton ###
instp = np.genfromtxt("instanton.txt")
plt.plot(instp[:,0],instp[:,1],lw=4.0,color='g')

### Reaction coordinates ###
TS = np.array([0.0,0.0])    # Location of the TS

# Uplus 
hessu = np.matmul(d2Vdx2(TS),Dmat)
ed1, edv1 = np.linalg.eig(hessu)
ep1 = np.argmin(ed1) 
uplus = edv1[:,ep1]

# Vplus 
hessv = np.matmul(Dmat,d2Vdx2(TS))
ed2, edv2 = np.linalg.eig(hessv)
ep2 = np.argmin(ed2) 
vplus = edv2[:,ep2]

# Plot reaction coordinates
s = 0.5
u1 = TS-s*uplus
u2 = TS+s*uplus
plt.plot(np.array([u1[0],u2[0]]),np.array([u1[1],u2[1]]),'--',lw=4.0,color='k', label='$\mathsf{u}_{+}$',zorder=5)

v1 = TS-s*vplus
v2 = TS+s*vplus
plt.plot(np.array([v1[0],v2[0]]),np.array([v1[1],v2[1]]),'--',lw=4.0,color='darkgrey', label='$\mathsf{v}_{+}$',zorder=5)

# Instanton momentum at endpoint 
pvec = np.genfromtxt("momentum.txt") # Import momentum vector
pvec /= np.linalg.norm(pvec)
p1 = TS-s*pvec
p2 = TS+s*pvec
plt.plot(np.array([p1[0],p2[0]]),np.array([p1[1],p2[1]]),'-',lw=4.0,color='red', label='$\\bar{\mathsf{p}}_\mathrm{f}$')


ax.xaxis.set_major_locator(MultipleLocator(1.0))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

plt.xlim(np.min(xplot),np.max(xplot))
plt.ylim(np.min(yplot),np.max(yplot))
plt.legend()
plt.tight_layout()
#plt.savefig("Anisotropy.pdf",format='pdf')

plt.show()
