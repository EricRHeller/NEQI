""" Nonequilibrium instanton optimization and rate calculation for the model in Figs. 1 and 2 """
__author__ = 'Eric R. Heller'
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

### System specification ###
A    = 2.0                                                      # Parameter of potential
B    = 1.8                                                      # Parameter of potential
dof  = 2                                                        # Number of degrees of freedom
ν    = 3.0                                                      # Driving strength
β    = 1.0                                                      # Inverse temperature
ε    = 1/β                                                      # Noise strength
Dmat = np.identity(dof)                                         # Isotropic diffusion
Dinv = np.linalg.inv(Dmat)
Ddet = np.linalg.det(Dmat)
μ    = ε*β*Dmat                                                 # Mobility

### Definitions of potential, force and path action ###
def potential(x):                                               # Potential generating the conservative force
    V = A/B**4*((x[0]+x[1])**4 + B**4 - 2*(x[0]+x[1])**2*B**2) + (x[0]-x[1])**2
    return V

def d2Vdx2(x):                                                  # Hessian
    h = np.zeros((dof,dof))
    h[0,0] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) + 2    
    h[0,1] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) - 2   
    h[1,0] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) - 2   
    h[1,1] = A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) + 2  
    return h

# Total force and its derivatives 
def force(x):    
    f = np.zeros(dof)
    f[0] = -A/B**4*(4*(x[0]+x[1])**3 - 4*B**2*(x[0]+x[1])) - 2*(x[0]-x[1]) + ν*((x[0]-x[1])**3 - (x[0]-x[1])) 
    f[1] = -A/B**4*(4*(x[0]+x[1])**3 - 4*B**2*(x[0]+x[1])) + 2*(x[0]-x[1]) + ν*((x[0]-x[1])**3 - (x[0]-x[1])) 
    f = np.matmul(μ,f)
    return f

def dforcedx(x):
    h = np.zeros((dof,dof))
    h[0,0] = -A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) -2 + ν*(3*(x[0]-x[1])**2 - 1)   
    h[0,1] = -A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) +2 - ν*(3*(x[0]-x[1])**2 - 1)  
    h[1,0] = -A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) +2 + ν*(3*(x[0]-x[1])**2 - 1)  
    h[1,1] = -A/B**4*(12*(x[0]+x[1])**2 - 4*B**2) -2 - ν*(3*(x[0]-x[1])**2 - 1)  
    h = np.matmul(μ,h)
    return h

def d2forcedx2(x):
    l = np.zeros((dof,dof,dof))
    l[0,0,0] = -A/B**4*24*(x[0]+x[1]) + ν*6*(x[0]-x[1])  
    l[0,0,1] = -A/B**4*24*(x[0]+x[1]) - ν*6*(x[0]-x[1])
    l[0,1,0] = -A/B**4*24*(x[0]+x[1]) - ν*6*(x[0]-x[1]) 
    l[0,1,1] = -A/B**4*24*(x[0]+x[1]) + ν*6*(x[0]-x[1]) 
    l[1,0,0] = -A/B**4*24*(x[0]+x[1]) + ν*6*(x[0]-x[1]) 
    l[1,0,1] = -A/B**4*24*(x[0]+x[1]) - ν*6*(x[0]-x[1])
    l[1,1,0] = -A/B**4*24*(x[0]+x[1]) - ν*6*(x[0]-x[1])
    l[1,1,1] = -A/B**4*24*(x[0]+x[1]) + ν*6*(x[0]-x[1]) 
    l = np.einsum('jm,mkl->jkl',μ,l)
    return l

# Path action and its derivatives 
def functionalx(x,xa,xb):                                       # xa and xb are the fixed ends, x is the vector of all intermediate beads
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    Darr = np.repeat(Dinv.reshape(1,dof,dof),len(xp),axis=0)
    pmat = np.zeros((len(xp),dof,1))
    for indn in range(len(xp)-1):
        pmat[indn,:,0] = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
    S = np.sum(np.matmul(tau/4*np.swapaxes(pmat,1,2),np.matmul(Darr, pmat)))
    return S 

def dSdx(x,xa,xb):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    Darr = np.repeat(Dinv.reshape(1,dof,dof),len(xp),axis=0)
    pmat = np.zeros((len(xp),dof,1))
    ppmat = np.zeros((len(xp),dof,dof))
    for indn in range(len(xp)-1):
        pmat[indn,:,0]  = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat[indn,:,:] = -np.diag(np.ones(dof))/tau - dforcedx(xp[indn])
    part1 = np.matmul(np.swapaxes(pmat,1,2),np.matmul(Darr, ppmat)) 
    part2 = np.matmul(np.swapaxes(pmat,1,2),Darr/tau)  

    dSdx = tau/2 * (part1[1:-1] + part2[:-2]) 
    return dSdx.ravel()

def d2Sdx2(x,xa,xb):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),np.concatenate([x,xb.reshape(1,dof)],axis=0)],axis=0)
    pmat   = np.zeros((len(xp),dof))
    ppmat  = np.zeros((len(xp),dof,dof))
    pppmat = np.zeros((len(xp),dof,dof,dof))
    for indn in range(len(xp)-1):
        pmat[indn]   = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat[indn]  = np.diag(np.ones(dof))/tau + dforcedx(xp[indn])
        pppmat[indn] = d2forcedx2(xp[indn])

    d2Sdx2 = np.zeros(2*(len(x), dof))
    for indn in range(1,len(x)+1):
        d2Sdx2[indn-1,:,indn-1,:] = 0.5*tau * (- np.einsum('ijk,i',pppmat[indn],np.matmul(Dinv,pmat[indn])) + np.matmul(ppmat[indn].T,np.matmul(Dinv,ppmat[indn])) + Dinv/tau**2) 
    for indn in range(1,len(x)):
        d2Sdx2[indn-1,:,indn,:] = -0.5*tau*np.matmul(ppmat[indn].T,Dinv/tau)  
        d2Sdx2[indn,:,indn-1,:] = -0.5*tau*np.matmul(Dinv/tau,ppmat[indn])
    d2Sdx2.shape = (x.size,)*2
    return d2Sdx2

# Gradient and Hessian of the action with flexible terminal end 
def dSdx_one(x,xa):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),x],axis=0)
    Darr = np.repeat(Dinv.reshape(1,dof,dof),len(xp),axis=0)
    pmat = np.zeros((len(xp),dof,1))
    ppmat = np.zeros((len(xp),dof,dof))
    for indn in range(len(xp)-1):
        pmat[indn,:,0]  = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat[indn,:,:] = -np.diag(np.ones(dof))/tau - dforcedx(xp[indn])
    part1 = np.matmul(np.swapaxes(pmat,1,2),np.matmul(Darr, ppmat)) 
    part2 = np.matmul(np.swapaxes(pmat,1,2),Darr/tau)  
    dSdx = tau/2 * (part1[1:] + part2[:-1]) 
    return dSdx.ravel()

def d2Sdx2_one(x,xa):
    x = x.reshape(-1,dof)
    xp = np.concatenate([xa.reshape(1,dof),x],axis=0)
    pmat   = np.zeros((len(xp),dof))
    ppmat  = np.zeros((len(xp),dof,dof))
    pppmat = np.zeros((len(xp),dof,dof,dof))
    for indn in range(len(xp)-1):
        pmat[indn]   = (xp[indn+1] - xp[indn])/tau - force(xp[indn])
        ppmat[indn]  = np.diag(np.ones(dof))/tau + dforcedx(xp[indn])
        pppmat[indn] = d2forcedx2(xp[indn])
   
    d2Sdx2 = np.zeros(2*(len(x), dof))
    for indn in range(1,len(x)+1):
        d2Sdx2[indn-1,:,indn-1,:] = 0.5*tau * (- np.einsum('ijk,i',pppmat[indn],np.matmul(Dinv,pmat[indn])) + np.matmul(ppmat[indn].T,np.matmul(Dinv,ppmat[indn])) + Dinv/tau**2) 
    for indn in range(1,len(x)):
        d2Sdx2[indn-1,:,indn,:] = -0.5*tau*np.matmul(ppmat[indn].T,Dinv/tau)  
        d2Sdx2[indn,:,indn-1,:] = -0.5*tau*np.matmul(Dinv/tau,ppmat[indn])
    d2Sdx2.shape = (x.size,)*2
    return d2Sdx2

### Locate fixed points of the force ###
print("### Locating reactant, product and TS fixed points of the total force ###")
# Reactant
guess = np.array([-1.0,-1.0])
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-12, method='hybr')
minR = sol.x
print("Reactant configuration",minR)

# Product
guess = np.array([1.0,1.0])
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-12, method='hybr')
minP = sol.x
print("Product configuration",minP)

# TS
guess = np.array([0.0,0.0])
sol = optimize.root(fun=force, x0=guess, jac=dforcedx, tol=1e-12, method='hybr')
TS = sol.x
print("TS",TS)

### Instanton optimization ###
# Initial guess for the instanton
def instguess(seg,start,end):                                   # Number of intermediate beads = N - 1
    y = np.linspace(start,end,seg-1)
    return y

print("### Creating initial guess for the instanton trajectory ###")
N = 200                                                         # Number of segments
T = 2.0                                                         # Propagation time
tau = T / N                                                     # Time step
endpoint = TS                                                   # Endpoint = TS for the activation path, endpoint = product fixed point (minP) for full transition path 
iguessx = instguess(N,minR,endpoint)                            # Initial guess 

### Functions for optimization ###
def S(x): # Action
    S = functionalx(x,minR,endpoint)
    return S

def Sgrad(x): 
    dS = dSdx(x,minR,endpoint)
    return dS

def Shess(x):
    d2S = d2Sdx2(x,minR,endpoint)
    return d2S

### Instanton optimization with one of scipy's minimization routines ###
print("### Instanton optimization ###")
sol = optimize.minimize(fun=S, x0=iguessx.ravel(), jac=Sgrad, hess=Shess, tol=1e-7, method='Newton-CG')
instx = sol.x.reshape(N-1,dof)                                  # Converged instanton
insto = np.concatenate([instx,endpoint.reshape(-1,dof)],axis=0) # Path including endpoint
instp = np.concatenate([minR.reshape(1,dof),insto],axis=0)      # Path including starting point and endpoint


### Reaction rate ###
print("### Rate calculation ###")
def instrate():
    action = S(instx)                                           # Instanton action
    gradO  = dSdx_one(insto,minR)                               # Gradient for path with flexible endpoint
    hessO  = d2Sdx2_one(insto,minR)                             # Hessian for path with flexible endpoint
    
    # Project out direction (reaction coordinate) along momentum at final bead
    pf = gradO[-dof:]                                           # Momentum vector at the endpoint
    momentum = np.linalg.norm(pf)                               # Norm of momentum vector at the endpoint
    pvec = np.zeros(insto.size)
    pvec[-dof:] = pf / momentum 
    Pmat = np.identity(insto.size) - np.outer(pvec,pvec)        # Projection matrix 
    hessP = np.matmul(Pmat.T,np.matmul(hessO,Pmat))             # Project out pf from d2Sdx2

    # Compute fluctuation factor
    eigI  = np.linalg.eigvalsh(2*tau*hessP)[2:]                 # Compute eigenvalues of fluctuation matrix and remove time-translation mode and reaction coordinate
    Sigma = np.sum(np.log(eigI)) + N*np.log(Ddet)               # Compute fluctuation determinant 
    Sigma = np.exp(Sigma) 

    Bt  = np.sum((instp[1:] - instp[:-1])**2) / tau             # Normalization factor from integral transformation
    pre = np.sqrt(Bt/(tau*momentum**2*Sigma)) / (4*np.pi*tau)   # Prefactor 
    k = pre * np.exp(-action/ε)                                 # Rate
    return k

# Kramers–Langer rate (only meaningful at equilibrium, ν = 0)
def KLT():
    barrier  = potential(TS) - potential(minR)                  # Height of potential-energy barrier
    hess_min = d2Vdx2(minR)                                     # Hessian at reactant
    hess_TS  = d2Vdx2(TS)                                       # Hessian at TS
    det_TS   = np.linalg.det(hess_TS)
    det_min  = np.linalg.det(hess_min)
    rothess  = np.matmul(μ,hess_TS)
    lamb     = -np.sort(np.linalg.eigvals(rothess))[0]          # Eigenvalue along unstable diffusive mode (reaction coordinate)
    pre      = lamb/(2*np.pi) * np.sqrt(-det_min/det_TS)        # Prefactor 
    k        = pre*np.exp(-β*barrier)                           # Rate
    return k

print("KLT rate (only meaningful at equilibrium, ν = 0)", KLT())
print("Instanton rate", instrate())

### Plotting ###
print("### Plot instanton on top of potential and vector field ###")
fig, ax  = plt.subplots(figsize=(8,6))
xplot = np.linspace(-1.8,1.8,101)
yplot = np.linspace(-1.8,1.8,101)
X, Y = np.meshgrid(xplot,yplot)
# Total force
FX = -A/B**4*(4*(X+Y)**3 - 4*B**2*(X+Y)) - 2*(X-Y) +  ν*((X-Y)**3 - (X-Y)) 
FY = -A/B**4*(4*(X+Y)**3 - 4*B**2*(X+Y)) + 2*(X-Y) +  ν*((X-Y)**3 - (X-Y)) 
qr = plt.streamplot(X,Y,FX,FY,density=1.5,color='C0',linewidth=1.5)

# Potential 
Z = np.zeros((len(xplot),len(yplot)))
for indx, xi in enumerate(xplot):
    for indy, yi in enumerate(yplot):
        Z[indx,indy] = potential(np.array([xi,yi]))
cm = plt.contourf(X,Y,Z,levels=20)
fig.colorbar(cm)

plt.scatter(minR[0],minR[1],s=70,marker='P',color='k',label='Reactant',zorder=3)
plt.scatter(minP[0],minP[1],s=70,marker='s',color='k',label='Product',zorder=3)
plt.scatter(TS[0],TS[1],s=70,marker='X',color='k',label='TS',zorder=3)
plt.scatter(instp[:,0],instp[:,1],s=30,color='g',zorder=2)

plt.xlabel("x")
plt.ylabel("y")
plt.xlim(np.min(xplot),np.max(xplot))
plt.ylim(np.min(yplot),np.max(yplot))
plt.legend()
plt.tight_layout()

plt.show()
