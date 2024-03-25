#!/usr/bin/env python3
""" DVR calculation of the rate for Fig. 2 """
__author__ = 'Dr. Jorge L. Rosa-Raíces'

import sys
import pickle

import numpy as np
import matplotlib as mpl
import scipy.sparse as sp
import scipy.sparse.linalg as la

# for Hermite DVR
from numpy.polynomial import hermite
from scipy.special import factorial

# plotting
from matplotlib import colors, pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from argparse import ArgumentParser

# miscellaneous functional programming
import functools as ft
import itertools as it
import operator as op

# type hinting
from abc import ABC, abstractmethod
from typing import Sequence, Iterable, Any
from numpy.typing import NDArray, ArrayLike

rho_cmap = plt.cm.turbo_r

parser = ArgumentParser(description='''Solve the FPE for the steady-state
                                       density of low-dimensional overdamped
                                       diffusive systems with the DVR
                                       method''')

parser.add_argument('-nx', type=int, default=150,
                    help='number of DVR grid points on x-axis')
parser.add_argument('-ny', type=int, default=150,
                    help='number of DVR grid points on y-axis')
parser.add_argument('-lx', type=float, default=9.,
                    help='DVR grid extent on x-axis')
parser.add_argument('-ly', type=float, default=6.,
                    help='DVR grid extent on y-axis')

parser.add_argument('-ν', type=float, default=4, help='shear rate')
parser.add_argument('-β', type=float, default=3, help='inverse temperature')

RHOMIN = 1e-5
RHOMAX = 1e+5


class Potential(ABC):
    ''' Potential abstract interface '''

    @abstractmethod
    def __call__(self, points: NDArray | Sequence[NDArray],
                 grad: bool = False, lapl: bool = False) -> Sequence[NDArray]:
        ...


class DVR(ABC):
    ''' DVR abstract base class '''

    def __init__(self, **kwds: Any) -> None:
        self.x = np.empty((0,), dtype=np.float64)

    @abstractmethod
    def _dvr_grid(self) -> Sequence[NDArray]:
        ...

    @abstractmethod
    def _dvr_basis(self, x: NDArray | None = None) -> NDArray:
        ...

    @abstractmethod
    def _dvr_difop(self, order: int) -> NDArray:
        ...

    def dfdx(self) -> NDArray:

        return self._dvr_difop(order=1)

    def d2fdx2(self) -> NDArray:

        return self._dvr_difop(order=2)

    def grid(self) -> Sequence[NDArray]:

        return self._dvr_grid()

    def basis(self) -> NDArray:

        return self._dvr_basis(x=self.x)

    def pes(self, pes: Potential, **kwds: bool) -> Sequence[NDArray]:

        return pes(self.grid(), **kwds)


class SincDirichletDVR(DVR):
    ''' One-dimensional DVR constructed with Sinc--Dirichlet functions
        orthogonal on L^2(R; dx) '''

    def __init__(self, N: int = 10, L: float = 2.0) -> None:

        super().__init__()
        self.N = N
        self.L = L

        self.dx = self.L / self.N
        self.n = np.arange(self.N)
        self.x = self.n * self.dx - 0.5 * (self.L - self.dx)
        self.w = np.full((self.N,), self.dx)

        return None

    def _dvr_grid(self) -> Sequence[NDArray]:

        return [self.x,]

    def _dvr_basis(self, x: NDArray | None = None) -> NDArray:

        if x is None:
            x_k = self.x[np.newaxis, :]
        else:
            x_k = x.astype(self.x.dtype)[np.newaxis, :]
        x_j = self.x[:, np.newaxis]

        basis = np.sinc((x_j - x_k) / self.dx) / np.sqrt(self.dx)

        return basis

    def _dvr_difop(self, order: int) -> NDArray:

        i = self.n[:, np.newaxis] - self.n[np.newaxis, :]

        if order == 1:
            result = np.divide(
                (-1.0) ** i, i.astype(np.float64), where=(i != 0),
                out=np.zeros_like(i, dtype=np.float64),
            )
            result /= self.dx

        elif order == 2:
            result = np.divide(
                (-1.0) ** i, (i ** 2).astype(np.float64), where=(i != 0),
                out=np.eye(self.N, dtype=np.float64) * np.pi ** 2 / 6.0,
            )
            result /= -0.5 * self.dx ** 2

        else:
            raise NotImplementedError

        return result


def all_equal(iterable: Iterable[Any]) -> bool:
    ''' Checks that all elements of an iterable are identical.
        Adapted from source: https://stackoverflow.com/questions/3844801 '''

    g = it.groupby(iterable)
    return next(g, True) and not next(g, False)  # type: ignore[arg-type]


def cartesian_product(arrays: Sequence[NDArray],
                      unpack: bool = True) -> NDArray:
    ''' Computes the Cartesian product of a list of equidimensional arrays.
        Adapted from source: https://stackoverflow.com/questions/11144513 '''

    assert all_equal(a.ndim for a in arrays)

    if unpack:
        shape = (len(arrays), -1,)
        axis = 0
    else:
        shape = (-1, len(arrays),)
        axis = -1

    result = np.stack(np.meshgrid(
        *arrays, indexing='ij', sparse=False, copy=False,
    ), axis=axis).reshape(shape)

    return result


def cartesian_findiff(findiffs: Sequence[NDArray]) -> Sequence[sp.spmatrix]:
    ''' Builds the rectangular finite-difference grid for a n-dimensional
        differential operator from n 1-dimensional finite-difference grids '''

    # check that args are consistent with intended function usage
    if not (all(f.ndim == 2 for f in findiffs) and
            all(all_equal(f.shape) for f in findiffs)):

        raise ValueError(("Argument `findiffs` to function/method "
                          "`cartesian_findiff` is not a sequence "
                          "of square matrices!"))

    sizes = list(f.shape[0] for f in findiffs)

    lsize = list(it.accumulate(sizes[:-1], op.mul))
    lsize.insert(0, 1)

    rsize = list(it.accumulate(sizes[::-1][:-1], op.mul))[::-1]
    rsize.append(1)

    return [sp.kron(sp.block_diag(lsize[k] * [sp.bsr_matrix(f),]),
                    sp.eye(rsize[k])) for (k, f,) in enumerate(findiffs)]


class ProductDVR(object):
    ''' Multidimensional DVR constructed from the
        Cartesian product of 1-dimensional DVRs '''

    def __init__(self, *dvrs: DVR) -> None:

        self.dvrs = dvrs
        self.ndim = len(dvrs)
        self.size = ft.reduce(op.mul, (dvr.x.size for dvr in dvrs))
        self.x = cartesian_product(list(it.chain.from_iterable(dvr.grid()
                                                               for dvr in dvrs)))

        return None

    def dfdx(self) -> Sequence[sp.spmatrix]:

        return cartesian_findiff([dvr.dfdx() for dvr in self.dvrs])

    def d2fdx2(self) -> Sequence[sp.spmatrix]:

        return cartesian_findiff([dvr.d2fdx2() for dvr in self.dvrs])

    def basis(self) -> NDArray:

        return ft.reduce(np.kron, [dvr.basis() for dvr in self.dvrs])

    def grid(self, mesh: bool = False) -> Sequence[NDArray]:

        grids_1d = list(it.chain.from_iterable(dvr.grid() for dvr
                                               in self.dvrs))

        if mesh:
            return np.meshgrid(*grids_1d, indexing='ij', sparse=True)
        else:
            return grids_1d

    def pes(self, pes: Potential, **kwds: bool) -> Sequence[NDArray]:

        return pes(self.x, **kwds)


class DoubleWell(Potential):

    def __call__(self, points: NDArray | Sequence[NDArray],
                 grad: bool = False, lapl: bool = False) -> Sequence[NDArray]:
        x, y = points[:2]
        A = 2.0
        B = 1.8

        result = [A/B**4*((x+y)**4 + B**4 - 2*(x+y)**2*B**2) + (x-y)**2 
        ,]
        if grad:
            result.append([
                A/B**4*(4*(x+y)**3 - 4*B**2*(x+y)) + 2*(x-y) , 
                A/B**4*(4*(x+y)**3 - 4*B**2*(x+y)) - 2*(x-y) , 
            ])
        if lapl:
            result.append(
                    A/B**4*(12*(x+y)**2 - 4*B**2) + 2  +  
                    A/B**4*(12*(x+y)**2 - 4*B**2) + 2  
            )

        return result


if __name__ == '__main__':

    # enable pretty-printing of arrays
    np.set_printoptions(suppress=True, precision=10,
                        threshold=sys.maxsize, linewidth=180)

    args = parser.parse_args()

    pes: Potential
    dvr: ProductDVR

    # instantiate DVR grid as Cartesian product of 1D DVR grids
    dvr = ProductDVR(SincDirichletDVR(N=args.nx, L=args.lx),
                     SincDirichletDVR(N=args.ny, L=args.ly),)
    pes = DoubleWell()  

    shear = [args.ν*((dvr.x[0]-dvr.x[1])**3 - (dvr.x[0]-dvr.x[1])),
             args.ν*((dvr.x[0]-dvr.x[1])**3 - (dvr.x[0]-dvr.x[1])),] 

    # evaluate potential, gradient and Laplacian on DVR grid
    V, gradV, laplV, = dvr.pes(pes, grad=True, lapl=True)

    # evaluate gradient and Laplacian in DVR basis
    gradK = dvr.dfdx()
    laplK = ft.reduce(op.add, dvr.d2fdx2())

    # build DVR representation of Fokker--Planck operator for sheared density
    FPO = laplK / args.β + sp.diags(laplV, format=laplK.format) + \
        sum(sp.diags(gV - s, format=laplK.format).dot(gK)
            for (gV, s, gK) in zip(gradV, shear, gradK))

    # obtain 'k' leading eigenpairs of FPO via iterative Arnoldi method
    # NOTE: very slow for larger problems
    evals, evecs = la.eigs(FPO, k=5, sigma=0+0j, tol=1e-4,
                           return_eigenvectors=True)

    print("Eigenvalues",evals)
    x: NDArray
    y: NDArray

    # estimate steady-state distributions from leading DVR eigenvector
    x, y, = dvr.grid(mesh=False)

    rho = np.matmul(evecs.real[:, evals.real.argmax()],
                    dvr.basis()).reshape((x.size, y.size,))
    rho_norm = np.trapz(np.trapz(rho, y, axis=1), x, axis=0)
    rho /= rho_norm

    rho_eq = np.reshape(np.exp(-V), (x.size, y.size,))
    rho_eq_norm = np.trapz(np.trapz(rho_eq, y, axis=1), x, axis=0)
    rho_eq /= rho_eq_norm

    rhox = np.trapz(rho, y, axis=1)
    rhoy = np.trapz(rho, x, axis=0)
    rhoxy = rho

    rhox_eq = np.trapz(rho_eq, y, axis=1)
    rhoy_eq = np.trapz(rho_eq, x, axis=0)
    rhoxy_eq = rho_eq

    # compute log-probabilities from distribution estimates
    eff = -np.log(np.clip(rho, RHOMIN, RHOMAX))
    effx = -np.log(np.clip(rhox, RHOMIN, RHOMAX))
    effy = -np.log(np.clip(rhoy, RHOMIN, RHOMAX))
    effxy = -np.log(np.clip(rhoxy, RHOMIN, RHOMAX))

    eff_eq = -np.log(np.clip(rho_eq, RHOMIN, RHOMAX))
    effx_eq = -np.log(np.clip(rhox_eq, RHOMIN, RHOMAX))
    effy_eq = -np.log(np.clip(rhoy_eq, RHOMIN, RHOMAX))
    effxy_eq = -np.log(np.clip(rhoxy_eq, RHOMIN, RHOMAX))

    doob = eff - eff_eq
    doobx = effx - effx_eq
    dooby = effy - effy_eq
    doobxy = effxy - effxy_eq

    rho_clip = (rho > RHOMIN) & (rho < RHOMAX)
    rhox_clip = (rhox > RHOMIN) & (rhox < RHOMAX)
    rhoy_clip = (rhoy > RHOMIN) & (rhoy < RHOMAX)
    rhoxy_clip = (rhoxy > RHOMIN) & (rhoxy < RHOMAX)

    xlim = (-0.5 * args.lx, +0.5 * args.lx,)
    ylim = (-0.5 * args.ly, +0.5 * args.ly,)
    extent = xlim + ylim

    gs_kw = dict(width_ratios=[1, 2], height_ratios=[2, 1])
    figxy, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.0, 3.5,),
                               gridspec_kw=gs_kw, layout='constrained',)

    ax_x, ax_xy, ax__, ax_y = axes.flatten()
    ax__.axis('off')

    ax_x.sharey(ax_xy)
    ax_x.xaxis.tick_bottom()
    ax_x.xaxis.set_label_position('top')
    ax_x.set_ylabel(r'$y$')
    ax_x.set_ylim(ylim)

    ax_y.sharex(ax_xy)
    ax_y.yaxis.tick_left()
    ax_y.yaxis.set_label_position('right')
    ax_y.set_xlabel(r'$x$')
    ax_y.set_xlim(xlim)

    
    data_x = [rhoy, y,]
    label_x = r'$\rho(y)$'

    data_y = [x, rhox,]
    label_y = r'$\rho(x)$'

    data_xy = rhoxy.T
    label_xy = r'$\rho(x, y)$'

    data_cmap = rho_cmap

    ax_x.plot(*data_x, color='black', drawstyle='steps-mid', lw=2.0)
    ax_x.set_xlabel(label_x)

    xticks = [data_x[0][np.isfinite(data_x)[0]].min(),
              data_x[0][np.isfinite(data_x)[0]].max(),]
    ax_x.set_xticks(xticks)
    ax_x.set_xticklabels(['{:3.2f}'.format(t) for t in xticks])
    ax_x.axvline(x=xticks[0], lw=0.5, c='k')
    ax_x.axvline(x=xticks[-1], lw=0.5, c='k')

    ax_y.plot(*data_y, color='black', drawstyle='steps-mid', lw=2.0)
    ax_y.set_ylabel(label_y)

    yticks = [data_y[-1][np.isfinite(data_y)[-1]].min(),
              data_y[-1][np.isfinite(data_y)[-1]].max(),]
    ax_y.set_yticks(yticks)
    ax_y.set_yticklabels(['{:3.2f}'.format(t) for t in yticks])
    ax_y.axhline(y=yticks[0], lw=0.5, c='k')
    ax_y.axhline(y=yticks[-1], lw=0.5, c='k')

    cticks = [data_xy[np.isfinite(data_xy)].min(),
              data_xy[np.isfinite(data_xy)].max(),]

    data_norm = colors.Normalize(vmin=cticks[0],
                                     vmax=cticks[-1],)

    im_xy = ax_xy.imshow(data_xy, cmap=data_cmap, norm=data_norm,
                         extent=extent, aspect='auto', origin='lower',
                         interpolation='none',)
    ax_xy.set_title(label_xy, fontsize='medium')

    cbar = figxy.colorbar(im_xy, ax=ax_xy, pad=0.01)
    cbar.ax.yaxis.set_ticks(cticks)
    cbar.ax.yaxis.set_ticklabels(['{:3.2f}'.format(t) for t in cticks])

    plt.show()
