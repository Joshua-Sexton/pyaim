#!/usr/bin/env python

import os
import sys
import time
import numpy
import ctypes
import signal

from pyscf import lib
from pyscf.lib import logger

import grid

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)
libcgto = lib.load_library('libcgto')

GRADEPS = 1e-10
RHOEPS = 1e-10
MINSTEP = 1e-6
MAXSTEP = 0.75
SAFETY = 0.9
ENLARGE = 1.6
HMINIMAL = numpy.finfo(numpy.float64).eps

def rhograd(self, x):

    deriv = 1
    if self.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    drv = getattr(libcgto, feval)

    x = numpy.reshape(x, (-1,3))
    x = numpy.asarray(x, dtype=numpy.double, order='F')
    ao = numpy.zeros((4,self.nao,1), dtype=numpy.double)

    drv(ctypes.c_int(1),
    (ctypes.c_int*2)(*self.shls_slice), 
    self.ao_loc.ctypes.data_as(ctypes.c_void_p),
    ao.ctypes.data_as(ctypes.c_void_p),
    x.ctypes.data_as(ctypes.c_void_p),
    self.non0tab.ctypes.data_as(ctypes.c_void_p),
    self.atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.natm),
    self.bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.nbas),
    self.env.ctypes.data_as(ctypes.c_void_p))

    ao = numpy.swapaxes(ao, -1, -2)
    pos = abs(self.mo_occ) > self.occdrop

    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros((4,1))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    c1 = numpy.dot(ao[1], cpos)
    rho[1] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    c1 = numpy.dot(ao[2], cpos)
    rho[2] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    c1 = numpy.dot(ao[3], cpos)
    rho[3] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    gradmod = numpy.linalg.norm(rho[-3:,0])

    return rho[0,0], rho[-3:,0]/(gradmod+HMINIMAL), gradmod

def gradrho(self, xpoint, h):

    h0 = h
    niter = 0
    rho, grad, gradmod = rhograd(self,xpoint)
    grdt = grad
    grdmodule = gradmod

    while (grdmodule > GRADEPS and niter < self.mstep):
        niter += 1
        ier = 1
        while (ier != 0):
            xtemp = xpoint + h0*grdt
            rho, grad, gradmod = rhograd(self,xtemp)
            escalar = numpy.einsum('i,i->',grdt,grad) 
            if (escalar < 0.707):
                if (h0 >= MINSTEP):
                    h0 = h0/2.0
                    ier = 1
                else:
                    ier = 0
            else:
                if (escalar > 0.9): 
                    hproo = h0*ENLARGE
                    if (hproo < h):
                        h0 = hproo
                    else:
                        h0 = h
                    h0 = numpy.minimum(MAXSTEP, h0)
                ier = 0
                xpoint = xtemp
                grdt = grad
                grdmodule = gradmod
            logger.debug(self,'scalar, step in gradrho %.6f %.6f', escalar, h0)

    logger.debug(self,'nsteps in gradrho %d', niter)

    return xpoint, grdmodule


class BaderSurf(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inuc = 0
        self.epsiscp = 0.180
        self.ntrial = 11
        self.leb = True
        self.npang = 5810
        self.nptheta = 90
        self.npphi = 180
        self.iqudt = 'legendre'
        self.rmaxsurf = 10.0
        self.rprimer = 0.4
        self.backend = 'rkck'
        self.epsroot = 1e-5
        self.epsilon = 1e-5 
        self.step = 0.1
        self.mstep = 120
        self.corr = False
        self.cas = False
        self.occdrop = 1e-6
##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = None
        self.mo_occ = None
        self.nocc = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.xnuc = None
        self.xyzrho = None
        self.rpru = None
        self.grids = None
        self.rsurf = None
        self.nlimsurf = None
        self.rmin = None
        self.rmax = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.atm = None
        self.bas = None
        self.nbas = None
        self.nprims = None
        self.nmo = None
        self.env = None
        self.ao_loc = None
        self.shls_slice = None
        self.nao = None
        self.non0tab = None
        self.cart = None
        self.rdm1 = None
        self._keys = set(self.__dict__.keys())

    def dump_input(self):

        if self.verbose < logger.INFO:
            return self

        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'* General Info')
        logger.info(self,'Date %s' % time.ctime())
        logger.info(self,'Python %s' % sys.version)
        logger.info(self,'Numpy %s' % numpy.__version__)
        logger.info(self,'Number of threads %d' % self.nthreads)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'Max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        logger.info(self,'Correlated ? %s' % self.corr)
        logger.info(self,'CASSCF canonical orbitals ? %s' % self.cas)

        logger.info(self,'* Molecular Info')
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])

        logger.info(self,'* Basis Info')
        logger.info(self,'Is cartesian %s' % self.cart)
        logger.info(self,'Number of molecular orbitals %d' % self.nmo)
        logger.info(self,'Orbital EPS occ criterion %e' % self.occdrop)
        logger.info(self,'Number of occupied molecular orbitals %d' % self.nocc)
        logger.info(self,'Number of molecular primitives %d' % self.nprims)
        logger.debug(self,'Occs : %s' % self.mo_occ) 

        logger.info(self,'* Surface Info')
        if (self.leb):
            logger.info(self,'Lebedev quadrature')
        else:
            logger.info(self,'Theta quadrature %s' % self.iqudt)
            logger.info(self,'Phi is always trapezoidal')
            logger.info(self,'N(theta,phi) points %d %d' % (self.nptheta,self.npphi))
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'Surface for nuc %d' % self.inuc)
        logger.info(self,'Rmaxsurface %f' % self.rmaxsurf)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Rprimer %f' % self.rprimer)
        logger.debug(self,'Rpru : %s' % self.rpru) 
        logger.info(self,'Epsiscp %f' % self.epsiscp)
        logger.info(self,'Epsroot %e' % self.epsroot)
        logger.info(self,'ODE solver %s' % self.backend)
        logger.info(self,'ODE tool %e' % self.epsilon)
        logger.info(self,'Max steps in ODE solver %d' % self.mstep)
        logger.info(self,'')

        return self

    def build(self):

        t0 = (time.clock(), time.time())
        lib.logger.TIMER_LEVEL = 3

        mol = lib.chkfile.load_mol(self.chkfile)
        self.nelectron = mol.nelectron 
        self.charge = mol.charge    
        self.spin = mol.spin      
        self.natm = mol.natm		
        self.atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
        self.bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
        self.nbas = self.bas.shape[0]
        self.env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
        self.ao_loc = mol.ao_loc_nr()
        self.shls_slice = (0, self.nbas)
        sh0, sh1 = self.shls_slice
        self.nao = self.ao_loc[sh1] - self.ao_loc[sh0]
        self.non0tab = numpy.ones((1,self.nbas), dtype=numpy.int8)
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in mol._atom])
        self.charges = mol.atom_charges()
        #if (self.cas):
        #    self.mo_coeff = lib.chkfile.load(self.chkfile, 'mcscf/mo_coeff')
        #else:
        #    self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        nprims, nmo = self.mo_coeff.shape 
        self.nprims = nprims
        self.nmo = nmo
        self.cart = mol.cart
        if (not self.leb):
            self.npang = self.npphi*self.nptheta

        if (self.corr):
            self.rdm1 = lib.chkfile.load(self.chkfile, 'rdm/rdm1') 
            nmo = self.rdm1.shape[0]
            natocc, natorb = numpy.linalg.eigh(self.rdm1)
            if (self.cas):
                self.mo_coeff = lib.chkfile.load(self.chkfile, 'mcscf/mo_coeff')
            natorb = numpy.dot(self.mo_coeff, natorb)
            self.mo_coeff = natorb
            self.mo_occ = natocc

        nocc = self.mo_occ[abs(self.mo_occ)>self.occdrop]
        nocc = len(nocc)
        self.nocc = nocc

        if (self.ntrial%2 == 0): self.ntrial += 1
        geofac = numpy.power(((self.rmaxsurf-0.1)/self.rprimer),(1.0/(self.ntrial-1.0)))
        self.rpru = numpy.zeros((self.ntrial))
        for i in range(self.ntrial): 
            self.rpru[i] = self.rprimer*numpy.power(geofac,(i+1)-1)
        self.rsurf = numpy.zeros((self.npang,self.ntrial), order='C')
        self.nlimsurf = numpy.zeros((self.npang), dtype=numpy.int32)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        if (self.iqudt == 'legendre'):
            self.iqudt = 1

        if (self.leb):
            self.grids = grid.lebgrid(self.npang)
        else:
            self.grids = grid.anggrid(self.iqudt,self.nptheta,self.npphi)

        self.xyzrho = numpy.zeros((self.natm,3))
        t = time.time()
        for i in range(self.natm):
            self.xyzrho[i], gradmod = gradrho(self,self.coords[i],self.step)
            if (gradmod > 1e-4):
                if (self.charges[i] > 2.0):
                    logger.info(self,'Good rho position %.6f %.6f %.6f', *self.xyzrho[i])
                else:
                    raise RuntimeError('Failed finding nucleus:', *self.xyzrho[i]) 
            else:
                logger.info(self,'Check rho position %.6f %.6f %.6f', *self.xyzrho[i])
                logger.info(self,'Setting xyzrho for atom to imput coords')
                self.xyzrho[i] = self.coords[i]
        self.xnuc = numpy.asarray(self.xyzrho[self.inuc])
        logger.info(self,'Time finding nucleus %.3f (sec)' % (time.time()-t))

        if (self.backend == 'rkck'):
            backend = 1
        elif (self.backend == 'rkdp'):
            backend = 2
        else:
            raise NotImplementedError('Only rkck or rkdp ODE solver yet available') 
        
        ct_ = numpy.asarray(self.grids[:,0], order='C')
        st_ = numpy.asarray(self.grids[:,1], order='C')
        cp_ = numpy.asarray(self.grids[:,2], order='C')
        sp_ = numpy.asarray(self.grids[:,3], order='C')

        t = time.time()
        feval = 'surf_driver'
        drv = getattr(libaim, feval)
        with lib.with_omp_threads(self.nthreads):
            drv(ctypes.c_int(self.inuc), 
                self.xyzrho.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.npang), 
                ct_.ctypes.data_as(ctypes.c_void_p),
                st_.ctypes.data_as(ctypes.c_void_p),
                cp_.ctypes.data_as(ctypes.c_void_p),
                sp_.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.ntrial), 
                self.rpru.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_double(self.epsiscp), 
                ctypes.c_double(self.epsroot), 
                ctypes.c_double(self.rmaxsurf), 
                ctypes.c_int(backend),
                ctypes.c_double(self.epsilon), 
                ctypes.c_double(self.step), 
                ctypes.c_int(self.mstep),
                ctypes.c_int(self.natm), 
                self.coords.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.cart),
                ctypes.c_int(self.nmo),  
                ctypes.c_int(self.nprims), 
                self.atm.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_int(self.nbas), 
                self.bas.ctypes.data_as(ctypes.c_void_p), 
                self.env.ctypes.data_as(ctypes.c_void_p), 
                self.ao_loc.ctypes.data_as(ctypes.c_void_p),
                self.mo_coeff.ctypes.data_as(ctypes.c_void_p),
                self.mo_occ.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self.occdrop), 
                self.nlimsurf.ctypes.data_as(ctypes.c_void_p),
                self.rsurf.ctypes.data_as(ctypes.c_void_p))
        logger.info(self,'Time finding surface %.3f (sec)' % (time.time()-t))
            
        self.rmin = 1000.0
        self.rmax = 0.0
        for i in range(self.npang):
            nsurf = int(self.nlimsurf[i])
            self.rmin = numpy.minimum(self.rmin,self.rsurf[i,0])
            self.rmax = numpy.maximum(self.rmax,self.rsurf[i,nsurf-1])
        logger.info(self,'Rmin for surface %.6f', self.rmin)
        logger.info(self,'Rmax for surface %.6f', self.rmax)

        logger.info(self,'Write HDF5 surface file')
        atom_dic = {'inuc':self.inuc,
                    'xnuc':self.xnuc,
                    'xyzrho':self.xyzrho,
                    'coords':self.grids,
                    'npang':self.npang,
                    'ntrial':self.ntrial,
                    'rmin':self.rmin,
                    'rmax':self.rmax,
                    'nlimsurf':self.nlimsurf,
                    'rsurf':self.rsurf}
        lib.chkfile.save(self.surfile, 'atom'+str(self.inuc), atom_dic)
        logger.info(self,'Surface of atom %d saved',self.inuc)
        logger.timer(self,'BaderSurf build', *t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'h2o.chk'
    natm = 3
    surf = BaderSurf(name)
    surf.epsilon = 1e-5
    surf.epsroot = 1e-5
    surf.verbose = 4
    surf.epsiscp = 0.220
    surf.mstep = 200
    surf.npang = 5810
    for i in range(natm):
        surf.inuc = i
        surf.kernel()

