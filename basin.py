#!/usr/bin/env python

import sys
import time
import h5py
import numpy
import signal
from pyscf import lib, dft
from pyscf.lib import logger

import grid

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

NPROPS = 3
PROPS = ['density', 'kinetic', 'laplacian']
EPS = 1e-7

# TODO: screaning of points
def rho(self,x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(self.mol, x, deriv=2)
    ngrids, nao = ao[0].shape
    pos = self.mo_occ > self.occdrop
    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros((3,ngrids))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    for i in range(1, 4):
        c1 = numpy.dot(ao[i], cpos)
        rho[1] += numpy.einsum('pi,pi->p', c1, c1)
    XX, YY, ZZ = 4, 7, 9
    ao2 = ao[XX] + ao[YY] + ao[ZZ]
    c1 = numpy.dot(ao2, cpos)
    rho[2] = numpy.einsum('pi,pi->p', c0, c1)
    rho[2] += rho[1]
    rho[2] *= 2
    rho[1] *= 0.5
    return rho

def inbasin(self,r,j):
    isin = False
    rs1 = 0.0
    irange = self.nlimsurf[j]
    for k in range(irange):
        rs2 = self.rsurf[j,k]
        if (r >= rs1-EPS and r <= rs2+EPS):
            if (((k+1)%2) == 0):
                isin = False
            else:
                isin = True
            return isin
        rs1 = rs2
    return isin

def out_beta(self):
    logger.info(self,'* Go outside betasphere')
    xcoor = numpy.zeros(3)
    nrad = self.nrad
    npang = self.npang
    iqudr = self.iqudr
    mapr = self.mapr
    r0 = self.brad
    rfar = self.rmax
    rad = self.rad
    t0 = time.time()
    rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
    coordsang = self.agrids
    rprops = numpy.zeros(NPROPS)
    for n in range(nrad):
        r = rmesh[n]
        coords = []
        weigths = []
        for j in range(npang):
            inside = True
            inside = inbasin(self,r,j)
            if (inside == True):
                cost = coordsang[j,0]
                sintcosp = coordsang[j,1]*coordsang[j,2]
                sintsinp = coordsang[j,1]*coordsang[j,3]
                xcoor[0] = r*sintcosp
                xcoor[1] = r*sintsinp
                xcoor[2] = r*cost    
                p = self.xnuc + xcoor
                coords.append(p)
                weigths.append(coordsang[j,4])
        coords = numpy.array(coords)
        weigths = numpy.array(weigths)
        val = rho(self,coords)
        props = numpy.einsum('pi,i->p', val, weigths)
        rprops += props*dvol[n]*rwei[n]
    for i in range(NPROPS):
        logger.info(self,'*--> %s density outside bsphere %8.5f ', PROPS[i], rprops[i])    
    logger.info(self,'Time out Bsphere %.3f (sec)' % (time.time()-t0))
    return rprops
    
def int_beta(self): 
    logger.info(self,'* Go with inside betasphere')
    xcoor = numpy.zeros(3)
    npang = self.bnpang
    coords = numpy.empty((npang,3))
    nrad = self.bnrad
    iqudr = self.biqudr
    mapr = self.bmapr
    r0 = 0
    rfar = self.brad
    rad = self.rad
    t0 = time.time()
    rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
    coordsang = grid.lebgrid(npang)
    rprops = numpy.zeros(NPROPS)
    for n in range(nrad):
        r = rmesh[n]
        for j in range(npang): # j-loop can be changed to map
            cost = coordsang[j,0]
            sintcosp = coordsang[j,1]*coordsang[j,2]
            sintsinp = coordsang[j,1]*coordsang[j,3]
            xcoor[0] = r*sintcosp
            xcoor[1] = r*sintsinp
            xcoor[2] = r*cost    
            p = self.xnuc + xcoor
            coords[j] = p
        val = rho(self,coords)
        props = numpy.einsum('pi,i->p', val, coordsang[:,4])
        rprops += props*dvol[n]*rwei[n]
    for i in range(NPROPS):
        logger.info(self,'*--> %s density inside bsphere %8.5f ', PROPS[i], rprops[i])    
    logger.info(self,'Time in Bsphere %.3f (sec)' % (time.time()-t0))
    return rprops

class Basin(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inuc = 0
        self.nrad = 101
        self.iqudr = 'legendre'
        self.mapr = 'becke'
        self.betafac = 0.4
        self.bnrad = 101
        self.bnpang = 3074
        self.biqudr = 'legendre'
        self.bmapr = 'becke'
        self.non0tab = False
        self.corr = False
        self.occdrop = 1e-6
##################################################
# don't modify the following attributes, they are not input options
        self.rdm1 = None
        self.nocc = None
        self.mol = None
        self.mo_coeff = None
        self.mo_occ = None
        self.ntrial = None
        self.npang = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.xnuc = None
        self.xyzrho = None
        self.agrids = None
        self.rsurf = None
        self.nlimsurf = None
        self.rmin = None
        self.rmax = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.nprims = None
        self.nmo = None
        self.rad = None
        self.brad = None
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
        logger.info(self,'Number of molecular orbitals %d' % self.nmo)
        logger.info(self,'Orbital EPS occ criterion %e' % self.occdrop)
        logger.info(self,'Number of occupied molecular orbitals %d' % self.nocc)
        logger.info(self,'Number of molecular primitives %d' % self.nprims)
        logger.debug(self,'Occs : %s' % self.mo_occ) 

        logger.info(self,'* Surface Info')
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'Properties for nuc %d' % self.inuc)
        logger.info(self,'Nuclear coordinate %.6f  %.6f  %.6f', *self.xnuc)
        logger.info(self,'Rho nuclear coordinate %.6f  %.6f  %.6f', *self.xyzrho[self.inuc])
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Rmin for surface %f', self.rmin)
        logger.info(self,'Rmax for surface %f', self.rmax)

        logger.info(self,'* Radial and angular grid Info')
        logger.info(self,'Npang points inside %d' % self.bnpang)
        logger.info(self,'Number of radial points outside %d', self.nrad)
        logger.info(self,'Number of radial points inside %d', self.bnrad)
        logger.info(self,'Radial outside quadrature %s', self.iqudr)
        logger.info(self,'Radial outside mapping %s', self.mapr)
        logger.info(self,'Radial inside quadrature %s', self.biqudr)
        logger.info(self,'Radial inside mapping %s', self.bmapr)
        logger.info(self,'Slater-Bragg radii %f', self.rad) 
        logger.info(self,'Beta-Sphere factor %f', self.betafac)
        logger.info(self,'Beta-Sphere radi %f', self.brad)
        logger.info(self,'')

        return self

    def build(self):

        t0 = (time.clock(), time.time())
        lib.logger.TIMER_LEVEL = 3

        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.nelectron = self.mol.nelectron 
        self.charge = self.mol.charge    
        self.spin = self.mol.spin      
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        nprims, nmo = self.mo_coeff.shape 
        self.nprims = nprims
        self.nmo = nmo
        if self.charges[self.inuc] == 1:
            self.rad = grid.BRAGG[self.charges[self.inuc]]
        else:
            self.rad = grid.BRAGG[self.charges[self.inuc]]*0.5

        if (self.corr):
            self.rdm1 = lib.chkfile.load(self.chkfile, 'rdm/rdm1') 
            natocc, natorb = numpy.linalg.eigh(self.rdm1)
            natorb = numpy.dot(self.mo_coeff, natorb)
            self.mo_coeff = natorb
            self.mo_occ = natocc
        nocc = self.mo_occ[abs(self.mo_occ)>self.occdrop]
        nocc = len(nocc)
        self.nocc = nocc

        idx = 'atom'+str(self.inuc)
        with h5py.File(self.surfile) as f:
            self.xnuc = f[idx+'/xnuc'].value
            self.xyzrho = f[idx+'/xyzrho'].value
            self.npang = f[idx+'/npang'].value
            self.ntrial = f[idx+'/ntrial'].value
            self.rmin = f[idx+'/rmin'].value
            self.rmax = f[idx+'/rmax'].value
            self.rsurf = f[idx+'/rsurf'].value
            self.nlimsurf = f[idx+'/nlimsurf'].value
            self.agrids = f[idx+'/coords'].value

        self.brad = self.rmin*self.betafac

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        if (self.iqudr == 'legendre'):
            self.iqudr = 1
        if (self.biqudr == 'legendre'):
            self.biqudr = 1

        if (self.mapr == 'becke'):
            self.mapr = 1
        elif (self.mapr == 'exp'):
            self.mapr = 2
        elif (self.mapr == 'none'):
            self.mapr = 0 
        if (self.bmapr == 'becke'):
            self.bmapr = 1
        elif (self.bmapr == 'exp'):
            self.bmapr = 2
        elif (self.bmapr == 'none'):
            self.bmapr = 0

        with lib.with_omp_threads(self.nthreads):
            brprops = int_beta(self)
            rprops = out_beta(self)

        logger.info(self,'Write info to HDF5 file')
        atom_dic = {'inprops':brprops,
                    'outprops':rprops,
                    'totprops':(brprops+rprops)}
        lib.chkfile.save(self.surfile, 'atom_props'+str(self.inuc), atom_dic)
        for i in range(NPROPS):
            logger.info(self,'*-> Total %s density %8.5f ', PROPS[i], (rprops[i]+brprops[i]))    
        logger.info(self,'')
        logger.info(self,'Basim properties of atom %d done',self.inuc)
        logger.timer(self,'Basin build', *t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'prueba.chk'
    natm = 3
    bas = Basin(name)
    bas.verbose = 4
    bas.nrad = 221
    bas.iqudr = 'legendre'
    bas.mapr = 'exp'
    bas.bnrad = 121
    bas.bnpang = 3074
    bas.biqudr = 'legendre'
    bas.bmapr = 'exp'
    bas.betafac = 0.4
    for i in range(natm):
        bas.inuc = i
        bas.kernel()

