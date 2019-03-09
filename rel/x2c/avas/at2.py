#!/usr/bin/env python

import os
import sys
import math
import time
import numpy
import ctypes
import signal
from functools import reduce

import avas
from pyscf import lib

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

_loaderpath = os.path.dirname(__file__)
libci = numpy.ctypeslib.load_library('libci.so', _loaderpath)

def find1(s):
    return [i for i,x in enumerate(bin(s)[2:][::-1]) if x is '1']

def str2orblst(string, norb):
    occ = []
    vir = []
    occ.extend([x for x in find1(string)])
    for i in range(norb): 
        if not (string & 1<<i):
            vir.append(i)
    return occ, vir

def num_strings(n, m):
    if m < 0 or m > n:
        return 0
    else:
        return math.factorial(n) // (math.factorial(n-m)*math.factorial(m))

def print_dets(self,strs):
    ndets = strs.shape[0]
    lib.logger.info(self, '*** Printing list of determinants')
    lib.logger.info(self, 'Number of determinants: %s', ndets)
    for i in range(ndets):
        lib.logger.info(self,'Det %d %s' % (i,bin(strs[i])))
    return self

def make_strings(self,orb_list,nelec):
    orb_list = list(orb_list)
    assert(nelec >= 0)
    if nelec == 0:
        return numpy.asarray([0], dtype=numpy.int64)
    elif nelec > len(orb_list):
        return numpy.asarray([], dtype=numpy.int64)
    def gen_str_iter(orb_list, nelec):
        if nelec == 1:
            res = [(1<<i) for i in orb_list]
        elif nelec >= len(orb_list):
            n = 0
            for i in orb_list:
                n = n | (1<<i)
            res = [n]
        else:
            restorb = orb_list[:-1]
            thisorb = 1 << orb_list[-1]
            res = gen_str_iter(restorb, nelec)
            for n in gen_str_iter(restorb, nelec-1):
                res.append(n | thisorb)
        return res
    strings = gen_str_iter(orb_list, nelec)
    assert(strings.__len__() == num_strings(len(orb_list),nelec))
    return numpy.asarray(strings, dtype=numpy.int64)

# TODO: When large number of dets it should be passed to C
def make_hdiag(self,h1e,h2e,h,strs):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    diagj = lib.einsum('iijj->ij', h2e)
    diagk = lib.einsum('ijji->ij', h2e)
    for i in range(ndets):
        stri = strs[i]
        occs = str2orblst(stri, norb)[0]
        e1 = h1e[occs,occs].sum()
        e2 = diagj[occs][:,occs].sum() \
           - diagk[occs][:,occs].sum()
        h[i] = e1 + e2*0.5
    return h

def c_contract(self,h1e,h2e,hdiag,civec,strs,nelec):
    ndets = strs.shape[0]
    norb = h1e.shape[0]
    ci1 = numpy.zeros_like(civec)
    h1e = numpy.asarray(h1e, order='C')
    h2e = numpy.asarray(h2e, order='C')
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    hdiag = numpy.asarray(hdiag, order='C')
    ci1 = numpy.asarray(ci1, order='C')

    libci.contract(h1e.ctypes.data_as(ctypes.c_void_p), 
                   h2e.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_int(norb), 
                   ctypes.c_int(nelec), 
                   strs.ctypes.data_as(ctypes.c_void_p), 
                   civec.ctypes.data_as(ctypes.c_void_p), 
                   hdiag.ctypes.data_as(ctypes.c_void_p), 
                   ctypes.c_ulonglong(ndets), 
                   ci1.ctypes.data_as(ctypes.c_void_p))
    return ci1

def make_rdm1(self,civec,strs,norb,nelec):
    ndets = strs.shape[0]
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    rdm1 = numpy.zeros((norb,norb), dtype=numpy.complex128)

    libci.rdm1(ctypes.c_int(norb), 
               ctypes.c_int(nelec), 
               strs.ctypes.data_as(ctypes.c_void_p), 
               civec.ctypes.data_as(ctypes.c_void_p), 
               ctypes.c_ulonglong(ndets), 
               rdm1.ctypes.data_as(ctypes.c_void_p))
    return rdm1

def make_rdm12(self,civec,strs,norb,nelec):
    ndets = strs.shape[0]
    strs = numpy.asarray(strs, order='C')
    civec = numpy.asarray(civec, order='C')
    rdm1 = numpy.zeros((norb,norb), dtype=numpy.complex128)
    rdm2 = numpy.zeros((norb,norb,norb,norb), dtype=numpy.complex128)

    libci.rdm12(ctypes.c_int(norb), 
                ctypes.c_int(nelec), 
                strs.ctypes.data_as(ctypes.c_void_p), 
                civec.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_ulonglong(ndets), 
                rdm1.ctypes.data_as(ctypes.c_void_p),
                rdm2.ctypes.data_as(ctypes.c_void_p))
    return rdm1,rdm2.transpose(0,2,1,3)

def _make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, nmo):
    nocc = ncas + ncore
    dm1 = numpy.zeros((nmo,nmo), dtype=numpy.complex128)
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 1
    dm1[ncore:nocc,ncore:nocc] = casdm1
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo), dtype=numpy.complex128)
    dm2[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i,i,j,j] += +1
            dm2[i,j,j,i] += -1
        dm2[i,i,ncore:nocc,ncore:nocc] = dm2[ncore:nocc,ncore:nocc,i,i] = casdm1
        dm2[i,ncore:nocc,ncore:nocc,i] = dm2[ncore:nocc,i,i,ncore:nocc] = -casdm1
    return dm1, dm2

def init_casci(self,ncore,norb,nelec,coeff):
    e_core = self.mol.energy_nuc() 
    nao, nmo = coeff.shape
    nvir = nmo - ncore - norb
    lib.logger.info(self, '\n *** A simple relativistic CI module')
    lib.logger.info(self, 'Number of occupied core 2C spinors %s', ncore)
    lib.logger.info(self, 'Number of virtual core 2C spinors %s', nvir)
    lib.logger.info(self, 'Number of electrons to be correlated %s', nelec)
    lib.logger.info(self, 'Number of 2C spinors to be correlated %s', norb)
    if (norb > 64):
        raise RuntimeError('''Only support up to 64 orbitals''')

    hcore = self.get_hcore()
    corevhf = numpy.zeros((nao,nao), dtype=numpy.complex128)
    if (ncore != 0):
        core_idx = numpy.arange(ncore)
        core_dm = numpy.dot(coeff[:, core_idx], coeff[:, core_idx].conj().T)
        e_core += numpy.einsum('ij,ji->', core_dm, hcore)
        corevhf = mf.get_veff(mol, core_dm)
        e_core += numpy.einsum('ij,ji->', core_dm, corevhf)*0.5
        e_core = e_core.real

    ci_idx = ncore + numpy.arange(norb)
    h1e = reduce(numpy.dot, (coeff[:, ci_idx].conj().T, hcore+corevhf, coeff[:,ci_idx]))
    t1 = (time.clock(), time.time())
    eri_mo = ao2mo.kernel(self.mol, coeff[:, ci_idx], compact=False, intor='int2e_spinor')
    lib.logger.timer(self,'ao2mo build', *t1)
    eri_mo = eri_mo.reshape(norb,norb,norb,norb)

    orb_list = list(range(norb))
    t1 = (time.clock(), time.time())
    strs = make_strings(self,orb_list,nelec) 
    #print_dets(self,strs)
    lib.logger.timer(self,'det strings build', *t1)
    ndets = strs.shape[0]
    lib.logger.info(self, 'Number of dets in civec %s', ndets)

    return ndets, strs, h1e, eri_mo, e_core

def kernel_casci(self,ndets,strs,h1e,eri_mo, e_core):
    t0 = (time.clock(), time.time())
    hdiag = numpy.zeros(ndets, dtype=numpy.complex128)
    t2 = (time.clock(), time.time())
    hdiag = make_hdiag(self,h1e,eri_mo,hdiag,strs) 
    lib.logger.timer(self,'<i|H|i> build', *t2)

    t3 = (time.clock(), time.time())
    ci0 = numpy.zeros(ndets, dtype=numpy.complex128)
    ci0[0] = 1.0
    def hop(c):
        hc = c_contract(self,h1e,eri_mo,hdiag,c,strs,nelec)
        return hc.ravel()
    level_shift = 0.01
    precond = lambda x, e, *args: x/(hdiag-e+level_shift)
    nthreads = lib.num_threads()
    conv_tol = 1e-12
    lindep = 1e-14
    max_cycle = 100
    max_space = 14
    lessio = False
    max_memory = 4000
    follow_state = False 
    nroots = 1
    with lib.with_omp_threads(nthreads):
        e, c = lib.davidson(hop, ci0, precond, tol=conv_tol, lindep=lindep,
                            max_cycle=max_cycle, max_space=max_space, max_memory=max_memory, 
                            dot=numpy.dot, nroots=nroots, lessio=lessio, verbose=mf.verbose,
                            follow_state=follow_state)
    lib.logger.timer(self,'<i|H|j> Davidson', *t3)

    e += e_core
    lib.logger.info(self, 'Core energy %s', e_core)
    lib.logger.info(self, 'Ground state energy %s', e)
    lib.logger.debug(self, 'Ground state civec %s', c)
    norm = numpy.einsum('i,i->',c.conj(),c)
    lib.logger.info(self, 'Norm of ground state civec %s', norm)
    lib.logger.timer(self,'CI build', *t0)
    return c

if __name__ == '__main__':
    from pyscf import gto, scf, x2c, ao2mo, mcscf

    name = 'at2_x2c_ci'
    mol = gto.Mole()
    mol.atom = '''
At 0.0 0.0  0.000
At 0.0 0.0  3.100
    '''
    mol.basis = 'dzp-dk'
    mol.verbose = 4
    mol.spin = 0
    mol.charge = 0
    mol.symmetry = 0
    mol.build()

    mf = x2c.RHF(mol)
    mf.chkfile = name+'.chk'
    mf.with_x2c.basis = 'unc-ano'
    dm = mf.get_init_guess() + 0.1j
    mf.kernel(dm)

    ncore = 156
    #norb = 14
    #nelec = 8
    #coeff = mf.mo_coeff
    aolst = ['At 6p', 'At 6s']
    norb, nelec, coeff = avas.avas(mf, aolst, ncore=ncore, threshold_occ=0.1, threshold_vir=1e-2)

    lib.logger.TIMER_LEVEL = 3

    ndets, strs, h1e, eri_mo, e_core = init_casci(mf,ncore,norb,nelec,coeff) 
    c = kernel_casci(mf,ndets,strs,h1e,eri_mo,e_core) 

    #t0 = (time.clock(), time.time())
    #rdm1 = make_rdm1(mf,c,strs,norb,nelec)
    #nele = numpy.einsum('ii->', rdm1)
    #lib.logger.info(mf, 'Number of electrons in active space %s', nele)
    #natocc, natorb = numpy.linalg.eigh(rdm1)
    #natorb = numpy.dot(coeff[:,:ncore+norb], natorb)
    #lib.logger.debug(mf, 'Natural occupations active space %s', natocc)
    #lib.logger.info(mf, 'Sum of natural occupations %s', natocc.sum())
    #lib.logger.timer(mf,'1-RDM build', *t0)

    #t0 = (time.clock(), time.time())
    #rdm1, rdm2 = make_rdm12(mf,c,strs,norb,nelec)
    #rdm1_check = numpy.einsum('ijkk->ij', rdm2) / (nelec-1)
    #norm = numpy.linalg.norm(rdm1-rdm1_check)
    #lib.logger.info(mf, 'Diff in 1-RDM %s', norm)
    #lib.logger.timer(mf,'1/2-RDM build', *t0)

    #TODO:include core/core-valence contribution and make consistent
    #t0 = (time.clock(), time.time())
    #hcore = mf.get_hcore()
    #h1e = reduce(numpy.dot, (coeff[:,:norb].conj().T, hcore, coeff[:,:norb]))
    #e_core = mol.energy_nuc() 
    #e1 = numpy.einsum('ij,ji->', rdm1, h1e)
    #e2 = numpy.einsum('ijkl,ijkl->', rdm2, eri_mo)*0.5
    #et = e1+e2+e_core
    #lib.logger.info(mf, 'Total energy with 1/2-RDM %s', et)
    #lib.logger.timer(mf,'1/2-RDM energy build', *t0)
