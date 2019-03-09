#!/usr/bin/env python

import time
import numpy
from functools import reduce
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf import __config__

def h1e_for_cas(casci, mo_coeff=None, ncas=None, ncore=None):
    '''CAS sapce one-electron hamiltonian

    Args:
        casci : a CASSCF/CASCI object or RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    if ncas is None: ncas = casci.ncas
    if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    hcore = casci.get_hcore()
    energy_core = casci.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = numpy.dot(mo_core, mo_core.T) * 2
        corevhf = casci.get_veff(casci.mol, core_dm)
        energy_core += numpy.einsum('ij,ji', core_dm, hcore)
        energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore+corevhf, mo_cas))
    return h1eff, energy_core

def kernel(casci, mo_coeff=None, ci0=None, verbose=logger.NOTE):
    '''CASCI solver
    '''
    if mo_coeff is None: mo_coeff = casci.mo_coeff
    log = logger.new_logger(casci, verbose)
    t0 = (time.clock(), time.time())
    log.debug('Start CASCI')

    ncas = casci.ncas
    nelecas = casci.nelecas

    # 2e
    eri_cas = casci.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to CAS space', *t0)

    # 1e
    h1eff, energy_core = casci.get_h1eff(mo_coeff)
    log.debug('core energy = %.15g', energy_core)
    t1 = log.timer('effective h1e in CAS space', *t1)

    if h1eff.shape[0] != ncas:
        raise RuntimeError('Active space size error. nmo=%d ncore=%d ncas=%d' %
                           (mo_coeff.shape[1], casci.ncore, ncas))

    # FCI
    max_memory = max(400, casci.max_memory-lib.current_memory()[0])
    e_tot, fcivec = casci.fcisolver.kernel(h1eff, eri_cas, ncas, nelecas,
                                           ci0=ci0, verbose=log,
                                           max_memory=max_memory,
                                           ecore=energy_core)

    t1 = log.timer('FCI solver', *t1)
    e_cas = e_tot - energy_core
    return e_tot, e_cas, fcivec

class CASCI(lib.StreamObject):
    '''CASCI

    Args:
        mf_or_mol : SCF object or Mole object
            SCF or Mole to define the problem size.
        ncas : int
            Number of active orbitals.
        nelecas : int or a pair of int
            Number of electrons in active space.

    Kwargs:
        ncore : int
            Number of doubly occupied core orbitals. If not presented, this
            parameter can be automatically determined.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        ncas : int
            Active space size.
        nelecas : tuple of int
            Active (nelec_alpha, nelec_beta)
        ncore : int or tuple of int
            Core electron number.  In UHF-CASSCF, it's a tuple to indicate the different core eletron numbers.

    Saved results:
        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray
            When canonicalization is specified, the orbitals are canonical
            orbitals which make the general Fock matrix (Fock operator on top
            of MCSCF 1-particle density matrix) diagonalized within each
            subspace (core, active, external).  If natorb (natural orbitals in
            active space) is specified, the active segment of the mo_coeff is
            natural orbitls.
    '''

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None):
        if isinstance(mf_or_mol, gto.Mole):
            mf = scf.RHF(mf_or_mol)
        else:
            mf = mf_or_mol

        mol = mf.mol
        self.mol = mol
        self._scf = mf
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mf.max_memory
        self.ncas = ncas
        if isinstance(nelecas, (int, numpy.integer)):
            nelecb = (nelecas-mol.spin)//2
            neleca = nelecas - nelecb
            self.nelecas = (neleca, nelecb)
        else:
            self.nelecas = (nelecas[0],nelecas[1])
        self.ncore = ncore
# CI solver parameters are set in fcisolver object
        self.fcisolver.lindep = getattr(__config__,
                                        'mcscf_casci_CASCI_fcisolver_lindep', 1e-10)
        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_casci_CASCI_fcisolver_max_cycle', 200)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_casci_CASCI_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = 0
        self.e_cas = None
        self.ci = None
        self.mo_coeff = mf.mo_coeff
        self.mo_energy = mf.mo_energy
        self.converged = False

        keys = set(('natorb', 'canonicalization', 'sorting_mo_energy'))
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def ncore(self):
        if self._ncore is None:
            ncorelec = self.mol.nelectron - sum(self.nelecas)
            assert(ncorelec % 2 == 0)
            return ncorelec // 2
        else:
            return self._ncore
    @ncore.setter
    def ncore(self, x):
        assert(x is None or isinstance(x, (int, numpy.integer)))
        self._ncore = x

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** CASCI flags ********')
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        assert(self.ncas > 0)
        log.info('max_memory %d (MB)', self.max_memory)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')
        return self

    def energy_nuc(self):
        return self._scf.energy_nuc()

    def get_hcore(self, mol=None):
        return self._scf.get_hcore(mol)

    def get_veff(self, mol=None, dm=None, hermi=1):
        if mol is None: mol = self.mol
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = numpy.dot(mocore, mocore.T) * 2
# don't call self._scf.get_veff because _scf might be DFT object
        vj, vk = self._scf.get_jk(mol, dm, hermi=hermi)
        return vj - vk * .5

    def _eig(self, h, *args):
        return scf.hf.eig(h, None)

    def get_h2cas(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.

        Note It is different to get_h2eff when df.approx_hessian is applied,
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        return self.ao2mo(mo_coeff)
    def get_h2eff(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.

        Note It is different to get_h2cas when df.approx_hessian is applied.
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        return self.ao2mo(mo_coeff)
    def ao2mo(self, mo_coeff=None):
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            ncore = self.ncore
            mo_coeff = self.mo_coeff[:,ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:,ncore:nocc]

        if self._scf._eri is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    get_h1cas = h1e_for_cas = h1e_for_cas

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        return self.h1e_for_cas(mo_coeff, ncas, ncore)
    get_h1eff.__doc__ = h1e_for_cas.__doc__

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        return self.kernel(mo_coeff, ci0, verbose)
    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None:
            ci0 = self.ci
        log = logger.new_logger(self, verbose)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = \
                kernel(self, mo_coeff, ci0=ci0, verbose=log)

        if self.canonicalization:
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = numpy.all(self.fcisolver.converged)
            if self.converged:
                log.info('CASCI converged')
            else:
                log.info('CASCI not converged')
        else:
            self.converged = True
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square', None):
            if isinstance(self.e_cas, (float, numpy.number)):
                ss = self.fcisolver.spin_square(self.ci, self.ncas, self.nelecas)
                log.note('CASCI E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                         self.e_tot, self.e_cas, ss[0])
            else:
                for i, e in enumerate(self.e_cas):
                    ss = self.fcisolver.spin_square(self.ci[i], self.ncas, self.nelecas)
                    log.note('CASCI state %d  E = %.15g  E(CI) = %.15g  S^2 = %.7f',
                             i, self.e_tot[i], e, ss[0])
        else:
            if isinstance(self.e_cas, (float, numpy.number)):
                log.note('CASCI E = %.15g  E(CI) = %.15g', self.e_tot, self.e_cas)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note('CASCI state %d  E = %.15g  E(CI) = %.15g',
                             i, self.e_tot[i], e)
        return self

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        '''One-particle density matrix in AO representation
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas is None: ncas = self.ncas
        if nelecas is None: nelecas = self.nelecas
        if ncore is None: ncore = self.ncore

        casdm1 = self.fcisolver.make_rdm1(ci, ncas, nelecas)
        mocore = mo_coeff[:,:ncore]
        mocas = mo_coeff[:,ncore:ncore+ncas]
        dm1 = numpy.dot(mocore, mocore.T) * 2
        dm1 = dm1 + reduce(numpy.dot, (mocas, casdm1, mocas.T))
        return dm1

