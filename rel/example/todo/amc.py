#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'amc'

mol = gto.Mole()
mol.basis = {'Am':'unc-ano','C':'unc-tzp-dk'}
mol.atom = '''
Am  0.0 0.0  0.000
C  0.0 0.0  2.150
'''
mol.charge = 0
mol.spin = 5
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.RDHF(mol)
mf.chkfile = name+'.chk'
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()
