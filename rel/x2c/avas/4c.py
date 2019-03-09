#!/usr/bin/env python

from pyscf import gto, scf, lib

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
H  0.0 0.0 0.00
H  0.0 0.0 9.75
'''
mol.verbose = 4
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.DRHF(mol)
mf.with_ssss = True
mf.kernel()

mol = gto.Mole()
mol.basis = 'unc-ano'
mol.atom = '''
H  0.0 0.0 0.00
'''
mol.verbose = 4
mol.spin = 1
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.DUHF(mol)
mf.with_ssss = True
mf.kernel()

