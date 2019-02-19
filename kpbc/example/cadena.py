#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf
from pyscf import lib
from pyscf import scf as mole_scf
from pyscf.tools import wfn_format

name = 'cadena'

cell = gto.Cell()
cell.atom='''
  H 0.000000000000   0.000000000000   0.000000000000
  H 1.000000000000   0.000000000000   0.000000000000
  H 2.000000000000   0.000000000000   0.000000000000
  H 3.000000000000   0.000000000000   0.000000000000
  H 4.000000000000   0.000000000000   0.000000000000
  H 5.000000000000   0.000000000000   0.000000000000
  H 6.000000000000   0.000000000000   0.000000000000
  H 7.000000000000   0.000000000000   0.000000000000
  H 8.000000000000   0.000000000000   0.000000000000
  H 9.000000000000   0.000000000000   0.000000000000
'''
cell.basis = 'def2-svp'
cell.precision = 1e-12
cell.dimension = 1
cell.a = [[10,0,0],[0,1,0],[0,0,1]]
cell.unit = 'A'
cell.verbose = 4
cell.build()

nk = [8,1,1]
kpts = cell.make_kpts(nk)
kpts -= kpts[0] # Shift to gamma
scf.chkfile.save_cell(cell, name+'.chk')
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

mf = scf.KRHF(cell, kpts).density_fit()
mf.with_df.auxbasis = 'def2-svp-jkfit'
#mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.with_df._cderi_to_save = name+'_eri.h5'
#mf.with_df._cderi = name+'_eri.h5' 
#mf = mole_scf.addons.remove_linear_dep_(mf)
ehf = mf.kernel()

orbs = 1
nprims, nmo = mf.mo_coeff[0].shape 
mo_coeff = numpy.zeros((nprims,orbs*len(kpts)), dtype=numpy.complex128)
mo_occ = numpy.zeros(orbs*len(kpts))
mo_energy = numpy.zeros(orbs*len(kpts))
ii = 0
for k in range(len(kpts)):
    for i in range(orbs):
        mo_coeff[:,ii] = mf.mo_coeff[k][:,i]
        mo_occ[ii] = 2.0
        mo_energy[ii] = 1.0
        ii += 1

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, cell, mo_coeff, mo_occ=mo_occ, mo_energy=mo_energy)

