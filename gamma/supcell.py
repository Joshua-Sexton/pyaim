#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.tools.pbc import super_cell, cell_plus_imgs

#cell = gto.Cell()
#cell.unit = 'B' 
#cell.a = [[ 4.6298286730500005, 0.0, 0.0], 
#          [-2.3149143365249993, 4.009549246030899, 0.0], 
#          [0.0, 0.0, 40]] 
#cell.atom = '''C 0 0 0  
#               C 0 2.67303283 0'''
#cell.dimension = 2 
#cell.verbose = 4
#cell.basis = 'cc-pvtz'
#cell.build()

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'cc-pvtz'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.precision = 1e-12
cell.build()

#cell = gto.Cell()
#cell.atom = '''C     0.      0.      0.    
#              C     0.8917  0.8917  0.8917
#              C     1.7834  1.7834  0.    
#              C     2.6751  2.6751  0.8917
#              C     1.7834  0.      1.7834
#              C     2.6751  0.8917  2.6751
#              C     0.      1.7834  1.7834
#              C     0.8917  2.6751  2.6751'''
#cell.basis = 'cc-pvtz'
#cell.a = numpy.eye(3)*3.5668
#cell.precision = 1e-12
#cell.verbose = 4
#cell.build()

nmp = [2, 2, 2]
supcell = super_cell(cell, nmp)
lib.logger.info(cell,'* Cell Info')
lib.logger.info(cell,'Cell dimension : %d', supcell.dimension)
lib.logger.info(cell,'Lattice vectors (Bohr)')
for i in range(3):
    lib.logger.info(cell,'Cell a%d axis : %.6f  %.6f  %.6f', i, *supcell.a[i])
lib.logger.info(cell,'Cell volume %g (Bohr^3)', supcell.vol)
lib.logger.info(cell, 'Supcell atoms')
coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in supcell._atom])
for i in range(supcell.natm):
    lib.logger.info(cell,'Nuclei %d position : %.6f  %.6f  %.6f', i, *coords[i])

#nmp = [1, 1, 1]
#lib.logger.info(cell, 'Supcell atoms')
#supcell = cell_plus_imgs(cell, nmp)
#coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in supcell._atom])
#for i in range(supcell.natm):
#    lib.logger.info(cell,'Nuclei %d position : %.6f  %.6f  %.6f', i, *coords[i])
