#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)

NPROPS = 3
PROPS = ['density', 'kinetic', 'laplacian']
NCOL = 15
DIGITS = 5

def print_aom(name, natm, nmo):                
    aom = numpy.zeros((natm,nmo,nmo))
    totaom = numpy.zeros((nmo,nmo))
    with h5py.File(name) as f:
        for i in range(natm):
            idx = 'ovlp'+str(i)
            aom[i] = f[idx+'/aom'].value
    	for i in range(natm):
            log.info('Follow AOM for atom %d', i)
	    dump_tri(sys.stdout, aom[i], ncol=NCOL, digits=DIGITS, start=0)
            totaom += aom[i]
    log.info('Follow total AOM')
    dump_tri(sys.stdout, totaom, ncol=NCOL, digits=DIGITS, start=0)
    i = numpy.identity(nmo)
    diff = numpy.linalg.norm(totaom-i)
    log.info('Diff in S matrix : %8.5f', diff)
    aom_file = name + '.aom'
    with open(aom_file, 'w') as f2:
    	for k in range(natm): # Over atoms == over primitives
            f2.write("%5d <=== AOM within this center\n" % (k+1))
            ij = 0
            for i in range(nmo):
                for j in range(i+1):
                    f2.write(' %16.10f' % (aom[k,i,j]))
                    ij += 1
                    if (ij%6 == 0):
                        f2.write("\n")
            f2.write("\n")

if __name__ == '__main__':
    name = 'cadena_cas.chk.h5'
    natm = 6
    nmo = 6
    print_aom(name,natm,nmo)

