#!/usr/bin/env python

from pyaim.gamma import aom

name = 'cadena_cas.chk'
natoms = 6

ovlp = aom.Aom(name)
ovlp.verbose = 4
ovlp.nrad = 301
ovlp.iqudr = 'legendre'
ovlp.mapr = 'becke'
ovlp.bnrad = 201
ovlp.betafac = 0.4
ovlp.bnpang = 3074
ovlp.biqudr = 'legendre'
ovlp.bmapr = 'exp'
ovlp.orbs = 6
ovlp.cas = True
ovlp.corr = False
for i in range(natoms):
    ovlp.inuc = i
    ovlp.kernel()
                 
