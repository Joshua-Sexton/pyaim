#!/usr/bin/env python

import sys
sys.path.append('..')
import surf4c

natm = 2
name = 'tlat.chk'
surface = surf4c.BaderSurf(name)
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.verbose = 4
surface.epsiscp = 0.220
surface.mstep = 200
surface.npang = 5810
surface.leb = True
surface.rmaxsurf = 10.0
surface.step = 0.1
for i in range(natm):
    surface.inuc = i
    surface.kernel()

