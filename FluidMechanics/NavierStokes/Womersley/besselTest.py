#!/usr/bin/env python

import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy
import cmath
import math
import scipy
import matplotlib
from matplotlib import pyplot as plt
from scipy.special import jn, jn_zeros


def besselManual(x):
    J0 = 1 - (x**2)/(2**2) + (x**4)/(2**2*4**2) - (x**6)/(2**2*4**2*6**2)
    return(J0)

gamma = 1j**(3./2.)*0.5
gamma = 0.
bessSci = jn(0,(gamma))
bessMan = besselManual(gamma)

print(bessSci)
print(bessMan)

plot = False
if plot:
    fig = plt.figure()
    plt.plot(x,analytic,'b-', x,poiseuille,'r-', x, numeric,'go')
    fig.legend((ana, pos, num), ('analytic', 'poiseuille', 'numeric'), 'upper right')
    plt.show()
