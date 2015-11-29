#!/usr/bin/env python

import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy
import cmath
import math
import scipy
from scipy.special import jn, jn_zeros

def poiseuilleAxialVelocity(time,amplitude,period,length,viscosity,r,R):
    ''' returns analytic solution to a parabolic velocity profile based
    for Poiseuille flow'''

    pOutlet = 0.0
    pInlet = amplitude*math.cos(2.0*math.pi*(time/(period)))
    uAxial = (pInlet-pOutlet)/(4.*viscosity*length)*(R**2-r**2)
    return(uAxial);


def womersleyAxialVelocity(t,pOffset,amplitude,R,r,period,viscosity,alpha,length):
    """ Computes analytic value for axial velocity assuming a Womersley profile """

    zeroTolerance = 1e-6
    angularFrequency = 2.0*math.pi/period
    gamma = 1j**(3./2.)*alpha
    ks = 0.0

    #uWomComplex = (amplitude*R**2.)/(viscosity*(alpha**2.))*1j*(1-((jn(0,(gamma*r/R)))/(jn(0,(gamma)))))*cmath.exp(1j*(angularFrequency*t+math.pi/2.))/length
    uWomComplex = (amplitude*R**2.)/(viscosity*(alpha**2.))*1j*(1-((jn(0,(gamma*r/R)))/(jn(0,(gamma)))))*cmath.exp(1j*(angularFrequency*t))/length
    uAxial = uWomComplex.real
    if abs(pOffset) > zeroTolerance:
        print('using p-offset')
        PoiseuilleAxialVelocity(uOffset,pOffset,viscosity,length,r,R)
        uAxial += uOffset
    return(uAxial);

def womersleyFlowrate(t,pOffset,amplitude,R,period,viscosity,alpha,length):
    """ Computes analytic flowrate assuming a Womersley profile """

    zeroTolerance = 1e-6
    angularFrequency = 2.0*math.pi/period
    gamma = 1j**(3./2.)*alpha
    ks = 0.0 

    #qWomComplex = (1j*math.pi*amplitude*R**4.)/(viscosity*(alpha**2.))*(1-(2.*jn(1,gamma))/(gamma*jn(0,gamma)))*cmath.exp(1j*(angularFrequency*(t)-math.pi/2.))/length
    qWomComplex = (1j*math.pi*amplitude*R**4.)/(viscosity*(alpha**2.))*(1-(2.*jn(1,gamma))/(gamma*jn(0,gamma)))*cmath.exp(1j*(angularFrequency*(t)))/length
    qAxial = qWomComplex.real
    #qAxial = 0.01265*math.sin(angularFrequency*(t)+math.pi/2.)
    return(qAxial);

def womersleyPressure(time,period,pOffset,amplitude,axialPosition,length):
    """ Computes analytic value for axial velocity assuming a Womersley profile """
    frequency = 2.0*math.pi/period
    #pressure = pOffset + amplitude - amplitude*math.cos(frequency*time)*(axialPosition/length)
    pressure = pOffset + amplitude*math.cos(frequency*time)*(1.0-axialPosition/length)
    #pressure = pOffset + amplitude*math.cos(2.0*math.pi*(t/(period)))#+math.pi/2.0)
    return(pressure);

