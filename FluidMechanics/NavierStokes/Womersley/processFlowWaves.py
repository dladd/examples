#!/usr/bin/env python

#> \file
#> \author David Ladd
#>
#> \section LICENSE
#>
#> Version: MPL 1.1/GPL 2.0/LGPL 2.1
#>
#> The contents of this file are subject to the Mozilla Public License
#> Version 1.1 (the "License"); you may not use this file except in
#> compliance with the License. You may obtain a copy of the License at
#> http://www.mozilla.org/MPL/
#>
#> Software distributed under the License is distributed on an "AS IS"
#> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#> License for the specific language governing rights and limitations
#> under the License.
#>
#> The Original Code is openCMISS
#>
#> The Initial Developer of the Original Code is University of Auckland,
#> Auckland, New Zealand and University of Oxford, Oxford, United
#> Kingdom. Portions created by the University of Auckland and University
#> of Oxford are Copyright (C) 2007 by the University of Auckland and
#> the University of Oxford. All Rights Reserved.
#>
#> Contributor(s): 
#>
#> Alternatively, the contents of this file may be used under the terms of
#> either the GNU General Public License Version 2 or later (the "GPL"), or
#> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
#> in which case the provisions of the GPL or the LGPL are applicable instead
#> of those above. if you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. if you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>


# Add Python bindings directory to PATH
import sys, os

import gzip
import numpy
import math
import re
from scipy import interpolate
from numpy import linalg,mean,sqrt
import pylab
import time
import womersleyAnalytic

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import animation

matplotlib.rc('lines', linewidth=2, color='r')

#=================================================================
# C o n t r o l   P a n e l
#=================================================================
numberOfProcessors = 2
woNums = ['10.0']
meshName = 'hexCylinder140'#'hexCylinder140'#'hexCylinder13'#

# Choose which timesteps to plot and their colours
lineColours = ['b','r','g']
lineColours2 = ['c','m','y']

axialComponent = 1
pOffset = 0.0
amplitude = 1.0
radius = 0.5
length = 10.017438
period = math.pi/2.
density = 1.0
cmfeStartTime = 0.0
cmfeStopTime = 1.0*period + 0.0000001
cmfeTimeIncrement = period/1000.
cmfeOutputFrequency = 10
#betas=['0','0.2','1']
betas=['0']
path = "./output/"
meshType = 'Quadratic'
nodeFile = "./input/" + meshName + "/" + meshName + ".C"
elementFile = "./input/" + meshName + "/" + meshName + ".M"
inletNode = 6#1
outletNode = 184#8113#3505#184
readDataFromExnode = True

#=================================================================
# C l a s s e s
#=================================================================
class fieldInfo(object):
    'base class for info about fields'

    def __init__(self):

        self.group = ''
        self.numberOfFields = 0
        self.fieldNames = []
        self.numberOfFieldComponents = []

#=================================================================
# F u n c t i o n s
#=================================================================

def findBetween( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def readFirstHeader(f,info):

    #Read header info
#    print('reading HEADER')
    line=f.readline()
    s = re.findall(r'\d+', line)
    info.numberOfFields = int(s[0])
    for field in range(info.numberOfFields):
        line=f.readline()
        fieldName = findBetween(line, str(field + 1) + ') ', ',')
        info.fieldNames.append(fieldName)
        numberOfComponents = int(findBetween(line, '#Components=', '\n'))
        info.numberOfFieldComponents.append(numberOfComponents)
#        print('  number of components ' + str(numberOfComponents))
        for skip in range(numberOfComponents):
            line=f.readline()

def readExnodeHeader(f,numberOfFieldComponents):

    #Read header info
#    print('reading HEADER')
    line=f.readline()
    s = re.findall(r'\d+', line)
    numberOfFields = int(s[0])
#    print('  number of fields ' + str(numberOfFields))
    for field in range(numberOfFields):
        line=f.readline()
        fieldName = findBetween(line, str(field + 1) + ') ', ',')
        numberOfComponents = int(findBetween(line, '#Components=', '\n'))
        numberOfFieldComponents.append(numberOfComponents)
#        print('  number of components ' + str(numberOfComponents))
        for skip in range(numberOfComponents):
            line=f.readline()
#            print(line)


def readExnodeFile(filename,info,nodeData,totalNumberOfNodes):

    try:
        with open(filename):
            f = open(filename,"r")

            #Read header
            line=f.readline()
            info.group=findBetween(line, ' Group name: ', '\n')
            numberOfFieldComponents = []
            numberOfFields = 0
            readExnodeHeader(f,numberOfFieldComponents)
            numberOfFields = info.numberOfFields
            numberOfFieldComponents = info.numberOfFieldComponents

            #Read node data
            endOfFile = False
            while endOfFile == False:
                previousPosition = f.tell()
                line=f.readline()
                line = line.strip()
                if line:
                    if 'Node:' in line:
                        s = re.findall(r'\d+', line)
                        node = int(s[0])
                        for field in range(numberOfFields):
                            for component in range(numberOfFieldComponents[field]):
                                line=f.readline()
                                line = line.strip()
                                value = float(line)
                                if abs(value - 1.2345678806304932) < 1.0e-6:
                                    value =0.0
                                nodeData[node-1,field,component] = value

                    elif 'Fields' in line:
                        f.seek(previousPosition)
                        numberOfFieldComponents = []
                        numberOfFields = 0
                        readExnodeHeader(f,numberOfFieldComponents)
                        numberOfFields = len(numberOfFieldComponents)

                else:
                    endOfFile = True
                    f.close()
    except IOError:
       print ('Could not open file: ' + filename)

def readExnodeFileNodes(nodes,filename,info,nodeData,totalNumberOfNodes):

    try:
        with open(filename):
            f = open(filename,"r")

            #Read header
            line=f.readline()
            info.group=findBetween(line, ' Group name: ', '\n')
            numberOfFieldComponents = []
            numberOfFields = 0
            readExnodeHeader(f,numberOfFieldComponents)
            numberOfFields = info.numberOfFields
            numberOfFieldComponents = info.numberOfFieldComponents

            #Read node data
            endOfFile = False
            while endOfFile == False:
                previousPosition = f.tell()
                line=f.readline()
                line = line.strip()
                if line:
                    if 'Node:' in line:
                        s = re.findall(r'\d+', line)
                        node = int(s[0])
                        if (node in nodes):
                            for field in range(numberOfFields):
                                for component in range(numberOfFieldComponents[field]):
                                    line=f.readline()
                                    line = line.strip()
                                    value = float(line)
                                    if abs(value - 1.2345678806304932) < 1.0e-6:
                                        value =0.0
                                    nodeData[nodes.index(node),field,component] = value

                    elif 'Fields' in line:
                        f.seek(previousPosition)
                        numberOfFieldComponents = []
                        numberOfFields = 0
                        readExnodeHeader(f,numberOfFieldComponents)
                        numberOfFields = len(numberOfFieldComponents)

                else:
                    endOfFile = True
                    f.close()
    except IOError:
       print ('Could not open file: ' + filename)


#=================================================================
# M a i n   P r o g r a m
#=================================================================
numberOfTimesteps=(int((cmfeStopTime - cmfeStartTime)/(cmfeTimeIncrement*float(cmfeOutputFrequency))) + 1)
print('number of timesteps: ' + str(numberOfTimesteps))

inputDir = './input/' + meshName +'/'
#Inlet boundary nodes
filename=inputDir + 'bc/inletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletNodes=int(f.readline())
        inletNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary node file: ' + filename)
#Outlet boundary nodes
filename=inputDir + 'bc/outletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfOutletNodes=int(f.readline())
        outletNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Outlet boundary node file: ' + filename)


try:
    with open(nodeFile):
        f = open(nodeFile,"r")
        line=f.readline()
        s = re.findall(r'\d+', line)
        totalNumberOfNodes = int(s[0])
        f.close()
except IOError:
    print ('Could not open file: ' + nodeFile)

if readDataFromExnode:
    field = fieldInfo()
    if (meshName == 'hexCylinder13'):
        totalNumberOfNodes = 3577
    filename = path + 'Wom' + woNums[0] + 'Dt' + str(round(cmfeTimeIncrement,5)) + meshName + '_RBS_Quadratic_FixOutlet_Beta0_Init_RHS/TimeStep_0.part0.exnode'
    print(filename)
    try:
        with open(filename):
            firstFile = open(filename,"r")
            line=firstFile.readline()
            field.group=findBetween(line, ' Group name: ', '\n')
            readFirstHeader(firstFile,field)
            firstFile.close()
    except IOError:
        print ('Could not open file: ' + filename)

    nodeData = numpy.zeros([len(betas),numberOfTimesteps,totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
    #nodeData = numpy.zeros([len(betas),numberOfTimesteps,2,field.numberOfFields,max(field.numberOfFieldComponents)])
    analyticData = numpy.zeros([numberOfTimesteps])
    analyticPressureData = numpy.zeros([numberOfTimesteps])

    b = 0
    for beta in betas:
        t = 0
        print('Reading data for beta: ' + beta)  
        for timestep in range(numberOfTimesteps):
            for proc in range(numberOfProcessors):
                filename = path + 'Wom' + woNums[0] + 'Dt' + str(round(cmfeTimeIncrement,5)) +  meshName + '_RBS_Quadratic_FixOutlet_Beta'+beta+'_Init_RHS/TimeStep_' + str(timestep*cmfeOutputFrequency) + '.part' + str(proc) +'.exnode'
                importNodeData = numpy.zeros([totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
                #importNodeData = numpy.zeros([2,field.numberOfFields,max(field.numberOfFieldComponents)])
                readExnodeFile(filename,field,importNodeData,totalNumberOfNodes)
                #readExnodeFileNodes([inletNode,outletNode],filename,field,importNodeData,totalNumberOfNodes)
                nodeData[b,timestep,:,:,:] += importNodeData[:,:,:]
        b += 1

    t = 0
    viscosity = density/(float(woNums[0])**2.0)
    print('Calculating analytic values')
    for timestep in range(numberOfTimesteps):
        analyticData[timestep] = womersleyAnalytic.womersleyFlowrate(timestep*cmfeTimeIncrement*cmfeOutputFrequency,# + math.pi/2.0,
                                                                     pOffset,amplitude,radius,period,
                                                                     viscosity,float(woNums[0]),length)
        analyticPressureData[timestep] = womersleyAnalytic.womersleyPressure(timestep*cmfeTimeIncrement*cmfeOutputFrequency,# + math.pi/2.0,
                                                                             pOffset,amplitude,radius,period,
                                                                             viscosity,float(woNums[0]),length)


plotTime = True
w=0
t = 0
if plotTime:    

    inletFlow = numpy.zeros([len(betas),numberOfTimesteps])
    outletFlow = numpy.zeros([len(betas),numberOfTimesteps])
    inletPressure = numpy.zeros([len(betas),numberOfTimesteps])
    inletBoundaryPressure = numpy.zeros([len(betas),numberOfTimesteps])
    outletBoundaryPressure = numpy.zeros([len(betas),numberOfTimesteps])
    outletPressure = numpy.zeros([len(betas),numberOfTimesteps])
    inletAveragePressure = numpy.zeros([len(betas),numberOfTimesteps])
    outletAveragePressure = numpy.zeros([len(betas),numberOfTimesteps])
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)

    times = numpy.arange(cmfeStartTime,cmfeStopTime,cmfeTimeIncrement*cmfeOutputFrequency)
    b=0
    for beta in betas:
        for timestep in range(numberOfTimesteps):
            inletFlow[b,timestep] = nodeData[b,timestep,inletNode-1,1,0]
            outletFlow[b,timestep] = -1.0*nodeData[b,timestep,outletNode-1,1,0]
            inletPressure[b,timestep] = nodeData[b,timestep,inletNode-1,2,3]
            outletPressure[b,timestep] = nodeData[b,timestep,outletNode-1,2,3]
            totalP = 0.0
            for node in inletNodes:
                totalP += nodeData[b,timestep,node-1,2,3]
            inletAveragePressure[b,timestep] = totalP/len(inletNodes)
            totalP = 0.0
            for node in outletNodes:
                totalP += nodeData[b,timestep,node-1,2,3]
            outletAveragePressure[b,timestep] = totalP/len(outletNodes)

            #inletBoundaryPressure[b,timestep] = nodeData[b,timestep,inletNode-1,1,9]
            #outletBoundaryPressure[b,timestep] = nodeData[b,timestep,outletNode-1,1,9]

            #inletFlow[b,timestep] = nodeData[b,timestep,0,1,8]
            #outletFlow[b,timestep] = -1.0*nodeData[b,timestep,1,1,8]
            #inletPressure[b,timestep] = nodeData[b,timestep,0,2,3]
            #inletBoundaryPressure[b,timestep] = nodeData[b,timestep,0,1,9]
            #outletPressure[b,timestep] = nodeData[b,timestep,1,2,3]

        ax0.plot(times,inletFlow[b,:],lineColours[b],label="inlet, beta="+beta,alpha=0.5)
        ax0.plot(times,outletFlow[b,:],lineColours2[b],label="outlet, beta="+beta,alpha=0.5)
        #ax0.plot(times,inletFlow[b,:],lineColours2[b])

        ax1.plot(times,inletPressure[b,:],lineColours[b],label="inlet point, beta="+beta,alpha=0.5)
        ax1.plot(times,outletPressure[b,:],lineColours2[b],label="outlet point, beta="+beta,alpha=0.5)
        ax1.plot(times,inletAveragePressure[b,:],'--'+lineColours[b],label="inlet average, beta="+beta,alpha=0.5)
        ax1.plot(times,outletAveragePressure[b,:],'--'+lineColours2[b],label="outlet average, beta="+beta,alpha=0.5)
        #ax1.plot(times,inletBoundaryPressure[b,:],'y',label="inlet set pressure, beta="+beta,alpha=0.5)
        #ax1.plot(times,outletBoundaryPressure[b,:],'g',label="outlet set pressure, beta="+beta,alpha=0.5)

        b+=1

    ax0.plot(times,analyticData[:],'k',label="analytic",alpha=0.5)
    ax1.plot(times,analyticPressureData[:],'k',label="analytic",alpha=0.5)
    #ax0.legend(loc='upper right', shadow=True, fancybox=True)#, prop=props)
    #ax1.legend(loc='upper right', shadow=True, fancybox=True)#, prop=props)
    # ax.xlabel('time (s)')
    # ax.ylabel('flow rate (\cmthreeps)')
    # ax.title('Boundary-stabilised Womersley flow rate')
    plt.show()

