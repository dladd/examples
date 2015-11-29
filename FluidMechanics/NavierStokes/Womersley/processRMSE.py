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
numberOfProcessors = 1
woNums = ['10.0']
meshName = 'hexCylinder140'

# Choose which timesteps to plot and their colours
timesteps = [1,2,3,4,5,6,7,8,9,10]
lineColours = ['b','r','c','y','b','r','c','m','g','k']

axialComponent = 1
pOffset = 0.0
amplitude = 1.0
radius = 0.5
length = 10.016782
period = math.pi/2.
density = 1.0
cmfeStartTime = 0.0
cmfeStopTime = period + 0.0000001
cmfeTimeIncrements = [period/400.]
#cmfeTimeIncrements = [period/20.,period/50.,period/100,period/400.]
#cmfeOutputFrequencies = [2,5,10,40]
cmfeOutputFrequencies = [20]
path = "./output/"
#path = "/media/F0F095D5F095A300/opencmissStorage/Womersley/"
vtuMesh = "/hpc/dlad004/opencmiss/examples/FluidMechanics/NavierStokes/Womersley/input/pyformexMesh/hexCylinder12QuadDef.vtu"
nodeFile = "./input/" + meshName + "/" + meshName + ".C"
elementFile = "./input/" + meshName + "/" + meshName + ".M"
centerLineNodeFile = "./input/" + meshName + "/bc/centerlineNodes.dat"
centerFaceNodeFile = "./input/" + meshName + "/bc/centerNodes.dat"
readDataFromExnode = True
exportDataFile = False
exportVelocity = False
exportWSS = False

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



#=================================================================
# M a i n   P r o g r a m
#=================================================================
numberOfTimesteps = []
t = 0
for cmfeTimeIncrement in cmfeTimeIncrements:
    numberOfTimesteps.append(int((cmfeStopTime - cmfeStartTime)/(cmfeTimeIncrement*cmfeOutputFrequencies[t])) + 1)
    t += 1

print('number of timesteps: ' + str(numberOfTimesteps))

nodeData = numpy.zeros([0,0,0,0,0])
analyticData = numpy.zeros([0,0,0])

try:
    with open(nodeFile):
        f = open(nodeFile,"r")
        line=f.readline()
        s = re.findall(r'\d+', line)
        totalNumberOfNodes = int(s[0])
        f.close()
except IOError:
    print ('Could not open file: ' + nodeFile)

filename=centerLineNodeFile
try:
    with open(filename):
        f = open(filename,"r")
        numberOfCenterLineNodes=int(f.readline())
        centerLineNodes=map(int,(re.split(',',f.read())))
        centerLineNodes= [x-1 for x in centerLineNodes]
        f.close()
except IOError:
   print ('Could not open center node file: ' + filename)

filename=centerFaceNodeFile
try:
    with open(filename):
        f = open(filename,"r")
        numberOfCenterFaceNodes=int(f.readline())
        centerFaceNodes=map(int,(re.split(',',f.read())))
        centerFaceNodes= [x-1 for x in centerFaceNodes]
        f.close()
except IOError:
   print ('Could not open center node file: ' + filename)

if readDataFromExnode:
    field = fieldInfo()
    filename = path + 'Wom' + woNums[0] + 'Dt' + str(round(cmfeTimeIncrements[0],5)) + meshName + '/TIME_STEP_0000.part0.exnode'
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

    nodeData = numpy.zeros([len(woNums),len(cmfeTimeIncrements),max(numberOfTimesteps),totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
    analyticData = numpy.zeros([len(woNums),len(cmfeTimeIncrements),max(numberOfTimesteps),totalNumberOfNodes])
    poiseuilleData = numpy.zeros([len(woNums),len(cmfeTimeIncrements),max(numberOfTimesteps),totalNumberOfNodes])

    w = 0
    maxCourant = []
    for wo in woNums:
        t = 0
        print('Reading data for Womersley #: ' + wo)  
        for cmfeTimeIncrement in cmfeTimeIncrements:
            print('   Time resolution: ' + str(cmfeTimeIncrement))
            for timestep in range(numberOfTimesteps[t]):
                print('Reading data for timestep: ' + str(timestep*cmfeOutputFrequencies[t]))  
                for proc in range(numberOfProcessors):
                    filename = path + 'Wom' + wo + 'Dt' + str(round(cmfeTimeIncrement,5)) +  meshName + '/TIME_STEP_' + str(timestep*cmfeOutputFrequencies[t]).zfill(4) + '.part' + str(proc) +'.exnode'
                    print(filename)
                    importNodeData = numpy.zeros([totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
                    readExnodeFile(filename,field,importNodeData,totalNumberOfNodes)
                    nodeData[w,t,timestep,:,:,:] += importNodeData[:,:,:]

            filename = path + 'Wom' + wo + 'Dt' + str(round(cmfeTimeIncrement,5)) +  meshName + '/Womersley_DT' + str(int(round(period/cmfeTimeIncrements[t],-1))) + 'H140N' + str(numberOfProcessors) +'.out'
            maxCourant.append(0.0)
            try:
                with open(filename):
                    f = open(filename,"r")
                    endOfFile = False
                    while endOfFile == False:
                        line=f.readline()
                        if line:
                            if 'Max Courant #' in line:
                                line = line.replace('Max Courant #       =','')
                                line = line.strip()
                                value = float(line)   
                                if maxCourant[t] < value:
                                    maxCourant[t] = value
                        else:
                            endOfFile = True
                            f.close()
            except IOError:
                print ('Could not open file: ' + filename)

            t += 1
        w += 1

    w = 0
    for wo in woNums:
        t = 0
        viscosity = density/(float(wo)**2.0)
        print('Calculating analytic values for Womersley #: ' + wo)  
        for cmfeTimeIncrement in cmfeTimeIncrements:
            for timestep in range(numberOfTimesteps[t]):
                for node in range(totalNumberOfNodes):
                    sumPositionSq = 0.0
                    for component in range(3):
                        if component != axialComponent:
                            value = nodeData[w,t,timestep,node,0,component]
                            sumPositionSq += value**2
                    r=math.sqrt(sumPositionSq)                
                    analyticData[w,t,timestep,node] = womersleyAnalytic.womersleyAxialVelocity(timestep*cmfeTimeIncrement*cmfeOutputFrequencies[t],
                                                                                             pOffset,amplitude,radius,r,period,
                                                                                             viscosity,float(wo),length)
    #                print('timestep: ' + str(timestep) + ' node: ' + str(node))
    #                print(analyticData[w,timestep,node])
                    poiseuilleData[w,t,timestep,node] = womersleyAnalytic.poiseuilleAxialVelocity(timestep*cmfeTimeIncrement*cmfeOutputFrequencies[t],amplitude,
                                                                                                  period,length,viscosity,r,radius)
            t += 1
        w += 1

    nodeIds = []
    nodeIds = [i for i in range(totalNumberOfNodes)]

    try:
        with open(elementFile):
            f = open(elementFile,"r")
            line=f.readline()
            s = re.findall(r'\d+', line)
            numberOfElements = int(s[0])
            elementData = numpy.zeros((numberOfElements,27))
            f.close()
    except IOError:
        print ('Could not open file: ' + elementFile)


    if exportDataFile:
        w=0
        for wo in range(len(woNums)):
            t = 0
            for cmfeTimeIncrement in cmfeTimeIncrements:
                for timestep in range(numberOfTimesteps[t]):
                    print('Writing combined velocity data file: ')
                    filename = 'dataDt' + str(int(round(period/cmfeTimeIncrements[t],-1))) + 'T' + str(timestep) + '.dat.gz'
                    #filename = 'data' + str(timestep) + '.dat.gz'
                    print(filename)
                    f = gzip.open(filename,"w")
                    f.write('velocity velocity velocity\n')
                    exportData = numpy.zeros((totalNumberOfNodes,3))
                    exportData = nodeData[w,t,timestep,:,1,0:3]
                    numpy.savetxt(f,exportData)
                    f.close
                t+=1
            w+=1


if exportVelocity:
    w=0
    for wo in range(len(woNums)):
        t = 0
        for cmfeTimeIncrement in cmfeTimeIncrements:
            for timestep in range(numberOfTimesteps[t]):
                dataFilename = 'dataDt' + str(int(round(period/cmfeTimeIncrements[t],-1))) + 'T' + str(timestep) + '.dat.gz'
                outputVelFilename = 'VelDt' + str(int(round(period/cmfeTimeIncrements[t],-1))) + 'T' + str(timestep) + '.vtu'
                print("Writing velocity data: " + outputVelFilename)
                command = "vmtkmeshdatareader -unnormalize 0 -filetype variabledata -datafile " + dataFilename + " -ifile " + vtuMesh + " -ofile " + outputVelFilename
                print(command)
                os.system(command)
            t += 1
        w+=1

if exportWSS:
    for timestep in range(numberOfTimesteps):
        outputVelFilename = 'VelTime.' + str(timestep) + '.vtu'
        outputWssFilename = 'WssTime.' + str(timestep) + '.vtp'
        print("Writing WSS data: " + outputWssFilename)
        command = "vmtkmeshwallshearrate -quadratureorder 5 -velocityarray velocity " +  "-ifile " + outputVelFilename + " -ofile " + outputWssFilename
        os.system(command)
    

plotTime = True
#timestep = 5
w=0
t = 0
if plotTime:    
    x = numpy.zeros((len(centerLineNodes)))
    analytic = numpy.zeros((len(centerLineNodes)))
    numeric = numpy.zeros((len(centerLineNodes)))
    poiseuille = numpy.zeros((len(centerLineNodes)))
    a = numpy.zeros((len(centerLineNodes),2))
    n = numpy.zeros((len(centerLineNodes),2))
    p = numpy.zeros((len(centerLineNodes),2))
    aSort = numpy.zeros((len(centerLineNodes),2))
    nSort = numpy.zeros((len(centerLineNodes),2))
    pSort = numpy.zeros((len(centerLineNodes),2))
    fig = plt.figure()
    print(nodeData.shape)
    tIndex = 0
    for timestep in timesteps:
        i = 0
        for node in centerLineNodes:
            x[i] = nodeData[w,t,timestep,node,0,0]
            analytic[i] = analyticData[w,t,timestep,node]
            numeric[i] = nodeData[w,t,timestep,node,1,1]
            poiseuille[i] = poiseuilleData[w,t,timestep,node]
            a[i,0] = x[i]
            a[i,1] = analytic[i]
            n[i,0] = x[i]
            n[i,1] = numeric[i]
            p[i,0] = x[i]
            p[i,1] = poiseuille[i]
            i += 1

        aSort = a[a[:,0].argsort()]
        nSort = n[n[:,0].argsort()]
        pSort = p[p[:,0].argsort()]

        x = aSort[:,0]
        analytic = aSort[:,1]
        numeric = nSort[:,1]
        poiseuille = pSort[:,1]

        ana, num = plt.plot(x,analytic,lineColours[tIndex],x, numeric,lineColours[tIndex]+'o')
        tIndex+=1

    fig.legend((ana,  num), ('analytic', 'numeric'), 'upper right')
    plt.show()


animateResults = True
if animateResults:
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.5, 0.5), ylim=(-0.05, 0.05))
    linelist = []

    for i in range(2):
        if i ==0:
            linestyle = 'b-'
        elif i ==1:
            linestyle = 'go'
        else:
            linestyle = 'r--'            
        newline, = ax.plot([], [], linestyle, lw=2)
        linelist.append(newline)

    # initialization function: plot the background of each frame
    def init():
        for line in linelist:
            line.set_data([], [])
        return linelist

    # animation function.  This is called sequentially
    def animate(timestep):
        x = numpy.zeros((len(centerLineNodes)))
        analytic = numpy.zeros((len(centerLineNodes)))
        poiseuille = numpy.zeros((len(centerLineNodes)))
        numeric = numpy.zeros((len(centerLineNodes)))
        a = numpy.zeros((len(centerLineNodes),2))
        n = numpy.zeros((len(centerLineNodes),2))
        p = numpy.zeros((len(centerLineNodes),2))
        aSort = numpy.zeros((len(centerLineNodes),2))
        nSort = numpy.zeros((len(centerLineNodes),2))
        pSort = numpy.zeros((len(centerLineNodes),2))
        i = 0
        w = 0
        t = 0
        for node in centerLineNodes:
            x[i] = nodeData[w,t,timestep,node,0,0]
            analytic[i] = analyticData[w,t,timestep,node]
            numeric[i] = nodeData[w,t,timestep,node,1,1]
            poiseuille[i] = poiseuilleData[w,t,timestep,node]
            a[i,0] = x[i]
            a[i,1] = analytic[i]
            n[i,0] = x[i]
            n[i,1] = numeric[i]
            p[i,0] = x[i]
            p[i,1] = poiseuille[i]
            i += 1
        aSort = a[a[:,0].argsort()]
        nSort = n[n[:,0].argsort()]
        pSort = p[p[:,0].argsort()]

        x = aSort[:,0]
        analytic = aSort[:,1]
        numeric = nSort[:,1]
        poiseuille = pSort[:,1]

        linelist[0].set_data(x, analytic)
        linelist[1].set_data(x, numeric)
#        linelist[2].set_data(x, poiseuille)
        return linelist

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=numberOfTimesteps[t], interval=100, blit=True)

    plt.show()

plot3D = False
if plot3D:
    w = 0
    for wo in woNums:
        t = 0
        for cmfeTimeIncrement in cmfeTimeIncrements:
            for timestep in range(numberOfTimesteps[t]):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

#                anaData = numpy.zeros((numberOfCenterFaceNodes,3))
                numData = numpy.zeros((numberOfCenterFaceNodes,3))
                xNum = numpy.zeros((numberOfCenterFaceNodes))
                zNum = numpy.zeros((numberOfCenterFaceNodes))

                gridRes = [150,150]
                anaData = numpy.zeros((gridRes[1],gridRes[0]))
                # create supporting points in polar coordinates
                r = numpy.linspace(0,0.5,gridRes[0])
                p = numpy.linspace(0,2*numpy.pi,gridRes[1])
                R,P = numpy.meshgrid(r,p)
                # transform them to cartesian system
                X,Z = R*numpy.cos(P),R*numpy.sin(P)

                # Z = ((R**2 - 1)**2)
                # x = z = numpy.arange(-0.5,0.5,1./float(gridRes))
                # X, Z = numpy.meshgrid(x, z)
                # R = numpy.sqrt(X**2 + Z**2)

                for i in range(gridRes[1]):
                    for j in range(gridRes[0]):
                        rLocation = math.sqrt(X[i,j]**2 + Z[i,j]**2)

                        anaData[i,j] = womersleyAnalytic.womersleyAxialVelocity(timestep*cmfeTimeIncrement*cmfeOutputFrequencies[t],
                                                                                pOffset,amplitude,radius,rLocation,period,
                                                                                viscosity,float(wo),length)
                i = 0
                for node in centerFaceNodes:
                    xNum[i] = nodeData[w,t,timestep,node,0,0]
                    zNum[i] = nodeData[w,t,timestep,node,0,2]
                    numData[i,:] = nodeData[w,t,timestep,node,1,0:3]
                    i += 1

                print("Time : " + str(timestep))
                ax.set_xlim(-0.5,0.5)
                ax.set_ylim(-0.5,0.5)
                ax.set_zlim(-0.04,0.04)
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_zlabel('velocity (m/s)')

                ax.scatter(xNum,zNum, numData[:,1],c='g',marker='o',zorder=1.0)
#                ax.plot_surface(X, Z, anaData, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=0)
                ax.plot_wireframe(X, Z, anaData, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.08, antialiased=True, zorder=0)

#                ax.plot_surface(xNum, zNum, anaData[:,1], rstride=1, cstride=1)
                plt.show()
            t += 1
        w += 1


analyseResults = True
onlyCenterFace = True
if analyseResults:
    #Check l2 norm of errors against analytic values
    for wo in range(len(woNums)):
        t = 0
        for cmfeTimeIncrement in cmfeTimeIncrements:
            l2Errors = []
            RmsErrors = []
            maxError = 0.0
            for timestep in range(numberOfTimesteps[t]):
                if onlyCenterFace:
                    anaData = numpy.zeros([numberOfCenterFaceNodes,3])
                    numData = numpy.zeros([numberOfCenterFaceNodes,3])
                    error = numpy.zeros([numberOfCenterFaceNodes,3])
                    i = 0
                    for node in centerFaceNodes:
                        anaData[i,1] = analyticData[wo,t,timestep,node]
                        numData[i,:] = nodeData[wo,t,timestep,node,1,0:3]
                        error[i,:] = numData[i,:] - anaData[i,:]
                        i += 1
                else:
                    anaData = numpy.zeros([totalNumberOfNodes,3])
                    numData = numpy.zeros([totalNumberOfNodes,3])
                    error = numpy.zeros([totalNumberOfNodes,3])
                    anaData[:,1] = analyticData[wo,t,timestep,:]
                    numData = nodeData[wo,t,timestep,:,1,0:3]
                    error = numData - anaData
                l2Errors.append(linalg.norm(error,2))
                RmsErrors.append(numpy.sqrt(numpy.mean(error**2)))
                if l2Errors[timestep] > maxError:
                    maxError = l2Errors[timestep]
                    errorIndex = timestep
                    velMag = numpy.zeros((totalNumberOfNodes))
                    velMag = numpy.apply_along_axis(linalg.norm,1,numData[:,:])
                    maxVel = max(velMag)

                    for node in range(totalNumberOfNodes):
                        if velMag[node] == maxVel:
                            break

            print('=========================================================')
            print(' Womersley # ' + woNums[wo] + 'time increment: ' + str(cmfeTimeIncrement))
            print('---------------------------------------------------------')
            print('   Max L2-norm error: ' + str(maxError))
            print('   Max RMS error    : ' + str(max(RmsErrors)))
            print('   Max velocity     : ' + str(maxVel))
            print('   Max L2-norm error/max(|u|): ' + str(maxError/maxVel*100))
            print('   Max Courant #    : ' + str(maxCourant[t]))
            print('   l2 Errors: ')
            print(l2Errors)
            print('   Rms Errors: ')
            print(RmsErrors)
            print('=========================================================')
    
            t += 1
