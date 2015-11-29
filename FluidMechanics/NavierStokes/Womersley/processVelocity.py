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


#=================================================================
# C o n t r o l   P a n e l
#=================================================================
numberOfProcessors = 8
woNums = ['10.0']
meshName = 'hexCylinder140'

axialComponent = 1
pOffset = 0.0
amplitude = 1.0
R = 0.5
length = 10.016782
period = math.pi/2.
density = 1.0
cmfeStartTime = 0.0
cmfeStopTime = 2.*period + 0.0000001
cmfeTimeIncrement = period/10.
cmfeOutputFrequency = 1
path = "./output/"
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
def tecplot_WriteRectilinearMesh(filename, X, Y, Z, vars):
    def pad(s, width):
        s2 = s
        while len(s2) < width:
            s2 = ' ' + s2
        if s2[0] != ' ':
            s2 = ' ' + s2
        if len(s2) > width:
            s2 = s2[:width]
        return s2
    def varline(vars, id, fw):
        s = ""
        for v in vars:
            s = s + pad(str(v[1][id]),fw)
        s = s + '\n'
        return s
 
    fw = 10 # field width
 
    f = open(filename, "wt")
 
    f.write('Variables="X","Y"')
    if len(Z) > 0:
        f.write(',"Z"')
    for v in vars:
        f.write(',"%s"' % v[0])
    f.write('\n\n')
 
    f.write('Zone I=' + pad(str(len(X)),6) + ',J=' + pad(str(len(Y)),6))
    if len(Z) > 0:
        f.write(',K=' + pad(str(len(Z)),6))
    f.write(', F=POINT\n')
 
    if len(Z) > 0:
        id = 0
        for k in xrange(len(Z)):
            for j in xrange(len(Y)):
                for i in xrange(len(X)):
                    f.write(pad(str(X[i]),fw) + pad(str(Y[j]),fw) + pad(str(Z[k]),fw))
                    f.write(varline(vars, id, fw))
                    id = id + 1
    else:
        id = 0
        for j in xrange(len(Y)):
            for i in xrange(len(X)):
                f.write(pad(str(X[i]),fw) + pad(str(Y[j]),fw))
                f.write(varline(vars, id, fw))
                id = id + 1
 
    f.close()

def vtkWriteAscii(filename, geometryData, velocityData, elementData):
    totalNumberOfNodes = geometryData.shape[0]
    totalNumberOfElements = elementData.shape[0]

    elementTransform = [0,2,8,6,18,20,26,24,1,5,7,3,19,23,25,21,9,11,17,15,12,14,10,16,4,22,13]

    f = open(filename, "wt")
    f.write('# vtk DataFile Version 2.0\n')
    f.write(filename + '\n')
    f.write('ASCII\n')
    f.write('DATASET UNSTRUCTURED_GRID\n')

    f.write('POINTS ' + str(totalNumberOfNodes) + ' float\n')
    for node in range(totalNumberOfNodes):
        f.write(str(geometryData[node,0]) + ' ' + str(geometryData[node,1]) + ' ' + str(geometryData[node,2]) + '\n')

    f.write('CELLS ' + str(totalNumberOfElements) + ' ' + str(27*totalNumberOfElements+totalNumberOfElements) +'\n')
    for element in range(totalNumberOfElements):
        line = '27 '
        for node in range(27):
            if node > 0:
                line += ' '
            line += str(int(elementData[element,elementTransform[node]]))
        line += '\n'
        f.write(line)

    f.write('CELL_TYPES  1\n29\n')

    f.write('POINT_DATA '+ str(totalNumberOfNodes) + '\n')
    f.write('VECTORS  Velocity float\n')
    for node in range(totalNumberOfNodes):
        f.write(str(velocityData[node,0]) + ' ' + str(velocityData[node,1]) + ' ' + str(velocityData[node,2]) + '\n')    

    f.close()


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

numberOfTimesteps = int((cmfeStopTime - cmfeStartTime)/(cmfeTimeIncrement*cmfeOutputFrequency)) + 1
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
    filename = path + 'Wom' + woNums[0] + 'Dt' + str(round(cmfeTimeIncrement,5)) + meshName + '/TIME_STEP_0000.part0.exnode'
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

    nodeData = numpy.zeros([len(woNums),numberOfTimesteps,totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
    analyticData = numpy.zeros([len(woNums),numberOfTimesteps,totalNumberOfNodes])
    poiseuilleData = numpy.zeros([len(woNums),numberOfTimesteps,totalNumberOfNodes])

    w = 0
    for wo in woNums:
        print('Reading data for Womersley #: ' + wo)  
        for timestep in range(numberOfTimesteps):
            print('Reading data for timestep: ' + str(timestep*cmfeOutputFrequency))  
            for proc in range(numberOfProcessors):
                filename = path + 'Wom' + wo + 'Dt' + str(round(cmfeTimeIncrement,5)) +  meshName + '/TIME_STEP_' + str(timestep*cmfeOutputFrequency).zfill(4) + '.part' + str(proc) +'.exnode'
                print(filename)
                importNodeData = numpy.zeros([totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
                readExnodeFile(filename,field,importNodeData,totalNumberOfNodes)
                nodeData[w,timestep,:,:,:] += importNodeData[:,:,:]
        w += 1

    w = 0
    for wo in woNums:
        viscosity = density/(float(wo)**2.0)
        print('Calculating analytic values for Womersley #: ' + wo)  
        for timestep in range(numberOfTimesteps):
            for node in range(totalNumberOfNodes):
                sumPositionSq = 0.0
                for component in range(3):
                    if component != axialComponent:
                        value = nodeData[w,timestep,node,0,component]
                        sumPositionSq += value**2
                r=math.sqrt(sumPositionSq)                
                analyticData[w,timestep,node] = womersleyAnalytic.womersleyAxialVelocity(timestep*cmfeTimeIncrement*cmfeOutputFrequency,
                                                                                         pOffset,amplitude,R,r,period,
                                                                                         viscosity,float(wo),length)
#                print('timestep: ' + str(timestep) + ' node: ' + str(node))
#                print(analyticData[w,timestep,node])
                poiseuilleData[w,timestep,node] = womersleyAnalytic.poiseuilleAxialVelocity(timestep*cmfeTimeIncrement*cmfeOutputFrequency,amplitude,
                                                                                            period,length,viscosity,r,R)
        w += 1

    nodeIds = []
    nodeIds = [i for i in range(totalNumberOfNodes)]
#    numpy.set_printoptions(threshold='nan')
#    print(poiseuilleData[0,2,:])

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
        for timestep in range(numberOfTimesteps):
            print('Writing combined velocity data file: ')
            filename = 'data' + str(timestep) + '.dat.gz'
            print(filename)
            f = gzip.open(filename,"w")
            f.write('velocity velocity velocity\n')
            exportData = numpy.zeros((totalNumberOfNodes,3))
            exportData = nodeData[timestep,:,1,0:3]
            numpy.savetxt(f,exportData)
            f.close

#print(analyticData[0,7,:])

if exportVelocity:
    for timestep in range(numberOfTimesteps):
        dataFilename = 'data' + str(timestep) + '.dat.gz'
        outputVelFilename = 'VelTime.' + str(timestep) + '.vtu'
        print("Writing velocity data: " + outputVelFilename)
        command = "vmtkmeshdatareader -unnormalize 0 -filetype variabledata -datafile " + dataFilename + " -ifile " + vtuMesh + " -ofile " + outputVelFilename
        print(command)
        os.system(command)

if exportWSS:
    for timestep in range(numberOfTimesteps):
        outputVelFilename = 'VelTime.' + str(timestep) + '.vtu'
        outputWssFilename = 'WssTime.' + str(timestep) + '.vtp'
        print("Writing WSS data: " + outputWssFilename)
        command = "vmtkmeshwallshearrate -quadratureorder 5 -velocityarray velocity " +  "-ifile " + outputVelFilename + " -ofile " + outputWssFilename
        os.system(command)
    

plotData = False
if plotData:
    cycles = [i for i in range(numberOfFullCycles)]
    #pylab.plot(cycles,rmsCycle)
    pylab.bar(cycles,rmsCycle)
    pylab.xlabel('cycle')
    pylab.ylabel('RMS velocity component difference over cycle (cm/s)')
    pylab.title('Cycle Convergence ')
    pylab.grid(True)
    pylab.savefig('cycleConvergence')
    pylab.show()

    #times = [i*cmfeTimeIncrement for i in range(numberOfStepsInCycle)]
    times = [i*cmfeTimeIncrement*cmfeOutputFrequency for i in range(numberOfTimesteps)]
    pylab.plot(times,rmsPcv,'b-')
    pylab.plot(times,rmsPcv/meanVelocity,'g-')
    pylab.xlabel('Time')
    #pylab.ylabel('Percent RMS Difference CFD vs. PCV (%)')
    pylab.ylabel('RMS Difference CFD vs. PCV (cm/s)')
    pylab.title('RMS PCV vs. CFD')
    pylab.grid(True)
    pylab.savefig('cfdVsPcv')
    pylab.show()

plotTime = True
timestep = 75
w=0
if plotTime:
#for timestep in range(10):
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
    i = 0
    for node in centerLineNodes:
        print(nodeData.shape)
        x[i] = nodeData[w,timestep,node,0,0]
        analytic[i] = analyticData[w,timestep,node]
        numeric[i] = nodeData[w,timestep,node,1,1]
        poiseuille[i] = poiseuilleData[w,timestep,node]
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

    fig = plt.figure()
#    ana, pos, num = plt.plot(x,analytic,'b-', x,poiseuille,'r-', x, numeric,'go')
    ana, num = plt.plot(x,analytic,'b-',x, numeric,'go')
#    fig.legend((ana, pos, num), ('analytic', 'poiseuille', 'numeric'), 'upper right')
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
        for node in centerLineNodes:
            x[i] = nodeData[w,timestep,node,0,0]
            analytic[i] = analyticData[w,timestep,node]
            numeric[i] = nodeData[w,timestep,node,1,1]
            poiseuille[i] = poiseuilleData[w,timestep,node]
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
                                   frames=numberOfTimesteps, interval=100, blit=True)

    plt.show()

analyseResults = True
onlyCenterFace = True
if analyseResults:
    #Check l2 norm of errors against analytic values
    for wo in range(len(woNums)):
        l2Errors = []
        RmsErrors = []
        maxError = 0.0
        for timestep in range(numberOfTimesteps):
            if onlyCenterFace:
                anaData = numpy.zeros([numberOfCenterFaceNodes,3])
                numData = numpy.zeros([numberOfCenterFaceNodes,3])
                error = numpy.zeros([numberOfCenterFaceNodes,3])
                i = 0
                for node in centerFaceNodes:
                    anaData[i,1] = analyticData[wo,timestep,node]
                    numData[i,:] = nodeData[wo,timestep,node,1,0:3]
                    error[i,:] = numData[i,:] - anaData[i,:]
                    i += 1
            else:
                anaData = numpy.zeros([totalNumberOfNodes,3])
                numData = numpy.zeros([totalNumberOfNodes,3])
                error = numpy.zeros([totalNumberOfNodes,3])
                anaData[:,1] = analyticData[wo,timestep,:]
                numData = nodeData[wo,timestep,:,1,0:3]
                error = numData - anaData
            # print('numeric')
            # print(numData)
            # print('analytic')
            # print(anaData)
            # print('analyticData')
            # print(analyticData[wo,timestep,:])
            # print('error')
            # print(error)
#            print('timestep # ' + str(timestep))
            l2Errors.append(linalg.norm(error,2))
            RmsErrors.append(numpy.sqrt(numpy.mean(error**2)))
#            print(l2Errors)
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
        print('   Max velocity     : ' + str(maxVel))
        print('   Max L2-norm error/max(|u|): ' + str(maxError/maxVel*100))
        print('   l2 Errors: ')
        print(l2Errors)
        print('   Rms Errors: ')
        print(RmsErrors)
        print('=========================================================')
    
    
