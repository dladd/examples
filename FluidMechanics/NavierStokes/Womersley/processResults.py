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
import matplotlib.pyplot as plt
from collections import OrderedDict

class fieldInfo(object):
    'base class for info about fields'

    def __init__(self):

        self.group = ''
        self.numberOfFields = 0
        self.fieldNames = []
        self.numberOfFieldComponents = []

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
# C o n t r o l   P a n e l
#=================================================================

ReynoldsNumbers = [100,400,1000,2500,3200,5000]#[100,400,1000,3200,5000]#[100,400,1000,3200,5000]
meshResolution = [60,60]#[40,40]#[80,80]#[20,20]#[50,50]#[200,200]#[100,100]#[40,40]#[20,20] # [10,10]
totalNumberOfNodes =14641#6561#25921#1681#10201#160801 #40401 # 6561 #1681 # 441
compareSolutions = ['ghia.txt','erturk.txt','botella.txt']
compareMarkers = ['ro','gs','b^']
compareNames = ['Ghia','Erturk','Botella']
numberOfProcessors = 16

#=================================================================
#=================================================================


nodeData = numpy.zeros([0,0,0,0])
numberOfRe = len(ReynoldsNumbers)

field = fieldInfo()        
filename = './StaticSolution.part0.exnode'
try:
    with open(filename):
        firstFile = open(filename,"r")
        line=firstFile.readline()
        field.group=findBetween(line, ' Group name: ', '\n')
        readFirstHeader(firstFile,field)
        firstFile.close()
except IOError:
    print ('Could not open file: ' + filename)

nodeData = numpy.zeros([numberOfRe,totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])

i = -1
for Re in ReynoldsNumbers:
    i+=1
    print('Reading data for Re ' + str(Re))
    for proc in range(numberOfProcessors):
        path = "./output/Re" + str(Re) + 'Dim' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + '/'
        filename = path + 'LidDrivenCavity.part' + str(proc) +'.exnode'
        importNodeData = numpy.zeros([totalNumberOfNodes,field.numberOfFields,max(field.numberOfFieldComponents)])
        readExnodeFile(filename,field,importNodeData,totalNumberOfNodes)
        nodeData[i,:,:,:] += importNodeData[:,:,:]
        
tolerance = 1e-8
vlineCount = -1
vLineY = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])
vLineU = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])
plotVLineU = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])
for node in xrange(totalNumberOfNodes):
    if nodeData[0,node,0,0] > (0.5 - tolerance) and nodeData[0,node,0,0] < (0.5 + tolerance):
        vlineCount += 1
        vLineY[:,vlineCount] = nodeData[:,node,0,1]
        vLineU[:,vlineCount] = nodeData[:,node,1,0]

nodeIds = []
nodeIds = [i for i in range(totalNumberOfNodes)]

#compareLocations=numpy.zeros((len(compareSolutions),100))
#compareData=numpy.zeros((len(compareSolutions),len(ReynoldsNumbers),100))
plotCompareData=numpy.zeros((len(compareSolutions),len(ReynoldsNumbers),100))


compareLocations=[[] for i in range(len(compareSolutions))]
#compareData=[[[] for i in range(len(compareSolutions))] for j in range(len(ReynoldsNumbers))]
compareData=[[[] for i in range(len(ReynoldsNumbers))] for j in range(len(compareSolutions))]
#compareData=[[] for i in range(len(compareSolutions))]



k = -1
foundResults = numpy.zeros((len(compareSolutions),len(ReynoldsNumbers)),dtype=numpy.int)
for filename in compareSolutions:
    k+=1
    with open(filename,"r") as f:
        line = f.readline()
        dataInfo = line.strip().split(' ')
        compareAll = numpy.loadtxt(f)
        compareLocations[k].append(compareAll[:,0])
        i = -1
        for Re in ReynoldsNumbers:
            i+=1
            j=0
            for refRe in dataInfo[1:]:
                j+=1
                if float(refRe) == Re:
                    foundResults[k,i] += 1
                    # compareData[publication,Re,yLocation,uValue]
           #         compareData[k,i,:] = compareAll[:,j]
                    compareData[k][i].append(compareAll[:,j].tolist())

plotData = True
labeled = [0]*len(compareNames)
p = [0]*len(compareNames)
if plotData:
    reIndex =-1
    for Re in ReynoldsNumbers:
        reIndex +=1
        for ref in range(len(compareSolutions)):
            if foundResults[ref,reIndex]:
                plotCompareData = numpy.zeros((len(compareData[ref][reIndex])))
                plotCompareData = numpy.array(compareData[ref][reIndex]) - 0.5*reIndex
                #little hack to avoid repeating the symbols in the legend
                if not labeled[ref]:
                    pylab.plot(plotCompareData[0][0],compareLocations[ref][0][0],compareMarkers[ref],label=compareNames[ref])
                    labeled[ref] = 1
                pylab.plot(plotCompareData,compareLocations[ref],compareMarkers[ref])

        plotVLineU[reIndex,:] = vLineU[reIndex,:] - 0.5*reIndex
        pylab.plot(plotVLineU[reIndex,:],vLineY[reIndex,:],'-k')

    ax = pylab.gca()
    ax.xaxis.set_ticklabels([]) #set_visible(False)
    pylab.legend(numpoints=1, shadow = True, loc = (0.85, 0.20))
    pylab.xlabel('x-velocity (non-dimensional)')
    pylab.ylabel('y-coordinate (non-dimensional)')
    pylab.grid(True)
    pylab.savefig('ReAnalysisMesh' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + 'Dofs' + str(totalNumberOfNodes))
    pylab.show()

    
