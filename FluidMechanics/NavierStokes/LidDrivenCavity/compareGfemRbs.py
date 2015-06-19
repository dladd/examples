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

ReynoldsNumbers = [100,100,400,400,1000,1000,2500,2500,3200,3200,5000]#,5000,5000]
#ReynoldsNumbers = [5000]
#ReynoldsNumbers = [100,400,1000,2500,3200,5000]
# 441=10Elem, 1681=20Elem, 3721=30Elem, 6561=40Elem 14641 = 60Elem, 
meshResolution = [20,20]#[80,80]#[20,20]#[50,50]#[200,200]#[100,100]#[40,40]#[20,20] # [10,10]
#totalNumberOfNodes =441#6561#3721#4225#1681#4225#1681#10201 #14641 #6561#25921#1681#10201#160801 #40401 # 6561 #1681 # 441 3721#
totalNumberOfNodes = 1681#441#81#25#1681#441
compareSolutions = ['ghia.txt','erturk.txt','botella.txt']
compareMarkers = ['yo','gs','c^']
compareNames = ['Ghia','Erturk','Botella']
numberOfProcessors = [2,2,2,2,2,2,2,2,2,2,2]#[4,4,4,4,4,4,4,4,4,4,4,4]
SUPG = [True,False,True,False,True,False,True,False,True,False,True]#True,False,True,False]#,True,False]
figsDir = "/hpc/dlad004/thesis/Thesis/figures/cfd/"
writeToFigs = False
plotLegend = False

#=================================================================
#=================================================================


nodeData = numpy.zeros([0,0,0,0])
numberOfRe = len(ReynoldsNumbers)

if any(SUPG):
    fieldSUPG = fieldInfo()        
    path = "./output/Re" + str(ReynoldsNumbers[0]) + 'Elem' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + '_SUPG_temp/'
    filename = path + '/LidDrivenCavity.part0.exnode'
    try:
        with open(filename):
            firstFile = open(filename,"r")
            line=firstFile.readline()
            fieldSUPG.group=findBetween(line, ' Group name: ', '\n')
            readFirstHeader(firstFile,fieldSUPG)
            firstFile.close()
    except IOError:
        print ('Could not open file: ' + filename)
    nodeDataSUPG = numpy.zeros([numberOfRe,totalNumberOfNodes,fieldSUPG.numberOfFields,max(fieldSUPG.numberOfFieldComponents)])

if not all(SUPG):
    fieldGFEM = fieldInfo()        
    path = "./output/Re" + str(ReynoldsNumbers[0]) + 'Elem' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + '_GFEM_temp/'
    filename = path + '/LidDrivenCavity.part0.exnode'
    try:
        with open(filename):
            firstFile = open(filename,"r")
            line=firstFile.readline()
            fieldGFEM.group=findBetween(line, ' Group name: ', '\n')
            readFirstHeader(firstFile,fieldGFEM)
            firstFile.close()
    except IOError:
        print ('Could not open file: ' + filename)
    nodeDataGFEM = numpy.zeros([numberOfRe,totalNumberOfNodes,fieldGFEM.numberOfFields,max(fieldGFEM.numberOfFieldComponents)])

i = -1
#print(nodeDataSUPG[0,431,0,1])
for Re in ReynoldsNumbers:
    i+=1
    print('Reading data for Re ' + str(Re))
    for proc in range(numberOfProcessors[i]):
        if SUPG[i]:
            path = "./output/Re" + str(Re) + 'Elem' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + '_SUPG_temp/'
            filename = path + 'LidDrivenCavity.part' + str(proc) +'.exnode'
            importNodeData = numpy.zeros([totalNumberOfNodes,fieldSUPG.numberOfFields,max(fieldSUPG.numberOfFieldComponents)])
            readExnodeFile(filename,fieldSUPG,importNodeData,totalNumberOfNodes)
            #print(importNodeData[:,0,1])
            nodeDataSUPG[i,:,:,:] += importNodeData[:,:,:]
            #print(nodeDataSUPG[i,431,0,1])
        else:
            path = "./output/Re" + str(Re) + 'Elem' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + '_GFEM_temp/'
            filename = path + 'LidDrivenCavity.part' + str(proc) +'.exnode'
            importNodeData = numpy.zeros([totalNumberOfNodes,fieldGFEM.numberOfFields,max(fieldGFEM.numberOfFieldComponents)])
            readExnodeFile(filename,fieldGFEM,importNodeData,totalNumberOfNodes)
            nodeDataGFEM[i,:,:,:] += importNodeData[:,:,:]
#print(nodeDataSUPG[i,431,0,1])

        
tolerance = 1e-8
centerNodeNum = -1
vLineY = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])
vLineU = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])
plotVLineU = numpy.zeros([numberOfRe,int(math.sqrt(totalNumberOfNodes))])

# Read centerline data
i = -1
for s in SUPG:
    i+=1
    centerNodeNum = -1
    
    if s:
        dependentFieldIndex = 2
        for node in xrange(totalNumberOfNodes):
            if nodeDataSUPG[i,node,0,0] > (0.5 - tolerance) and nodeDataSUPG[i,node,0,0] < (0.5 + tolerance):
                centerNodeNum += 1
                vLineY[i,centerNodeNum] = nodeDataSUPG[i,node,0,1]
                vLineU[i,centerNodeNum] = nodeDataSUPG[i,node,dependentFieldIndex,0]
    else:
        dependentFieldIndex = 1
        for node in xrange(totalNumberOfNodes):
            if nodeDataGFEM[i,node,0,0] > (0.5 - tolerance) and nodeDataGFEM[i,node,0,0] < (0.5 + tolerance):
                centerNodeNum += 1
                vLineY[i,centerNodeNum] = nodeDataGFEM[i,node,0,1]
                vLineU[i,centerNodeNum] = nodeDataGFEM[i,node,dependentFieldIndex,0]

# Read literature reference data
plotCompareData=numpy.zeros((len(compareSolutions),len(ReynoldsNumbers),100))
compareLocations=[[] for i in range(len(compareSolutions))]
compareData=[[[] for i in range(len(ReynoldsNumbers))] for j in range(len(compareSolutions))]
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
                    compareData[k][i].append(compareAll[:,j].tolist())

plotData = True
labeled = [0]*len(compareNames)
labSupg = False
labGfem = False
spacing = 0.5
p = [0]*len(compareNames)
numberOfUniqueRe = 0
if plotData:
    reIndex =-1
    for Re in ReynoldsNumbers:
        reIndex +=1
        newRe = True
        if reIndex > 0:
             if Re == ReynoldsNumbers[reIndex-1]:
                 newRe = False
             else:
                 numberOfUniqueRe+=1
        if newRe:
            for ref in range(len(compareSolutions)):
                if foundResults[ref,reIndex]:
                    plotCompareData = numpy.zeros((len(compareData[ref][reIndex])))
                    plotCompareData = numpy.array(compareData[ref][reIndex]) - spacing*reIndex
                    #little hack to avoid repeating the symbols in the legend
                    if not labeled[ref]:
                        pylab.plot(plotCompareData[0][0],compareLocations[ref][0][0],compareMarkers[ref],label=compareNames[ref],alpha=1.0)
                        labeled[ref] = 1
                    pylab.plot(plotCompareData,compareLocations[ref],compareMarkers[ref],alpha=1.0)

            plotVLineU[reIndex,:] = vLineU[reIndex,:] - spacing*reIndex
        else:
            plotVLineU[reIndex,:] = vLineU[reIndex,:] - spacing*(reIndex-1)

        if SUPG[reIndex]:
            if labSupg:
                pylab.plot(plotVLineU[reIndex,:],vLineY[reIndex,:],'-r',alpha=0.5)
            else:
                pylab.plot(plotVLineU[reIndex,:],vLineY[reIndex,:],'-r',label="RBS",alpha=0.5)
                labSupg = True
        else:
            if labGfem:
                pylab.plot(plotVLineU[reIndex,:],vLineY[reIndex,:],'-b',alpha=0.5)
            else:
                pylab.plot(plotVLineU[reIndex,:],vLineY[reIndex,:],'-b',label="GFEM",alpha=0.5)
                labGfem = True

    ax = pylab.gca()
    ax.set_aspect(1.5)
    ax.xaxis.set_ticklabels([]) #set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    if plotLegend:
#        pylab.legend(handles, labels, numpoints=1, shadow = True, loc = (0, -0.4), ncol=5)
        newHandles = (handles[2],handles[4],handles[0],handles[1],handles[3])        
        newLabels = (labels[2],labels[4],labels[0],labels[1],labels[3])        
        pylab.legend(newHandles, newLabels, numpoints=1, shadow = True, loc = (0.9, 0.10), ncol=1)
    pylab.xlabel('x-velocity (cm/s)')
    pylab.ylabel('y-coordinate (cm)')
    pylab.title('Mesh element resolution: ' + str(meshResolution[0])+'x'+str(meshResolution[0])+' (' + str(totalNumberOfNodes) +' DOFs)')
    pylab.grid(True)
    # x-scale bar
    pylab.annotate('', xy=(0,-0.03),  xycoords='data',
                   xytext=(1.0,-0.03), textcoords='data',
                   annotation_clip=False,
                   arrowprops=dict(arrowstyle="|-|",
                                   ec="k",
                                   shrinkA=0.01,
                                   shrinkB=0.01,
                                   mutation_scale=2.0
                               )
    )
    pylab.text(0.4,-0.1,'1.0')
    #fname = 'ReAnalysisMesh' + str(meshResolution[0]) + 'x' + str(meshResolution[1]) + 'Dofs' + str(totalNumberOfNodes) + '.pdf'
    fname = 'LidReAnalysis_TH_' + 'Dofs' + str(totalNumberOfNodes) + '.pdf'
    #fname = 'C0LidReAnalysis' + 'Dofs' + str(totalNumberOfNodes) + '.pdf'
    if writeToFigs:
        fname = figsDir + fname
    
    pylab.savefig(fname,format='pdf',dpi=300,bbox_inches='tight')
    #pylab.show()

    
