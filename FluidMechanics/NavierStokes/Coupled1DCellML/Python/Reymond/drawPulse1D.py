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
import gc
import numpy
import math
import re
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt

class fieldInfo(object):
    'base class for info about fields'

    def __init__(self):

        self.group = ''
        self.numberOfFields = 0
        self.fieldNames = []
        self.numberOfFieldComponents = []

def findBetween( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def readExnodeHeader(f,fieldComponents):
    #Read header info
    line=f.readline()
    s = re.findall(r'\d+', line)
    numberOfVersions = 1
    numberOfFields = int(s[0])
    for field in range(numberOfFields):
        line=f.readline()
        fieldName = findBetween(line, str(field + 1) + ') ', ',')
        numberOfComponents = int(findBetween(line, '#Components=', '\n'))
        fieldComponents.append(numberOfComponents)
        for skip in range(numberOfComponents):
            line=f.readline()
            if ('#Versions' in line):
                numberOfVersions = int(findBetween(line, '#Versions=', '\n'))
    return(numberOfVersions)

def readExnodeFile(filename,numberOfNodes):
    # Read data from an exnode file
    try:
        with open(filename):
            f = open(filename,"r")
            #Read header
            line=f.readline()
            groupName=findBetween(line, ' Group name: ', '\n')
            fieldComponents = []
            numberOfVersions = readExnodeHeader(f,fieldComponents)
            numberOfFields = len(fieldComponents)
#            nodeData = numpy.zeros([numberOfNodes,numberOfFields,3,4])
            nodeData = numpy.zeros([numberOfNodes,12,3,4])

            #Read node data
            endOfFile = False
            while endOfFile == False:
                previousPosition = f.tell()
                line=f.readline()
                line = line.strip()
                if line:
                    if 'Node:' in line:
                        s = re.findall(r'\d+', line)
                        node = int(s[0]) - 1
                        for field in range(numberOfFields):
                            for component in range(fieldComponents[field]):
                                line=f.readline()
                                line = line.strip()
                                values = map(float, line.split('  '))
                                for version in range(numberOfVersions):
                                    nodeData[node,field,component,version] = values[version]
                    elif 'Fields' in line:
                        f.seek(previousPosition)
                        fieldComponents = []
                        numberOfFields = 0
                        numberOfVersions = readExnodeHeader(f,fieldComponents)
                        numberOfFields = len(fieldComponents)
                else:
                    endOfFile = True
                    f.close()
            return(nodeData);
    except IOError:
       print ('Could not open file: ' + filename)

def plotQP (Q,P,realTime,i,arteryNames,startTime,stopTime,writeFigs,writeFigs2,writeFigs2Display):
    if numberOfDatasets < 2:
        if writeFigs2:
            matplotlib.rcParams.update({'font.size': 22})
            plt.subplot(111)
            plt.plot(realTime, Q[0,i,:], 'k')
        else:
            plt.subplot(211)
            l1, = plt.plot(realTime, Q[0,0,:], 'k--')
            l2, = plt.plot(realTime, Q[0,1,:], 'b')
            l3, = plt.plot(realTime, Q[0,2,:], 'r')
        plt.ylabel(r'Flow rate ($cm^3/s$)')#,fontsize=14)
        plt.xlabel(r'time ($ms$)')#,fontsize=14)
        plt.xlim((startTime,stopTime))
        plt.grid(True)
        if writeFigs2:
            plt.xlim((startTime,stopTime))
            plt.title(arteryNames[i],fontsize=26)
            filename = arteryNames[i].replace(' ','-')+'-flow'
            plt.savefig('/hpc/dlad004/thesis/Papers/CellMLCoupling/figures/'+filename+'.pdf', format='pdf')
            if writeFigs2Display:
                plt.show()
    #    plt.title(r'Flow and pressure in the 1D Visible Human example',fontsize=16)

        if writeFigs2:
            plt.subplot(111)
            plt.plot(realTime, P[0,i,:], 'k')
        else:
            plt.subplot(212)
            l1, = plt.plot(realTime, P[0,0,:], 'k--')
            l2, = plt.plot(realTime, P[0,1,:], 'b')
            l3, = plt.plot(realTime, P[0,2,:], 'r')
            plt.legend( (l1, l2,l3), (arteryNames[0],arteryNames[1],arteryNames[2]), loc='lower center', bbox_to_anchor=(0.5,-0.5),ncol=3,shadow=False)
        #plt.legend( (l1, l2), (arteryNames[0],arteryNames[1]), 'upper right', shadow=True)
        plt.ylabel(r'Pressure ($mmHg$)')#,fontsize=14)
        plt.xlabel(r'time ($ms$)')#,fontsize=14)
        plt.xlim((startTime,stopTime))
        plt.grid(True)
        if writeFigs2:
            plt.title(arteryNames[i],fontsize=26)
            filename = arteryNames[i].replace(' ','-')+'-pressure'
            plt.savefig('/hpc/dlad004/thesis/Papers/CellMLCoupling/figures/'+filename+'.pdf', format='pdf')
            if writeFigs2Display:
                plt.show()
        else:
            plt.subplots_adjust(bottom=0.2)
            plt.show()
            if writeFigs:
                plt.savefig('/hpc/dlad004/thesis/Thesis/figures/cfd/1DArteriesPulseQP.pdf', format='pdf')
    else:
        plt.subplot(211)
        for dataset in range(numberOfDatasets):
            l = 0
            for artery in range(len(arteryNames)):
                if dataset == 1:
                    color = colors[l] + '--'
                else:
                    color = colors[l]
                plt.plot(realTime,Q[dataset,artery,:],color)
                l+=1
        plt.ylabel(r'Flow rate ($cm^3/s$)')#,fontsize=14)
        plt.xlim((0,stopTime))
        plt.grid(True)
    #    plt.title(r'Flow and pressure in the 1D Visible Human example',fontsize=16)

        plt.subplot(212)
        for dataset in range(numberOfDatasets):
            l = 0
            for artery in range(len(arteryNames)):
                if dataset == 1:
                    color = colors[l] + '--'
                else:
                    color = colors[l]
                plt.plot(realTime,P[dataset,artery,:],color)
                l+=1
        plt.ylabel(r'Pressure ($mmHg$)')#,fontsize=14)
        plt.xlabel(r'time ($ms$)')#,fontsize=14)
        plt.xlim((0,stopTime))
        plt.grid(True)
        plt.subplots_adjust(bottom=0.2)
        #plt.legend( (l1, l2,l3), (arteryNames[0],arteryNames[1],arteryNames[2]), loc='lower center', bbox_to_anchor=(0.5,-0.5),ncol=3,shadow=False)
        #plt.title('Pressure in straight segment low resolution')
        if writeFigs:
            plt.savefig('/hpc/dlad004/thesis/Thesis/figures/cfd/1DArteriesPulseQP.pdf', format='pdf')
        plt.show()

#=================================================================
# C o n t r o l   P a n e l
#=================================================================
timeIncrement = 0.2
outputFrequency = 10
timeIncrements = [timeIncrement]
outputFrequencies = [outputFrequency]
cycleTime = 790.0
numberOfCycles = 4.0
startTime = 0.0
stopTime = cycleTime*numberOfCycles
pulsePeriod = cycleTime
totalNumberOfNodes = 197
paths = ['./output/']
numberOfProcessors = 1
density = 0.00105
zeroTolerance = 1.0e-15

# Plot flow and pressure at these nodes
writeFigs = False
writeFigs2 = False
writeFigs2Display = False
plotNodeQP = False
plotPulseConvergence = False
nodes = [1,38,20,65,78,81,98,96]#,65,133]
plotNode = 1
arteryNames = ["Aortic root","R common carotid","L subclavian","Thoracic aorta","L renal","Abdominal aorta","R common iliac","L external iliac"]
colors = ['k','b','r','y','g','c','m']#,'k--','b--','r--','y--','g--','c--','m--']

# Check conservation of mass between inlet and sum of outlets
checkBranchFlow = True
inlets = [1]
outlets = [17, 33, 37, 45, 49, 57, 61, 71, 85, 89, 105, 107, 113, 115, 117, 125, 131, 145, 153, 157, 161, 167, 181, 189, 193, 195, 197] 
#outlets = [25,37,45,47,59,61,77,79,95,105,115,125,131,135]
evalAtPulse = 1

writeInit = False

#=================================================================
#=================================================================
numberOfDatasets = len(paths)
numberOfTimesteps = int((stopTime-startTime)/(timeIncrement*outputFrequency))+1
times = []
for dataset in range(numberOfDatasets):
    times.append([int(startTime/timeIncrement)+
                  i*outputFrequencies[dataset] for i in range(numberOfTimesteps)])
realTime = [startTime+ i*timeIncrement*outputFrequency for i in range(numberOfTimesteps)]
numberOfNodes = len(nodes)
nodeData = numpy.zeros([numberOfDatasets,numberOfTimesteps,totalNumberOfNodes,12,3,4])

print("Reading output file data (this can take a while...)")
for dataset in range(numberOfDatasets):
    path = paths[dataset]
    for timestep in range(numberOfTimesteps):
        for proc in range(numberOfProcessors):
            filename = path + 'MainTime_' + str(times[dataset][timestep]) + '.part' + str(proc) +'.exnode'
            #print(filename)
            timeData = readExnodeFile(filename,totalNumberOfNodes)
            nodeData[dataset,timestep,:,:,:,:] = timeData

# nodeData[dataset,time,node,field,component,version] = values[version]

Q = numpy.zeros([numberOfDatasets,numberOfNodes,numberOfTimesteps])
A = numpy.zeros([numberOfDatasets,numberOfNodes,numberOfTimesteps])
P = numpy.zeros([numberOfDatasets,numberOfNodes,numberOfTimesteps])

for dataset in range(numberOfDatasets):
    for i in range(numberOfNodes):
        node = nodes[i]
        Q[dataset,i,:] = nodeData[dataset,:,node-1,1,0,0]
        A[dataset,i,:] = nodeData[dataset,:,node-1,1,1,0]
        P[dataset,i,:] = nodeData[dataset,:,node-1,5,0,0]*0.0075*(1000.**2.) # convert Pa to mmHg

if writeInit:
    initData = numpy.zeros([totalNumberOfNodes,4,4])
    fields = [1,1,2,2]
    components = [0,1,0,1]
    for i in range(4):
        initData[:,i,:] = nodeData[0,numberOfTimesteps-1,:,fields[i],components[i],:]
#    initData[:,1] = nodeData[0,numberOfTimesteps-1,:,1,1,0]
#    numpy.savetxt('Input/init.csv',initData,delimiter=',')
    numpy.save('Input/init.npy',initData)
    

timestepsInPulsePeriod = int(pulsePeriod/(timeIncrement*outputFrequency))
print('timesteps in period: '+ str(timestepsInPulsePeriod))
numberOfPulses = int((stopTime-startTime)/pulsePeriod)
print('number of pulses: '+ str(numberOfPulses))
cycleData = numpy.zeros([numberOfDatasets,numberOfPulses,timestepsInPulsePeriod,totalNumberOfNodes,2])
cycleDifference = numpy.zeros([numberOfDatasets,numberOfPulses,timestepsInPulsePeriod,totalNumberOfNodes,2])
percentCycleDifference = numpy.zeros([numberOfDatasets,numberOfPulses,timestepsInPulsePeriod,totalNumberOfNodes,2])
rmsCycle = numpy.zeros([numberOfDatasets,numberOfPulses,2])
percentRmsCycle = numpy.zeros([numberOfDatasets,numberOfPulses,2])
stdCycle = numpy.zeros([numberOfDatasets,numberOfPulses,2])
cycleTime = [i*timeIncrement*outputFrequency for i in range(timestepsInPulsePeriod)]

for dataset in range(numberOfDatasets):
    for cycle in range(numberOfPulses):
        startStep = cycle*timestepsInPulsePeriod
        stopStep = (cycle+1)*timestepsInPulsePeriod
        cycleData[dataset,cycle,:,:,0] = nodeData[dataset,startStep:stopStep,:,1,0,0]
        cycleData[dataset,cycle,:,:,1] = nodeData[dataset,startStep:stopStep,:,5,0,0]*0.0075*(1000.**2.) # convert Pa to mmHg

for dataset in range(numberOfDatasets):
    for cycle in range(numberOfPulses):
        if cycle > 0:
            cycleDifference[dataset,cycle,:,:,:] = (cycleData[dataset,cycle,:,:,:] - cycleData[dataset,cycle-1,:,:,:])
            percentCycleDifference[dataset,cycle,:,:,:] = (cycleData[dataset,cycle,:,:,:] - cycleData[dataset,cycle-1,:,:,:])/(cycleData[dataset,cycle,:,:,:]+zeroTolerance)*100.
            plotDifferences = False
            if plotDifferences:
                plt.subplot(211)
                l1, = plt.plot(cycleTime, cycleData[dataset,cycle-1,:,nodes[1]-1,0], 'b')
                l2, = plt.plot(cycleTime, cycleData[dataset,cycle,:,nodes[1]-1,0], 'r')
                plt.legend( (l1, l2), ('previous pulse', 'current'), 'upper right', shadow=True)
                plt.ylabel('Flow (cm^3/s)')
                plt.xlabel('time (ms)')
                plt.grid(True)
                plt.subplot(212)
                l1, = plt.plot(cycleTime, cycleData[dataset,cycle-1,:,nodes[1]-1,1], 'b')
                l2, = plt.plot(cycleTime, cycleData[dataset,cycle,:,nodes[1]-1,1], 'r')
                plt.ylabel('Pressure (mmHg)')
                plt.xlabel('time (ms)')
                plt.grid(True)
                plt.show()

        else:
    #        cycleDifference[dataset,cycle,:,:,:] = (cycleData[numberOfPulses-1,:,:,:] - cycleData[dataset,cycle,:,:,:])/(cycleData[numberOfPulses-1,:,:,:]+zeroTolerance)*100.
            cycleDifference[dataset,cycle,:,:,:] = 0.0
            percentCycleDifference[dataset,cycle,:,:,:] = 0.0

        rmsCycle[dataset,cycle,0] = numpy.sqrt(numpy.mean(cycleDifference[dataset,cycle,:,:,0])**2)
        rmsCycle[dataset,cycle,1] = numpy.sqrt(numpy.mean(cycleDifference[dataset,cycle,:,:,1])**2)

        percentRmsCycle[dataset,cycle,0] = numpy.sqrt(numpy.mean(percentCycleDifference[dataset,cycle,:,:,0])**2)
        percentRmsCycle[dataset,cycle,1] = numpy.sqrt(numpy.mean(percentCycleDifference[dataset,cycle,:,:,1])**2)

        stdCycle[dataset,cycle,0] = numpy.std(cycleDifference[dataset,cycle,:,:,0])
        stdCycle[dataset,cycle,1] = numpy.std(cycleDifference[dataset,cycle,:,:,1])

print('cycleDifference: ')
print(rmsCycle)

print('percentCycleDifference: ')
print(percentRmsCycle)

if plotNodeQP:
#    plt.rc('text', usetex=True)
#    plt.rc('font', family='serif')

    if writeFigs2:
        for i in range(len(arteryNames)):
            plotQP(Q,P,realTime,i,arteryNames,startTime,stopTime,writeFigs,writeFigs2,writeFigs2Display)
    else:
        plotQP(Q,P,realTime,i,arteryNames,startTime,stopTime,writeFigs,writeFigs2)

    # if numberOfDatasets < 2:
    #     if writeFigs2:
    #         plt.subplot(111)
    #     else:
    #         plt.subplot(211)
        
    #     if len(arteryNames) == 1:
    #         plt.plot(realTime, Q[0,0,:], 'k')
    #     else:
    #         l1, = plt.plot(realTime, Q[0,0,:], 'k--')
    #         l2, = plt.plot(realTime, Q[0,1,:], 'b')
    #         l3, = plt.plot(realTime, Q[0,2,:], 'r')
    #     plt.ylabel(r'Flow rate ($cm^3/s$)')#,fontsize=14)
    #     plt.xlabel(r'time ($ms$)')#,fontsize=14)
    #     plt.xlim((startTime,stopTime))
    #     plt.grid(True)
    #     if writeFigs2:
    #         plt.xlim((startTime,stopTime))
    #         plt.title(arteryNames[0],fontsize=16)
    #         plt.show()
    #         filename = arteryNames[0].replace(' ','-')
    #         plt.savefig('/hpc/dlad004/thesis/Papers/CellMLCoupling/figures/'+filename+'.pdf', format='pdf')
    # #    plt.title(r'Flow and pressure in the 1D Visible Human example',fontsize=16)

    #     if writeFigs2:
    #         plt.subplot(111)
    #     else:
    #         plt.subplot(212)
    #     if len(arteryNames) == 1:
    #         plt.plot(realTime, P[0,0,:], 'k')
    #     else:
    #         l1, = plt.plot(realTime, P[0,0,:], 'k--')
    #         l2, = plt.plot(realTime, P[0,1,:], 'b')
    #         l3, = plt.plot(realTime, P[0,2,:], 'r')
    #         plt.legend( (l1, l2,l3), (arteryNames[0],arteryNames[1],arteryNames[2]), loc='lower center', bbox_to_anchor=(0.5,-0.5),ncol=3,shadow=False)
    #     #plt.legend( (l1, l2), (arteryNames[0],arteryNames[1]), 'upper right', shadow=True)
    #     plt.ylabel(r'Pressure ($mmHg$)')#,fontsize=14)
    #     plt.xlabel(r'time ($ms$)')#,fontsize=14)
    #     plt.xlim((startTime,stopTime))
    #     plt.grid(True)
    #     if writeFigs2:
    #         plt.title(arteryNames[0],fontsize=16)
    #         plt.show()
    #         filename = arteryNames[0].replace(' ','-')
    #         plt.savefig('/hpc/dlad004/thesis/Papers/CellMLCoupling/figures/'+filename+'.pdf', format='pdf')
    #     else:
    #         plt.subplots_adjust(bottom=0.2)
    #         plt.show()
    #         if writeFigs:
    #             plt.savefig('/hpc/dlad004/thesis/Thesis/figures/cfd/1DArteriesPulseQP.pdf', format='pdf')


    # else:
    #     plt.subplot(211)
    #     for dataset in range(numberOfDatasets):
    #         l = 0
    #         for artery in range(len(arteryNames)):
    #             if dataset == 1:
    #                 color = colors[l] + '--'
    #             else:
    #                 color = colors[l]
    #             plt.plot(realTime,Q[dataset,artery,:],color)
    #             l+=1
    #     plt.ylabel(r'Flow rate ($cm^3/s$)')#,fontsize=14)
    #     plt.xlim((0,stopTime))
    #     plt.grid(True)
    # #    plt.title(r'Flow and pressure in the 1D Visible Human example',fontsize=16)

    #     plt.subplot(212)
    #     for dataset in range(numberOfDatasets):
    #         l = 0
    #         for artery in range(len(arteryNames)):
    #             if dataset == 1:
    #                 color = colors[l] + '--'
    #             else:
    #                 color = colors[l]
    #             plt.plot(realTime,P[dataset,artery,:],color)
    #             l+=1
    #     plt.ylabel(r'Pressure ($mmHg$)')#,fontsize=14)
    #     plt.xlabel(r'time ($ms$)')#,fontsize=14)
    #     plt.xlim((0,stopTime))
    #     plt.grid(True)
    #     plt.subplots_adjust(bottom=0.2)
    #     #plt.legend( (l1, l2,l3), (arteryNames[0],arteryNames[1],arteryNames[2]), loc='lower center', bbox_to_anchor=(0.5,-0.5),ncol=3,shadow=False)
    #     #plt.title('Pressure in straight segment low resolution')
    #     if writeFigs:
    #         plt.savefig('/hpc/dlad004/thesis/Thesis/figures/cfd/1DArteriesPulseQP.pdf', format='pdf')
    #     plt.show()

cycles = numpy.arange(1,numberOfPulses+1)

if plotPulseConvergence:
    barplot = False
    if barplot:
        plt.subplot(111)
        print(cycles)
        barWidth = 0.35
        opacity = 1.0
        errorConfig = {'ecolor': '0.3'}
        bar1 = plt.bar(cycles-barWidth, rmsCycle[:,0], barWidth, alpha=opacity, color='b',label='Flow')
        bar2 = plt.bar(cycles, rmsCycle[:,1], barWidth, alpha=opacity, color='r',label='Pressure')
        #plt.legend( (bar1, bar2), ("Flow","Pressure"), 'upper right', shadow=True)
        plt.legend()
        plt.ylabel('Percent RMS error (%)')
        plt.xlabel('Pulse number')
        plt.xticks(cycles)
        plt.yscale('log')
        plt.grid(True)
        plt.title('RMS % difference vs. previous pulse values')

        plt.tight_layout()
        plt.show()
    else:
        # plt.subplot(111)
        # cycles = numpy.arange(1,numberOfPulses+1)
        # l1, = plt.plot(cycles, rmsCycle[:,0], 'b')
        # l2, = plt.plot(cycles, rmsCycle[:,1], 'r')
        # plt.legend( (l1, l2), ('Flow rate', 'Pressure'), 'upper right', shadow=True)
        # plt.ylabel('RMS difference (cm^3/s,mmHg)')
        # plt.xlabel('Pulse number')
        # plt.xticks(cycles)
        # plt.yscale('log')
        # plt.grid(True)
        # plt.title('Difference with previous pulse values')
        # plt.tight_layout()
        # plt.show()

        plt.subplot(111)
        l1, = plt.plot(cycles, percentRmsCycle[0,:,0], 'b')
        l2, = plt.plot(cycles, percentRmsCycle[0,:,1], 'r')
        plt.legend( (l1, l2), (r'Flow rate', r'Pressure'), 'upper right', shadow=True)
        plt.ylabel(r'RMS percent change ($\%$)')
        plt.xlabel(r'Pulse number')
        plt.yscale('log')
        plt.grid(True)
        plt.title(r'Percent difference vs. previous pulse values')
        if writeFigs:
            plt.savefig('/hpc/dlad004/thesis/Thesis/figures/cfd/1DArteriesPulseConvergence.pdf', format='pdf')
        #plt.tight_layout()
        plt.show()

if checkBranchFlow:
    lines = []
    barWidth = 0.35
    inflowData = numpy.zeros((numberOfDatasets,len(inlets),timestepsInPulsePeriod))
    outflowData = numpy.zeros((numberOfDatasets,len(outlets),timestepsInPulsePeriod))

    plotStartTime = float(evalAtPulse*pulsePeriod)
    plotTime = [(plotStartTime + i*timeIncrement*outputFrequency) for i in range(timestepsInPulsePeriod)]
    massIn = numpy.zeros((numberOfDatasets,numberOfPulses))
    massOut = numpy.zeros((numberOfDatasets,numberOfPulses))
    percentError = numpy.zeros((numberOfDatasets,numberOfPulses))
    error = numpy.zeros((numberOfDatasets,numberOfPulses))
    #cycleData = numpy.zeros([numberOfPulses,timestepsInPulsePeriod,totalNumberOfNodes,2])

    for dataset in range(numberOfDatasets):
        for cycle in range(numberOfPulses):
            l = 0
            iindex = 0
            oindex = 0
            for inlet in inlets:
                inflowData[dataset,iindex,:] = cycleData[dataset,cycle,:,inlet-1,0]
        #        plt.plot(plotTime,inflowData[iindex,:],colors[l])
                massIn[dataset,cycle] += density*scipy.integrate.simps(inflowData[dataset,iindex,:])
                l += 1
                iindex += 1
            for outlet in outlets:
                outflowData[dataset,oindex,:] = cycleData[dataset,cycle,:,outlet-1,0]
        #        plt.plot(plotTime,outflowData[oindex,:],colors[l])
                massOut[dataset,cycle] += density*scipy.integrate.simps(outflowData[dataset,oindex,:])
                l += 1
                oindex += 1

            error[dataset,cycle] = massIn[dataset,cycle] - massOut[dataset,cycle]
            percentError[dataset,cycle] = (massIn[dataset,cycle] - massOut[dataset,cycle])/massIn[dataset,cycle]*100.

    
#    plt.plot(cycles,percentError,'k')

#     plt.subplot(211)
#     plt.bar(cycles-barWidth/2,error,barWidth,color='b')
# #    plt.plot(cycles,error,'b')
#     plt.ylabel('Mass error (g)')
#     plt.xlabel('Pulse number')
#     plt.grid(True)
#     plt.title('Error in mass conservation (inlet - outlets)')

    print('percent mass error (inlet - outlets): ')
    print(percentError)

    # Free up some memory
    nodeData = numpy.zeros([numberOfDatasets,numberOfTimesteps,totalNumberOfNodes,8,3,4])
    gc.collect()

    plt.subplot(111)
#    plt.bar(cycles-barWidth/2,percentError,barWidth,color='b')
    if numberOfDatasets > 1:
        l1, = plt.plot(cycles, percentError[0,:], 'b')
        l2, = plt.plot(cycles, percentError[1,:], 'r')
        plt.legend( (l1, l2), (r'$\Delta{t}=0.1$ $ms$', 
                               r'$\Delta{t}=0.01$ $ms$'), 'upper right', shadow=True)
    else:
        plt.plot(cycles,percentError[0,:],'b')
    plt.ylabel(r'Percent mass error ($\%\epsilon_{mass}$)')
    plt.xlabel(r'Pulse number')
    plt.ylim((0,11.0))
    plt.yscale('log')
#    plt.xticks([0,1,2,4,6,8,10,12])
    plt.xlim((0,numberOfCycles+1))
    plt.grid(True)
    plt.title(r'Percent error in mass conservation (inlet - outlets)')
    if writeFigs:
        plt.savefig('/hpc_atog/dlad004/thesis/Thesis/figures/cfd/1DMassConservation.pdf', format='pdf')
    plt.show()    
