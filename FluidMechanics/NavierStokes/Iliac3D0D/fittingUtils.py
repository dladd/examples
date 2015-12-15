#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This module performs an inverse distance weight (IDW) fit based on geometric locations
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
#> \example /DataPointFitting/DataPointsToDependentField/DataPointsToDependentFieldExample.py
## Example script to fit data points to a dependent field of a generated mesh using openCMISS calls in python.
## \par Latest Builds:
#<


# Add Python bindings directory to PATH
import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy
from numpy import linalg
import bisect
import time

def writeDataPoints(locations,velocity,filename):
    '''outputs data points field values and locations to a .C file '''
    data = numpy.hstack((locations,velocity))
    print("writing " + filename)
    f = open(filename,'w')
    numberOfDataPoints = str(len(locations)) + ' '
    line = numberOfDataPoints + numberOfDataPoints + numberOfDataPoints + '\n'
    f.write(line)
    numpy.savetxt(f,data,delimiter=' ')
    f.close()

def CalculateWeights(p,vicinityFactor,dataResolution,sourceLocations,
                     targetLocations,targetNodeList,wallNodes,dataList,weight,sumWeights):
    '''Calculate Inverse Distance Weighting interpolation parameters '''

    zeroTolerance = 1e-10
    numberOfSourcePoints = len(sourceLocations)
    numberOfTargetPoints = len(targetLocations)
    numberOfWallNodes = len(wallNodes)
    timeIdwStart = time.time()        

    fastSearch = True
    if fastSearch:
        # This does a bit of array sorting by the first index to speed this up
        dataIndex = numpy.argsort(sourceLocations[:,0])
        sortedData = sourceLocations[dataIndex]
        numberOfSearchFailures =0

    for node in targetNodeList:
        meshPosition = numpy.zeros((3))
        #set values to zero if wall node
        nodeNumberCmiss = node + 1
        if(nodeNumberCmiss not in wallNodes[:]):
            # get the geometric position for this node
            meshPosition[:] = targetLocations[node,:]
            if(fastSearch):
                minX = bisect.bisect_left(sortedData[:,0],meshPosition[0]-vicinityFactor*dataResolution[0])
                maxX = bisect.bisect_right(sortedData[:,0],meshPosition[0]+vicinityFactor*dataResolution[0])
                numberOfCandidates = maxX - minX
                dataPoint = -1
                fittingPoint = 0
                for candidate in xrange(numberOfCandidates):
                    if(abs(sortedData[minX+candidate,0] - meshPosition[0]) < vicinityFactor*dataResolution[0] and
                       abs(sortedData[minX+candidate,1] - meshPosition[1]) < vicinityFactor*dataResolution[1] and
                       abs(sortedData[minX+candidate,2] - meshPosition[2]) < vicinityFactor*dataResolution[2]):
                        dataPoint = dataIndex[minX+candidate]
                        dataList[node].append(dataPoint)
                        difference = numpy.subtract(sourceLocations[dataPoint,:], meshPosition)
                        distance = linalg.norm(difference,2)
                        weight[node,fittingPoint] = 1.0/((distance+zeroTolerance)**p)
                        sumWeights[node] += weight[node,fittingPoint]
                        fittingPoint+=1
                if dataPoint == -1:
                    print("Could not find data points in vicinity of mesh node: " + str(node) + "\n")
                    print("Tolerance: " + str(vicinityFactor*dataResolution[0]) + "\n")
                    numberOfSearchFailures += 1

            else:
                numberOfSearchFailures = 0
                #find indices of data points within a distance tolerance of this node
                fittingPoint=0
                for dataPoint in xrange(numberOfSourcePoints):
                    dataPointId = dataPoint + 1
                    if(abs(sourceLocations[dataPoint,0] - meshPosition[0]) < vicinityFactor*dataResolution[0] and
                       abs(sourceLocations[dataPoint,1] - meshPosition[1]) < vicinityFactor*dataResolution[1] and
                       abs(sourceLocations[dataPoint,2] - meshPosition[2]) < vicinityFactor*dataResolution[2]):

                        #within tolerance - save dataPoint to list
                        dataList[node].append(dataPoint)
                        difference = numpy.subtract(sourceLocations[dataPoint,:], meshPosition)
                        distance = linalg.norm(difference,2)
                        weight[node,fittingPoint] = 1.0/((distance+zeroTolerance)**p)
                        sumWeights[node] += weight[node,fittingPoint]
                        fittingPoint+=1

    t2 = time.time()        
    sys.stdout.write("\n")
    print("Time to do geometric fit (s): " + str(t2-timeIdwStart))
    print("total number of search failures: " + str(numberOfSearchFailures))



def VectorFit(sourceVectors,targetVectors,targetNodeList,wallNodes,dataList,weight,sumWeights):
    '''Solve Inverse Distance Weighting interpolation problem'''
    print("Fitting Velocity data")

    print('shape of data point vectors: ')
    print(sourceVectors.shape)
    timeStart = time.time()        
    # Use geometry weights to define time-dependent node based vector values
    for node in targetNodeList:
        nodeVector = numpy.zeros((3))
        velocityVector = numpy.zeros((3))
        nodeNumberCmiss = node + 1
        if (nodeNumberCmiss in wallNodes[:]):
            nodeVector[:] = 0.0
            targetVectors[node,:] = 0.0
        else:
            for fittingPoint in xrange(len(dataList[node])):
                dataPoint = dataList[node][fittingPoint]
                velocityVector[:] = sourceVectors[dataPoint,:]
                nodeVector[:] += weight[node,fittingPoint]*velocityVector[:]/sumWeights[node]

            # vector for this node is the sum of the contributing fitting point vectors
            targetVectors[node,:] = nodeVector[:]

    idwMean = numpy.mean(targetVectors)
    print('IDW Mean: ' + str(idwMean))
    timeStop = time.time()        
    timeIdw = timeStop - timeStart
    print("finished idw vector fit, time to fit: " + str(timeIdw))
