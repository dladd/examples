#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an example script to fit a data point vector field from phase contrast velocimetry to a field interpolated on a Carotid mesh
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
import pylab
import scipy
from scipy import interpolate
import math
import re
import bisect
import time
import optparse
import fittingUtils as fit

#---------------------------
# Intialise OpenCMISS
#---------------------------
from opencmiss import CMISS

(coordinateSystemUserNumber,
    regionUserNumber,
    basisUserNumber,
    meshUserNumber,
    decompositionUserNumber,
    geometricFieldUserNumber,
    equationsSetFieldUserNumber,
    dependentFieldUserNumber,
    independentFieldUserNumber,
    dataPointFieldUserNumber,
    materialFieldUserNumber,
    equationsSetUserNumber,
    problemUserNumber) = range(1,14)

#---------------------------
# Input flags
#---------------------------
parser = optparse.OptionParser()
parser.add_option("-a", "--aorta", default=False, action="store_true", help="flag whether to solve the tetrahedral aorta rather than the hexahedral carotid mesh. default=False (hexahedral mesh)")
parser.add_option("-z", "--addZeroLayer", default=False, action="store_true", help="flag whether to add zero layer to IDW fit. default=True")
parser.add_option("-s", "--startTime", action="store", type="float", dest="startTime", default=0.,
                   help='time to start fitting from PCV data')
parser.add_option("-e", "--stopTime", action="store", type="float", dest="stopTime", default=0.80,
                   help='time to stop fitting from PCV data')
parser.add_option("-i", "--timeIncrement", action="store", type="float", dest="timeIncrement", default=0.01,
                   help='time increment for the resultant fit fields from PCV')
parser.add_option("-f", "--startFit", action="store", type="int", dest="startFit", default=0,
                   help='specify fitting to start at another specified timestep')
parser.add_option("-v", "--velocityScaleFactor", action="store", type="float", dest="velocityScaleFactor", default=1.0,
                   help='scale factor for imported velocity data from PCV')

(options, args) = parser.parse_args()
aorta = options.aorta
addZeroLayer = options.addZeroLayer
startTime = options.startTime
stopTime = options.stopTime
startFit = options.startFit
timeIncrement = options.timeIncrement
velocityScaleFactor = options.velocityScaleFactor

stopTime += 1e-8 # in case of machine error
outputExtras = False
#---------------------------
# Set up mesh
#---------------------------
if (aorta):
    meshName = 'Aorta'
    inputDir = './input/Aorta/'
    pcvDataDir = inputDir + 'pcvData/'
    outputDir = './output/Aorta/'
    pcvTimeIncrement = 0.0408
    numberOfPcvDataFiles = 17
    dataPointResolution = [1.6666666269302,1.6666666269302,2.2000000476837]
else:
    meshName = 'RCA2'            
    inputDir = './input/RCA2/'
    pcvDataDir = inputDir + 'pcvData/'
    outputDir = './output/RCA2/'
    pcvTimeIncrement = 0.0456
    numberOfPcvDataFiles = 17
    dataPointResolution = [0.859375,0.859375,1.4000001] # can read using dcmdump, search pixel spacing and slice thickness
fieldmlInput = inputDir + 'mesh/' + meshName + '.xml'
print('input file: ' + fieldmlInput)

#Wall boundary nodes
filename=inputDir + 'mesh/bc/wallNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfWallNodes=int(f.readline())
        wallNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Wall boundary node file: ' + filename)

#---------------------------
# Setup computational info
#---------------------------

CMISS.DiagnosticsSetOn(CMISS.DiagnosticTypes.IN,[1,2,3,4,5],"Diagnostics",["DOMAIN_MAPPINGS_LOCAL_FROM_GLOBAL_CALCULATE"])

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()

#---------------------------
# Begin field setup
#---------------------------

#Initialise fieldML IO
fieldmlInfo=CMISS.FieldMLIO()
fieldmlInfo.InputCreateFromFile(fieldmlInput)

# Creation a RC coordinate system
coordinateSystem = CMISS.CoordinateSystem()
fieldmlInfo.InputCoordinateSystemCreateStart("ArteryMesh.coordinates",coordinateSystem,coordinateSystemUserNumber)
coordinateSystem.CreateFinish()
numberOfDimensions = coordinateSystem.DimensionGet()

# Create a region
region = CMISS.Region()
region.CreateStart(regionUserNumber,CMISS.WorldRegion)
region.label = "FittingRegion"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Create nodes
nodes=CMISS.Nodes()
fieldmlInfo.InputNodesCreateStart("ArteryMesh.nodes.argument",region,nodes)
nodes.CreateFinish()
numberOfNodes = nodes.numberOfNodes
print("number of nodes: " + str(numberOfNodes))


#---------------------------
# Basis setup
#---------------------------

gaussQuadrature = [3,3,3]
quadratureOrder = 5
basisNumberQuadratic = 1
numberOfMeshComponents=1
meshComponent=1

if (aorta):
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.triquadratic_simplex",basisNumberQuadratic)
    CMISS.Basis_QuadratureOrderSetNum(basisNumberQuadratic,quadratureOrder)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
else:
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.triquadratic_lagrange",basisNumberQuadratic)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)

#---------------------------
# Create Mesh
#---------------------------
mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("ArteryMesh.mesh.argument",mesh,meshUserNumber,region)
mesh.NumberOfComponentsSet(numberOfMeshComponents)
fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent,"ArteryMesh.template.triquadratic")
mesh.CreateFinish()
numberOfElements = mesh.numberOfElements
print("number of elements: " + str(numberOfElements))

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CalculateFacesSet(False)
decomposition.CreateFinish()


#=======================================================================
# D a t a    P o i n t   F i t t i n g
#=======================================================================

#-----------------------------
# Read in PCV point geometry
#-----------------------------
filename=pcvDataDir + 'veldata.geom'
try:
    with open(filename):
        print("Reading " + filename)
        f = open(filename,"r")
        numberOfPcvPoints=int(f.readline())
        print('# of points: ' + str(numberOfPcvPoints))
        pcvGeometry = numpy.loadtxt(f)
        f.close()
except IOError:
   print ('Error reading  PCV geometry file: ' + filename)        

numberOfNonzeroPcvPoints = numberOfPcvPoints
if addZeroLayer:
    # Read in zero exterior layer data
    filename=pcvDataDir + 'zeroLayer.geom'
    try:
        with open(filename):
            print("Reading " + filename)
            f = open(filename,"r")
            numberOfZeroPcvPoints=int(f.readline())
            numberOfPcvPoints += numberOfZeroPcvPoints
            print('# of zero points: ' + str(numberOfZeroPcvPoints))
            pcvZeroLayer = numpy.loadtxt(f)
            f.close()
    except IOError:
       print ('Error reading  PCV zero layer file: ' + filename)        
    pcvGeometry = numpy.vstack((pcvGeometry,pcvZeroLayer))

# Set up data points with geometric values
dataPoints = CMISS.DataPoints()
dataPoints.CreateStart(region,numberOfPcvPoints)
for dataPoint in range(numberOfPcvPoints):
    dataPointId = dataPoint + 1
    dataList = pcvGeometry[dataPoint,:]
    dataPoints.ValuesSet(dataPointId,dataList)
    dataPoints.LabelSet(dataPointId,"PCV Data Points")
dataPoints.CreateFinish()

#--------------------------------------------
# Read in velocity data
#--------------------------------------------

numberOfTimesteps = int((stopTime - startTime)/timeIncrement) + 1
numberOfPcvTimesteps = int(((stopTime+pcvTimeIncrement*2.0) - startTime)/pcvTimeIncrement) + 1

# Read in PCV data
#-----------------------------
pcvData=numpy.zeros((numberOfPcvTimesteps,numberOfPcvPoints,3))
print('shape of data point vectors: ')
print(pcvData.shape)
# actual values to be used
for pcvTimestep in range(numberOfPcvTimesteps):
#    pcvTimestepId = pcvTimestep + 1 
    numberOfCyclesPast = int(pcvTimestep/numberOfPcvDataFiles)
    pcvTimestepId = pcvTimestep - numberOfCyclesPast*numberOfPcvDataFiles + 1
    filename=pcvDataDir + 'veldata' + str(pcvTimestepId) + '.pcv'
    print("Reading " + filename)
    try:
        with open(filename):
            f = open(filename,"r")
            numberOfVelocityPcvPoints=int(f.readline())
            pcvData[pcvTimestep,0:numberOfNonzeroPcvPoints,:] = numpy.loadtxt(f)*velocityScaleFactor
            f.close()
    except IOError:
       print ('Error reading pcv velocity data file: ' + filename)

    if outputExtras:
        outputFile = outputDir + "pcv/dataPoints" + str(pcvTimestep) + ".C"
        exnodeFile = outputDir + "pcv/dataPoints" + str(pcvTimestep) + ".exnode"
        fit.writeDataPoints(pcvGeometry,pcvData[pcvTimestep],outputFile)
        os.system("perl $scripts/meshConversion/dataPointsConversion.pl "+ outputFile + " 1000000 " + exnodeFile)

print(pcvData.shape)


#----------------------------------------------------------
# Spline interpolation of PCV data in time
#----------------------------------------------------------
# Spline interpolate time data for each velocity component 
# (this interpolates the data from the pcv timestep frequency to the
#  fitting output timesteps)
#----------------------------------------------------------
velocityDataPoints=numpy.zeros((numberOfTimesteps,numberOfPcvPoints,3))
#print(velocityData.shape)
pcvTimesteps = numpy.arange(startTime,stopTime+pcvTimeIncrement*2.0,pcvTimeIncrement)
#print(pcvTimesteps)
outputTimesteps = numpy.arange(startTime,stopTime,timeIncrement)
print("output timesteps: ")
print(outputTimesteps)
for dataPoint in range(numberOfNonzeroPcvPoints):
#    print("interpolating time values for data point: " + str(dataPoint+1) + " of " + str(numberOfPcvPoints))
    for component in range(numberOfDimensions):
        pointData = pcvData[:,dataPoint,component]
        tck = interpolate.splrep(pcvTimesteps,pointData,s=0)
        velocityDataPoints[:,dataPoint,component] = interpolate.splev(outputTimesteps,tck,der=0)

print("data point times interpolated. Array size:")
print(velocityDataPoints.shape)

plotOneDataPoint = True
if(plotOneDataPoint):
    if (computationalNodeNumber == 0):
        tempPcvTimesteps = numpy.arange(startTime,stopTime,pcvTimeIncrement)
        plotDataPoint = 166
#        plotDataPoint = 289
        ymin = -0.15
        ymax = 1.1

        component = 0
        print("plotting x-velocity point " + str(plotDataPoint))
        interpolatedDataU = velocityDataPoints[:,plotDataPoint,component]
        pylab.subplot(221)
        pylab.plot(outputTimesteps,interpolatedDataU,'-b.')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'ro')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'--g',lw=1.5)
        pylab.legend((r'Spline-interpolated values at 1ms', r'Velocimetry data values at 45ms', r'Linear interpolation'), shadow = True, loc = (0.1, 0.5))
        pylab.ylim(ymin,ymax)
        pylab.xlabel('time (s)')
        pylab.ylabel('x-velocity (m/s)')
#            pylab.title('Data point ' + str(plotDataPoint))
        pylab.grid(True)
        pylab.savefig('simple_plot')
#            pylab.show()

        component = 1
        print("plotting y-velocity point " + str(plotDataPoint))
        interpolatedDataV = velocityDataPoints[:,plotDataPoint,component]
        pylab.subplot(222)
        pylab.plot(outputTimesteps,interpolatedDataV,'-b.')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'ro')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'--g',lw=1.5)
#            pylab.legend((r'Spline-interpolated values at 1ms', r'Velocimetry data values at 45ms', r'Linear interpolation'), shadow = True, loc = (0.3, 0.7))
        pylab.ylim(ymin,ymax)
        pylab.xlabel('time (s)')
        pylab.ylabel('y-velocity (m/s)')
#            pylab.title('Data point ' + str(plotDataPoint))
        pylab.grid(True)
        pylab.savefig('simple_plot')
#            pylab.show()

        component = 2
        print("plotting z-velocity point " + str(plotDataPoint))
        interpolatedDataW = velocityDataPoints[:,plotDataPoint,component]
        pylab.subplot(223)
        pylab.plot(outputTimesteps,interpolatedDataW,'-b.')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'ro')
        pylab.plot(tempPcvTimesteps,pcvData[0:numberOfPcvTimesteps-2,plotDataPoint,component],'--g',lw=1.5)
#            pylab.legend((r'Spline-interpolated values at 1ms', r'Velocimetry data values at 45ms', r'Linear interpolation'), shadow = True, loc = (0.3, 0.7))
        pylab.ylim(ymin,ymax)
        pylab.xlabel('time (s)')
        pylab.ylabel('z-velocity (m/s)')
#            pylab.title('Data point ' + str(plotDataPoint))
        pylab.grid(True)
        pylab.savefig('simple_plot')
#            pylab.show()

        print("plotting velocity magnitude " + str(plotDataPoint))
        interpolatedDataMag=numpy.zeros(numberOfTimesteps)
        for timestep in range(numberOfTimesteps):
            interpolatedDataMag[timestep] = linalg.norm(velocityDataPoints[timestep,plotDataPoint,:])
        pylab.subplot(224)
        pylab.plot(outputTimesteps,interpolatedDataMag,'-b.')
        pcvDataMag=numpy.zeros(numberOfPcvTimesteps-2)
        for timestep in range(numberOfPcvTimesteps-2):
            pcvDataMag[timestep]=linalg.norm(pcvData[timestep,plotDataPoint,:])
        pylab.plot(tempPcvTimesteps,pcvDataMag,'ro')
        pylab.plot(tempPcvTimesteps,pcvDataMag,'--g',lw=1.5)
#            pylab.legend((r'Spline-interpolated values at 1ms', r'Velocimetry data values at 45ms', r'Linear interpolation'), shadow = True, loc = (0.3, 0.7))
        pylab.ylim(ymin,ymax)
        pylab.xlabel('time (s)')
        pylab.ylabel('velocity magnitude (m/s)')
#            pylab.title('Data point ' + str(plotDataPoint))
        pylab.grid(True)
        pylab.savefig('simple_plot')
        pylab.show()

#=================================================================
# Geometric Field
#=================================================================

print("Setting up Geometric Field")
# Create a field for the geometry
geometricField = CMISS.Field()
fieldmlInfo.InputFieldCreateStart(region,decomposition,geometricFieldUserNumber,
                                  geometricField,CMISS.FieldVariableTypes.U,
                                  "ArteryMesh.coordinates")
geometricField.CreateFinish()
fieldmlInfo.InputFieldParametersUpdate(geometricField,"ArteryMesh.node.coordinates",
                                       CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
fieldmlInfo.Finalise()

if outputExtras:
    # Export mesh geometry
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("Geometry","FORTRAN")
    fields.ElementsExport("Geometry","FORTRAN")
    fields.Finalise()
    print("Exported Geometric Mesh")

#=================================================================
# Dependent Field
#=================================================================

print("Setting up Dependent Field")
# Create dependent field (fitted values from data points)
dependentField = CMISS.Field()
dependentField.CreateStart(dependentFieldUserNumber,region)
dependentField.LabelSet("FittingField")
dependentField.TypeSet(CMISS.FieldTypes.GENERAL)
dependentField.MeshDecompositionSet(decomposition)
dependentField.GeometricFieldSet(geometricField)
dependentField.DependentTypeSet(CMISS.FieldDependentTypes.DEPENDENT)
dependentField.NumberOfVariablesSet(1)
dependentField.VariableTypesSet([CMISS.FieldVariableTypes.U])
dependentField.DimensionSet(CMISS.FieldVariableTypes.U,CMISS.FieldDimensionTypes.VECTOR)
dependentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.U,numberOfDimensions)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Velocity")
for dimension in range(numberOfDimensions):
    dimensionId = dimension+1
    dependentField.ComponentInterpolationSet(CMISS.FieldVariableTypes.U,dimensionId,CMISS.FieldInterpolationTypes.NODE_BASED)
dependentField.ScalingTypeSet(CMISS.FieldScalingTypes.NONE)
for dimension in range(numberOfDimensions):
    dimensionId = dimension + 1
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimensionId,meshComponent)
dependentField.CreateFinish()

# Initialise dependent field
for component in range(numberOfDimensions):
    componentId = component + 1
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,componentId,0.0)

if outputExtras:
    print('exporting dependent elems')
    # Export mesh geometry
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    fields.ElementsExport("Dependent","FORTRAN")
    fields.Finalise()

#=================================================================
# Inverse Distance Weighting
#=================================================================
p = 2.1
vicinityFactor = 1.15
nodeList = []
nodeLocations = numpy.zeros((numberOfNodes,numberOfDimensions))
nodeData = numpy.zeros((numberOfNodes,numberOfDimensions))
# Get node locations from the mesh topology
for node in xrange(numberOfNodes):
    nodeId = node + 1
    nodeNumber = nodes.UserNumberGet(nodeId)
    nodeNumberPython = nodeNumber - 1
    nodeDomain=decomposition.NodeDomainGet(nodeNumber,1)
    if (nodeDomain == computationalNodeNumber):
        nodeList.append(nodeNumberPython)
        for component in xrange(numberOfDimensions):
            componentId = component + 1
            value = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                         CMISS.FieldParameterSetTypes.VALUES,
                                                         1,1,nodeNumber,componentId)
            nodeLocations[nodeNumberPython,component] = value

# Calculate weights based on data point/node distance
print("Calculating geometric-based weighting parameters")
dataList = [[] for i in range(numberOfNodes+1)]
sumWeights = numpy.zeros((numberOfNodes+1))
weight = numpy.zeros((numberOfNodes+1,((vicinityFactor*2)**3 + numberOfWallNodes*3)))
fit.CalculateWeights(p,vicinityFactor,dataPointResolution,pcvGeometry,
                     nodeLocations,nodeList,wallNodes,dataList,weight,sumWeights)

# Apply weights to interpolate nodal velocity
for timestep in xrange(startFit,numberOfTimesteps):
    # Calculate node-based velocities
    velocityNodes = numpy.zeros((numberOfNodes,3))
    fit.VectorFit(velocityDataPoints[timestep],velocityNodes,nodeList,wallNodes,dataList,weight,sumWeights)

    # Update Field
    for nodeNumberPython in nodeList:
        nodeNumberCmiss = nodeNumberPython + 1
        for component in xrange(numberOfDimensions):
            componentId = component + 1
            value = velocityNodes[nodeNumberPython,component]
            dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                      CMISS.FieldParameterSetTypes.VALUES,
                                                      1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                      nodeNumberCmiss,componentId,value)

    # Export results
    timeStart = time.time()            
    print("exporting CMGUI data")
    # Export results
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    print('mesh name: ' + meshName)
    fields.NodesExport( outputDir +'/fittingResults/' + meshName + "_t"+ str(timestep),"FORTRAN")
    fields.Finalise()
    timeStop = time.time()            
    print("Finished CMGUI data export, time: " + str(timeStop-timeStart))

CMISS.Finalise()
