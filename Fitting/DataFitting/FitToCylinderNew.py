#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an example script to fit a data point vector field to a dependent field
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
import scipy
from scipy import optimize
import math
import re
import bisect
import time
import optparse
import idw

# Intialise OpenCMISS
from opencmiss import CMISS

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
    return

def parabolic(geometricPoint,radius,uMean,axialComponent):
    ''' returns analytic solution to a parabolic velocityprofile based
    on the geometric location, radius, and mean velocity'''

    velocityData = numpy.zeros(3)
    radialGeometry = numpy.empty_like(geometricPoint)
    radialGeometry[:] = geometricPoint
    radialGeometry[axialComponent] = 0.0
    r = linalg.norm(radialGeometry)
    velocityData[axialComponent]=-2.0*uMean*(1-(r**2/radius**2))
    if (r > radius):
        velocityData[:] = 0.0
    return(velocityData);

def solveFemWithParameters(sobelovParameters):
    '''Solve Finite element fit with specified values of tau and kappa'''
    tau=abs(sobelovParameters[0])
    kappa=abs(sobelovParameters[1])
    # Initialise dependent field to zero
    # for component in range(numberOfDimensions):
    #     componentId = component + 1
    #     #DEBUG
    #     if component == 2:
    #         value = -1.0
    #     else:
    #         value = 0.0            
    #     dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.
    #                                                FieldParameterSetTypes.VALUES,componentId,value)
    # Set kappa and tau - Sobelov smoothing parameters
    materialField.ParameterSetUpdateConstantDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,tau)
    materialField.ParameterSetUpdateConstantDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,kappa)
    #=================================================================
    # Solve 
    #=================================================================
    # Solve the problem
    print("Solving...")
    problem.Solve()
    #=================================================================
    # Analyse results
    #=================================================================
    AbsError = numpy.zeros((numberOfNodes,numberOfDimensions))
    ErrorMag = numpy.zeros((numberOfNodes))
    for node in range(numberOfNodes):
        nodeId = node + 1
        nodeDomain=decomposition.NodeDomainGet(nodeId,meshComponent)
        for component in range(numberOfDimensions):
            componentId = component+1
            AbsError[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(dependentField,CMISS.FieldVariableTypes.U,1,
                                                                                CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                                nodeId,componentId)
        ErrorMag[node] = linalg.norm(AbsError[node,:])
    RMSError = numpy.sqrt(numpy.mean(ErrorMag**2))

    print("------------------------------------\n\n")
    print("Tau: " + str(tau))
    print("Kappa: " + str(kappa))
    print("Error Array: ")
    print(AbsError)
    print("Error: " + str(RMSError))
    print("------------------------------------")
    return(RMSError);


# Set solve type flags
fitFem = False
fitIdw = False
optimise = False
quadraticMesh = False
tetMesh = False
addNoise = False
axialComponent = 2
length = 5.0
radius = 0.5
uMean = 1.0
Reynolds = 1.0
fefIterations = 1

# Init fitting params
p = 2.0
vicinityFactor = 2.0
# Set sobelov smoothing parameters
tau = 0.0001
kappa = 0.005

parser = optparse.OptionParser()
parser.add_option("-k","--kappa_fef_parameter", action="store", type="float", dest="kappa",
                   help='curvature parameter for FEF', default=kappa)
parser.add_option("-u","--tau_fef_parameter", action="store", type="float", dest="tau",
                   help='tension parameter for FEF', default=kappa)
parser.add_option("-p","--p_idw_parameter", action="store", type="float", dest="p",
                   help='power parameter for IDW fit', default=p)
parser.add_option("-v","--VF_idw_parameter", action="store", type="float", dest="vicinityFactor",
                   help='power parameter for IDW fit', default=vicinityFactor)
parser.add_option("-r","--resolution", action="store", type="int", dest="diameterResolution",
                   help='number of data points across the mesh diameter', default=5)
parser.add_option("-a","--fefIterations", action="store", type="int", dest="fefIterations",
                   help='number of iterations to run the FEF procedure', default=1)
parser.add_option("-q", "--quadratic", default=quadraticMesh, action="store_true", help="flag whether quadratic mesh. default=False")
parser.add_option("-t", "--tetrahedral", default=tetMesh, action="store_true", help="flag whether tetrahedral mesh. default=False (hexahedral mesh)")
parser.add_option("-o", "--optimise", default=optimise, action="store_true", help="flag whether to optimise parameters. default=False")
parser.add_option("-i", "--fitIdw", default=fitIdw, action="store_true", help="flag to turn off IDW fit. default=True")
parser.add_option("-f", "--fitFem", default=fitFem, action="store_true", help="flag to turn off FEM fit. default=True")
parser.add_option("-z", "--addZeroLayer", default=False, action="store_true", help="flag whether to add zero layer to IDW fit. default=True")
parser.add_option("-j", "--setProjectionBoundariesIdw", default=False, action="store_true", help="flag whether to use projection boundaries for IDW fit. default=True")
parser.add_option("-m", "--highResMesh", default=False, action="store_true", help="flag to use a high resolution mesh. default=False")
parser.add_option("-l", "--lowResMesh", default=False, action="store_true", help="flag to use a low resolution mesh. default=False")
parser.add_option("-e", "--useExternalModule", default=False, action="store_true", help="flag to use the idw module rather than in-source method. default=False")
parser.add_option("-n", "--SNR", action="store", type="float", dest="SNR", default=0.,
                   help='signal to noise ratio: (no noise if not flagged)')

(options, args) = parser.parse_args()
diameterResolution = options.diameterResolution
tetMesh = options.tetrahedral
quadraticMesh = options.quadratic
highResMesh = options.highResMesh
lowResMesh = options.lowResMesh
useExternalModule = options.useExternalModule
optimise = options.optimise
SNR = options.SNR
fitIdw = options.fitIdw
fitFem = options.fitFem
fefIterations = options.fefIterations
addZeroLayer = options.addZeroLayer
setProjectionBoundariesIdw = options.setProjectionBoundariesIdw
p = options.p
vicinityFactor = options.vicinityFactor
tau = options.tau
kappa = options.kappa
print("options: "+ str(options))

if (abs(SNR) > 0.0001):
    addNoise = True

meshTypes = {
    'hexCylinder7Linear' : 'Lin Hex',
    'hexCylinder7Quadratic' : 'Quad Hex',
    'tetCyl2Linear' : 'Lin Tet',
    'tetCyl2Quadratic' : 'Quad Tet',
}

# Set up mesh
if (tetMesh):
    if (quadraticMesh):
        meshName = 'tetCyl2Quadratic'
    else:
        meshName = 'tetCyl2Linear'
    inputDir = './input/tetCyl2/'
else:
    if (quadraticMesh):
        if (highResMesh):
            meshName = 'hexCylinder9Quadratic'            
            inputDir = './input/hexCylinder9/'
            axialComponent = 1
        elif (lowResMesh):
            meshName = 'hexCylinder11Quadratic'            
            inputDir = './input/hexCylinder11/'
            axialComponent = 2
        else:
            #meshName = 'hexCylinder10Quadratic'
            #inputDir = './input/hexCylinder10/'
            #axialComponent = 1
            meshName = 'hexCylinder7Quadratic'
            inputDir = './input/hexCylinder7/'
            axialComponent = 2
    else:
        meshName = 'hexCylinder7Linear'
        inputDir = './input/hexCylinder7/'
fieldmlInput = inputDir + meshName + '.xml'
print('input file: ' + fieldmlInput)

#Wall boundary nodes
filename=inputDir + 'bc/wallNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfWallNodes=int(f.readline())
        wallNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Wall boundary node file: ' + filename)


# Set data point grid dimensions
lengthResolution = diameterResolution*int(length)
dataResolution = [diameterResolution,diameterResolution,diameterResolution]
dataResolution[axialComponent] = lengthResolution

print("candidate data resolution: " + str(dataResolution))

setBoundaries = True

(coordinateSystemUserNumber,
    regionUserNumber,
    basisUserNumber,
    generatedMeshUserNumber,
    meshUserNumber,
    decompositionUserNumber,
    geometricFieldUserNumber,
    dummyCoordinateSystemUserNumber,
    dummyRegionUserNumber,
    dummyBasisUserNumber,
    dummyGeneratedMeshUserNumber,
    dummyMeshUserNumber,
    dummyDecompositionUserNumber,
    dummyGeometricFieldUserNumber,
    dummySourceFieldUserNumber,
    equationsSetFieldUserNumber,
    dependentFieldUserNumber,
    independentFieldUserNumber,
    dataPointFieldUserNumber,
    materialFieldUserNumber,
    analyticFieldUserNumber,
    dependentDataFieldUserNumber,
    dataProjectionUserNumber,
    equationsSetUserNumber,
    problemUserNumber) = range(1,26)

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()

#Initialise fieldML IO
fieldmlInfo=CMISS.FieldMLIO()
fieldmlInfo.InputCreateFromFile(fieldmlInput)

# Creation a RC coordinate system
coordinateSystem = CMISS.CoordinateSystem()
fieldmlInfo.InputCoordinateSystemCreateStart("CylinderMesh.coordinates",coordinateSystem,coordinateSystemUserNumber)
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
fieldmlInfo.InputNodesCreateStart("CylinderMesh.nodes.argument",region,nodes)
nodes.CreateFinish()
numberOfNodes = nodes.numberOfNodes
if (quadraticMesh):
    basisNumberQuadratic = 1
    basisNumberLinear = 2
    meshComponentQuadratic = 1
    meshComponentLinear = 2
else:
    basisNumberQuadratic = 2
    basisNumberLinear = 1
    meshComponentQuadratic = 2
    meshComponentLinear = 1
print("number of nodes: " + str(numberOfNodes))

gaussQuadrature = [3,3,3]
quadratureOrder = 5

if (tetMesh):
    if (quadraticMesh):
        fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.triquadratic_simplex",basisNumberQuadratic)
        CMISS.Basis_QuadratureOrderSetNum(basisNumberQuadratic,quadratureOrder)
        CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
    else:
        fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.trilinear_simplex",basisNumberLinear)
        CMISS.Basis_QuadratureOrderSetNum(basisNumberLinear,quadratureOrder)
        CMISS.Basis_CreateFinishNum(basisNumberLinear)
else:
    if (quadraticMesh):
        fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.triquadratic_lagrange",basisNumberQuadratic)
        CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
        CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
        CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
    else:
        fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.trilinear_lagrange",basisNumberLinear)
        CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberLinear,gaussQuadrature)
        CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberLinear,True)
        CMISS.Basis_CreateFinishNum(basisNumberLinear)

# Create Mesh
numberOfMeshComponents=1
meshComponent=1

mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("CylinderMesh.mesh.argument",mesh,meshUserNumber,region)
mesh.NumberOfComponentsSet(numberOfMeshComponents)
if (quadraticMesh):
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponentQuadratic,"CylinderMesh.template.triquadratic")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponentLinear,"CylinderMesh.template.trilinear")
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

#=================================================================
# Data Points
#=================================================================

# Create a numpy array of data point locations
dataPointLocations = []
dataResolutionIncrement = [2.0*radius/float(dataResolution[0]+1),
                           2.0*radius/float(dataResolution[1]+1),
                           2.0*radius/float(dataResolution[2]+1)]
dataResolutionIncrement[axialComponent] = length/float(dataResolution[axialComponent]+1)
dataInit = [-radius,-radius,-radius]
#dataInit[axialComponent] = -dataResolutionIncrement[axialComponent]
dataInit[axialComponent] = dataResolutionIncrement[axialComponent]/2.0

print("Data resolution increment: " + str(dataResolutionIncrement))
print("Data init: " + str(dataInit))

zeroTolerance = 1.0e-8
if (addZeroLayer):
    inc = numpy.array(dataResolutionIncrement)
    init = numpy.array(dataInit)
    dataInit = init - inc
    dataResolution = [i + 4 for i in dataResolution]
    print(dataResolution)
    dataPointZ = dataInit[2]
    for z in range(dataResolution[2]):
        dataPointZ = dataPointZ + dataResolutionIncrement[2]
        dataPointY = dataInit[1]
        for y in range(dataResolution[1]):
            dataPointY = dataPointY + dataResolutionIncrement[1]
            dataPointX = dataInit[0]
            for x in range(dataResolution[0]):
                dataPointX = dataPointX + + dataResolutionIncrement[0]
                radialPosition = [dataPointX,dataPointY,dataPointZ]
                radialPosition[axialComponent] = 0.0
                if (linalg.norm(radialPosition) <= radius + 2.0*radius/float(dataResolution[0]+1) + zeroTolerance):
    #                print("Data Point position: " + str([dataPointX,dataPointY,dataPointZ]))
                    dataPointLocations.append([dataPointX,dataPointY,dataPointZ])
#    dataResolution = [i - 4 for i in dataResolution]
else:
    #dataResolution[axialComponent] += 2
    dataPointZ = dataInit[2]
    for z in range(dataResolution[2]):
        dataPointZ = dataPointZ + dataResolutionIncrement[2]
        dataPointY = dataInit[1]
        for y in range(dataResolution[1]):
            dataPointY = dataPointY + dataResolutionIncrement[1]
            dataPointX = dataInit[0]
            for x in range(dataResolution[0]):
                dataPointX = dataPointX + + dataResolutionIncrement[0]
                radialPosition = [dataPointX,dataPointY,dataPointZ]
                radialPosition[axialComponent] = 0.0
                if (linalg.norm(radialPosition) < radius):
    #                print("Data Point position: " + str([dataPointX,dataPointY,dataPointZ]))
                    dataPointLocations.append([dataPointX,dataPointY,dataPointZ])
    dataResolution[axialComponent] -= 2

numberOfDataPoints=len(dataPointLocations)
print("number of candidate data points: " + str(dataResolution[0]*dataResolution[1]*dataResolution[2]))
print("number of embedded data points: " + str(numberOfDataPoints))
dataPointLocations = numpy.array(dataPointLocations)
velocityData=numpy.zeros((numberOfDataPoints,numberOfDimensions))

for point in range(numberOfDataPoints):
    velocityData[point,:]=parabolic(dataPointLocations[point,:],radius,uMean,axialComponent)
#    print("Data Point position: " + str(dataPointLocations[point,:]))

if (addNoise):
    noise = numpy.zeros((numberOfDataPoints,3))
    percentError = numpy.zeros((numberOfDataPoints,3))
    analyticDataPointVelocity=numpy.zeros((numberOfDataPoints,numberOfDimensions))
    dataPointNoiseError=numpy.zeros((numberOfDataPoints,numberOfDimensions))
    dataPointNoisePercentError=numpy.zeros((numberOfDataPoints,numberOfDimensions))
    dataPointMagErrorBusch=numpy.zeros((numberOfDataPoints))
    dataPointDirErrorBusch=numpy.zeros((numberOfDataPoints))
    signalMean = numpy.mean(velocityData[:,axialComponent])
    signalMean = numpy.mean(velocityData)
    SDsignal = abs(signalMean/SNR)
    for component in range(3):
        noise[:,component] = numpy.random.normal(0.0,SDsignal,numberOfDataPoints)
    print("noise mean: " + str(numpy.mean(noise)))
    analyticVelocityDataPoints = numpy.copy(velocityData)
    # Add noise to non-zerolayer data points
    print(velocityData)
    for dataPoint in range(numberOfDataPoints):
        if (abs(velocityData[dataPoint,axialComponent]) > zeroTolerance):
            velocityData[dataPoint,:] += noise[dataPoint,:]
    dataPointNoiseError = abs(noise)
    dataPointNoiseErrorRMS = numpy.sqrt(numpy.mean(dataPointNoiseError**2))
    dataPointNoiseErrorSD = numpy.std(dataPointNoiseError)
    print("RMS noise data points: " + str(dataPointNoiseErrorRMS))
    print("SD noise data points: " + str(dataPointNoiseErrorSD))
    for point in range(numberOfDataPoints):
        dataPointMagErrorBusch[point] = numpy.linalg.norm(noise[point,:]/(analyticVelocityDataPoints[point,:]+zeroTolerance))
        dataPointDirErrorBusch[point] = 1. - numpy.absolute(numpy.dot(analyticVelocityDataPoints[point,:],velocityData[point,:]))/(numpy.dot(numpy.absolute(analyticVelocityDataPoints[point,:]+zeroTolerance),numpy.absolute(velocityData[point,:])))

    rmsDataBuschDir = numpy.sqrt(numpy.mean(dataPointDirErrorBusch**2))

    print('Busch dir error: ')
    print(rmsDataBuschDir)

# Set up data points with geometric values
dataPoints = CMISS.DataPoints()
dataPoints.CreateStart(region,numberOfDataPoints)
for dataPoint in range(numberOfDataPoints):
    dataPointId = dataPoint + 1
    dataList = dataPointLocations[dataPoint,:]
    dataPoints.ValuesSet(dataPointId,dataList)
dataPoints.CreateFinish()

if addNoise:
    dataPointFilename = "dataPointsSNR" + str(SNR) + '.C'
    writeDataPoints(dataPointLocations,velocityData,dataPointFilename)
    dataPointOutputFilename = "dataPointsSNR" + str(SNR) + '.exnode'
    os.system("perl $scripts/meshConversion/dataPointsConversion.pl " + dataPointFilename + " 1000000 " +dataPointOutputFilename)
else:
    writeDataPoints(dataPointLocations,velocityData,"dataPoints.C")
    os.system("perl $scripts/meshConversion/dataPointsConversion.pl dataPoints.C 1000000 dataPoints.exnode")
    print('Data point writing completed')

#=================================================================
# Geometric Field
#=================================================================

print("Setting up Geometric Field")
# Create a field for the geometry
geometricField = CMISS.Field()
if (quadraticMesh):
    fieldmlInfo.InputFieldCreateStart(region,decomposition,geometricFieldUserNumber,
                                      geometricField,CMISS.FieldVariableTypes.U,
                                      "CylinderMesh.coordinates")
    geometricField.CreateFinish()
    fieldmlInfo.InputFieldParametersUpdate(geometricField,"CylinderMesh.node.coordinates",
                                           CMISS.FieldVariableTypes.U,
                                           CMISS.FieldParameterSetTypes.VALUES)
    fieldmlInfo.Finalise()
else:
    fieldmlInfo.InputFieldCreateStart(region,decomposition,geometricFieldUserNumber,
                                      geometricField,CMISS.FieldVariableTypes.U,
                                      "CylinderMesh.coordinates")
    geometricField.CreateFinish()
    fieldmlInfo.InputFieldParametersUpdate(geometricField,"CylinderMesh.node.coordinates",
                                           CMISS.FieldVariableTypes.U,
                                           CMISS.FieldParameterSetTypes.VALUES)
    fieldmlInfo.Finalise()

# # Export mesh geometry
# fields = CMISS.Fields()
# fields.CreateRegion(region)
# fields.NodesExport("Geometry","FORTRAN")
# fields.ElementsExport("Geometry","FORTRAN")
# fields.Finalise()
# print("Exported Geometric Mesh")

#=================================================================
# Data Projection on Geometric Field
#=================================================================

# Set up data projection
dataProjection = CMISS.DataProjection()
dataProjection.CreateStart(dataProjectionUserNumber,dataPoints,mesh)
dataProjection.projectionType = CMISS.DataProjectionProjectionTypes.ALL_ELEMENTS
dataProjection.CreateFinish()

timeProjectionStart = time.time()

print('Evaluating data projection')
# Evaluate data projection based on geometric field
dataProjection.ProjectionEvaluate(geometricField)

print('Calculating data projection')
# Create mesh topology for data projection
mesh.TopologyDataPointsCalculateProjection(dataProjection)
# Create decomposition topology for data projection
decomposition.TopologyDataProjectionCalculate()
timeProjectionFinish = time.time()
dataProjectionTime = timeProjectionFinish - timeProjectionStart
print("Data projection finished- time: " + str(dataProjectionTime))

for dataPoint in range(numberOfDataPoints):
    dataPointId = dataPoint+1
    xi = dataProjection.ResultXiGet(dataPointId,3)
    nanFlag = False
    boundaryFlag = False
    tolerance = 1e-6
    for i in xrange(3):
        if (math.isnan(xi[i])):
            nanFlag = True
            print("projection xi NaN for data point: " + str(dataPointId))
            print(xi)
        if ((abs(xi[i]) < tolerance) or ((abs(xi[i] - 1.0)) < tolerance)):
            boundaryFlag = True
    if (nanFlag):
        dataPointLocation = dataPoints.ValuesGet(dataPointId,3)
        print("location: " + str(dataPointLocation))

#=================================================================
# Equations Set
#=================================================================

print("Setting up equations set")
# Create vector fitting equations set
equationsSetField = CMISS.Field()
equationsSet = CMISS.EquationsSet()
equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
        CMISS.EquationsSetClasses.FITTING,
        CMISS.EquationsSetTypes.DATA_FITTING_EQUATION,
        CMISS.EquationsSetSubtypes.DATA_POINT_VECTOR_STATIC_FITTING,
        equationsSetFieldUserNumber, equationsSetField)
equationsSet.CreateFinish()

#=================================================================
# Dependent Field
#=================================================================

print("Setting up Dependent Field")
# Create dependent field (fitted values from data points)
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"FemFit")
equationsSet.DependentCreateFinish()
# Initialise dependent field
for component in range(numberOfDimensions):
    componentId = component + 1
    #DEBUG
    if component == axialComponent:
        value = -1.0
    else:
        value = 0.0            
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,componentId,value)

# Analytic values to test against - based on geometry field
#-----------------------------------------------------------
dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.U,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)
dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.DELUDELN,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)

dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                           1,0.0)
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.DELUDELN,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                           1,0.0)

#set node-based analytic field based on parabolic profile
print('setting up analytic field')
if fitFem:
    for node in range(numberOfNodes):
        nodeId = node + 1
        value = numpy.zeros(3)
        meshPosition = numpy.zeros((3))
        for component in range(numberOfDimensions):
            componentId = component+1
            geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                CMISS.FieldParameterSetTypes.VALUES,
                                                                1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
            meshPosition[component] = geometricValue
        value = parabolic(meshPosition,radius,uMean,axialComponent)
        for component in range(numberOfDimensions):
            componentId = component+1        
            dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                   CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                                   1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                    nodeId,componentId,value[component])

#=================================================================
# Independent Field
#=================================================================

print("Setting up Independent Field")
independentField = CMISS.Field()
independentField.CreateStart(independentFieldUserNumber,region)
independentField.LabelSet("Independent")
independentField.TypeSet(CMISS.FieldTypes.GENERAL)
independentField.MeshDecompositionSet(decomposition)
independentField.GeometricFieldSet(geometricField)
independentField.DependentTypeSet(CMISS.FieldDependentTypes.INDEPENDENT)

independentField.NumberOfVariablesSet(4)
independentField.VariableTypesSet([CMISS.FieldVariableTypes.U,
                                CMISS.FieldVariableTypes.V,
                                  CMISS.FieldVariableTypes.U1,
                                  CMISS.FieldVariableTypes.U2])
independentField.DimensionSet(CMISS.FieldVariableTypes.U,CMISS.FieldDimensionTypes.VECTOR)
independentField.DimensionSet(CMISS.FieldVariableTypes.V,CMISS.FieldDimensionTypes.SCALAR)
independentField.DimensionSet(CMISS.FieldVariableTypes.U1,CMISS.FieldDimensionTypes.VECTOR)
independentField.DimensionSet(CMISS.FieldVariableTypes.U2,CMISS.FieldDimensionTypes.VECTOR)
independentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.U,numberOfDimensions)
independentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.V,1)
independentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.U1,numberOfDimensions)
independentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.U2,numberOfDimensions)
independentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"dataPointVector")
independentField.VariableLabelSet(CMISS.FieldVariableTypes.V,"dataPointWeight")
independentField.VariableLabelSet(CMISS.FieldVariableTypes.U1,"IdwFit")
independentField.VariableLabelSet(CMISS.FieldVariableTypes.U2,"Analytic")

print('debug: independent 2')
independentField.ComponentInterpolationSet(CMISS.FieldVariableTypes.V,1,CMISS.FieldInterpolationTypes.DATA_POINT_BASED)
for dimension in range(numberOfDimensions):
    dimensionId = dimension+1
    independentField.ComponentInterpolationSet(CMISS.FieldVariableTypes.U,dimensionId,CMISS.FieldInterpolationTypes.DATA_POINT_BASED)
    independentField.ComponentInterpolationSet(CMISS.FieldVariableTypes.U1,dimensionId,CMISS.FieldInterpolationTypes.NODE_BASED)
    independentField.ComponentInterpolationSet(CMISS.FieldVariableTypes.U2,dimensionId,CMISS.FieldInterpolationTypes.NODE_BASED)

independentField.ScalingTypeSet(CMISS.FieldScalingTypes.NONE)

print('debug: independent 3')
independentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,1,meshComponent)
for dimension in range(numberOfDimensions):
    dimensionId = dimension + 1
    independentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimensionId,meshComponent)
    independentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,dimensionId,meshComponent)
    independentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,dimensionId,meshComponent)

independentField.DataProjectionSet(dataProjection)

print('debug: independent 4')
independentField.CreateFinish()
equationsSet.IndependentCreateStart(independentFieldUserNumber,independentField)
equationsSet.IndependentCreateFinish()
# Initialise independent field

print('debug: independent 5')
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,1,1.0)
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,CMISS.FieldParameterSetTypes.VALUES,1,0.0)
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U2,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

print('setting analytic values')
# Analytic values
#-----------------------------------------------------------
independentField.ParameterSetCreate(CMISS.FieldVariableTypes.U1,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)
independentField.ParameterSetCreate(CMISS.FieldVariableTypes.U2,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)

independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                           1,0.0)
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U2,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                           1,0.0)

#set node-based analytic field to function of geometric field
for node in range(numberOfNodes):
    nodeId = node + 1
    nodeDomain=decomposition.NodeDomainGet(nodeId,meshComponent)
    if (nodeDomain == computationalNodeNumber):
        value = numpy.zeros(3)
        meshPosition = numpy.zeros((3))
        for component in range(numberOfDimensions):
            componentId = component+1
            geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                CMISS.FieldParameterSetTypes.VALUES,
                                                                1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
            meshPosition[component] = geometricValue
        value = parabolic(meshPosition,radius,uMean,axialComponent)
        for component in range(numberOfDimensions):
            componentId = component+1        
            independentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U1,
                                                   CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                                   1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                    nodeId,componentId,value[component])
            independentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U2,
                                                   CMISS.FieldParameterSetTypes.VALUES,
                                                   1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                    nodeId,componentId,value[component])
            independentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U2,
                                                   CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                                   1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                    nodeId,componentId,value[component])

# Set independent field U values to parabolic profile at data points,
#    V values to user-specified weights (just 1 here as these are all embedded points)
#-----------------------------------------------------------------------------------------
for element in range(numberOfElements):
    elementId = element + 1
    elementDomain = decomposition.ElementDomainGet(elementId)
    if (elementDomain == computationalNodeNumber):
        numberOfProjectedDataPoints = decomposition.TopologyNumberOfElementDataPointsGet(elementId)
        for dataPoint in range(numberOfProjectedDataPoints):
            dataPointId = dataPoint + 1
            dataPointNumber = decomposition.TopologyElementDataPointUserNumberGet(elementId,dataPointId)
            # set data point field values
            for component in range(numberOfDimensions):
                componentId = component + 1
                dataPointNumberIndex = dataPointNumber - 1
                value = velocityData[dataPointNumberIndex,component]
                independentField.ParameterSetUpdateElementDataPointDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,elementId,dataPointId,componentId,value)

            # Set data point weights
            value = 1.0
            independentField.ParameterSetUpdateElementDataPointDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,elementId,dataPointId,1,value)

print('exporting dependent elems')
# Export mesh geometry
fields = CMISS.Fields()
fields.CreateRegion(region)
fields.ElementsExport("Dependent","FORTRAN")
fields.Finalise()

#=================================================================
# Inverse Distance Weighting
#=================================================================

#if (fitIdw):

def solveIdwWithParameters(idwParameters):
    '''Solve IDW'''

    p=idwParameters[0]
    vicinityFactor=idwParameters[1]
    timeIdwStart = time.time()        

    # IDW (based on geometry) for node-based field
    errorData = numpy.zeros((numberOfNodes,3))
    errorFitMagDataBusch = numpy.zeros((numberOfNodes))
    errorFitDirDataBusch = numpy.zeros((numberOfNodes))
    setBoundariesIdw = False

    dataList = [[] for i in range(numberOfNodes+1)]
    sumWeights = numpy.zeros((numberOfNodes+1))
    analyticData = numpy.zeros((numberOfNodes,3))

    timeIdwStart = time.time()        
    print("Calculating Geometric position based Inverse Distance Weights. proc: " + str(computationalNodeNumber))
    if (setBoundariesIdw):
        wallNodeLocations = []
        for node in (wallNodes):
            nodeId = node
            nodeNumber = nodes.UserNumberGet(nodeId)
            meshPosition = []
            nodeDomain=decomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                # get the geometric position for this wall node
                for component in xrange(numberOfDimensions):
                    componentId = component + 1
                    geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                        CMISS.FieldParameterSetTypes.VALUES,
                                                                        1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                        nodeNumber,componentId)
                    meshPosition.append(geometricValue)
                wallNodeLocations.append(meshPosition)
        wallNodeLocations = numpy.array(wallNodeLocations)
        idwDataPointLocations = numpy.vstack((dataPointLocations,wallNodeLocations))
        wallVelocity = numpy.zeros((numberOfWallNodes,3))
        idwVelocityData = numpy.vstack((velocityData,wallVelocity))
        weight = numpy.zeros((numberOfNodes+1,((vicinityFactor*2)**3 + numberOfWallNodes)))
        idwNumberOfDataPoints = numberOfDataPoints + numberOfWallNodes
    else:
        idwDataPointLocations = dataPointLocations
        idwVelocityData = velocityData
        if(setProjectionBoundariesIdw):       
            weight = numpy.zeros((numberOfNodes+1,2*((vicinityFactor*2)**3 + numberOfWallNodes*3)))
        else:
            weight = numpy.zeros((numberOfNodes+1,((vicinityFactor*2)**3 + numberOfWallNodes*3)))
        idwNumberOfDataPoints = numberOfDataPoints

    fastSearch = False
    if fastSearch:
        # This does a bit of array sorting by the first index to speed this up
        dataIndex = numpy.argsort(idwDataPointLocations[:,0])
        sortedData = idwDataPointLocations[dataIndex]
        numberOfSearchFailures =0

    for node in xrange(numberOfNodes):
        nodeId = node + 1
        nodeNumber = nodes.UserNumberGet(nodeId)
        meshPosition = numpy.zeros((3))
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            #set values to zero if wall node
            if(nodeNumber not in wallNodes[:]):
                # get the geometric position for this node
                for component in xrange(numberOfDimensions):
                    componentId = component + 1
                    value = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                 CMISS.FieldParameterSetTypes.VALUES,
                                                                 1,1,nodeNumber,componentId)
                    meshPosition[component] = value

                analyticData[node,:] = parabolic(meshPosition,radius,uMean,axialComponent)
                if(fastSearch):
                    minX = bisect.bisect_left(sortedData[:,0],meshPosition[0]-vicinityFactor*dataResolution[0])
                    maxX = bisect.bisect_right(sortedData[:,0],meshPosition[0]+vicinityFactor*dataResolution[0])
                    numberOfCandidates = maxX - minX
                    dataPoint = -1
                    for candidate in xrange(numberOfCandidates):
                        if(abs(sortedData[minX+candidate,0] - meshPosition[0]) < vicinityFactor*dataResolution[0] and
                           abs(sortedData[minX+candidate,1] - meshPosition[1]) < vicinityFactor*dataResolution[1] and
                           abs(sortedData[minX+candidate,2] - meshPosition[2]) < vicinityFactor*dataResolution[2]):
                            dataPoint = dataIndex[minX+candidate]
                            dataList[nodeNumber].append(dataPoint)
                            difference = numpy.subtract(idwDataPointLocations[dataPoint,:], meshPosition)
                            distance = linalg.norm(difference,2)
                            weight[nodeNumber,dataPoint] = 1.0/(distance**p)
                            sumWeights[nodeNumber] += weight[nodeNumber,dataPoint]
                    if dataPoint == -1:
                        print("Could not find data points in vicinity of mesh node: " + str(nodeId) + "\n")
                        print("Tolerances:  relative: " + str(relativeFitTolerance) + "     absolute: " + str(absoluteFitTolerance) + "\n")
                        numberOfSearchFailures += 1

                else:
                    numberOfSearchFailures = 0
                    #find indices of data points within a distance tolerance of this node
                    fittingPoint=0
                    for dataPoint in xrange(numberOfDataPoints):
                        dataPointId = dataPoint + 1
                        if(abs(idwDataPointLocations[dataPoint,0] - meshPosition[0]) < vicinityFactor*dataResolutionIncrement[0] and
                           abs(idwDataPointLocations[dataPoint,1] - meshPosition[1]) < vicinityFactor*dataResolutionIncrement[1] and
                           abs(idwDataPointLocations[dataPoint,2] - meshPosition[2]) < vicinityFactor*dataResolutionIncrement[2]):

                            #within tolerance - save dataPoint to list
                            dataList[nodeNumber].append(dataPoint)
                            difference = numpy.subtract(idwDataPointLocations[dataPoint,:], meshPosition)
                            distance = linalg.norm(difference,2)
                            weight[nodeNumber,fittingPoint] = 1.0/((distance+zeroTolerance)**p)
                            sumWeights[nodeNumber] += weight[nodeNumber,fittingPoint]
                            fittingPoint+=1

    if(setProjectionBoundariesIdw):       
        # add additional data points with zero wall values for nodes within a tolerance of the wall
        for node in xrange(numberOfNodes):
            nodeId = node + 1
            nodeNumber = nodes.UserNumberGet(nodeId)
            meshPosition = numpy.zeros((3))
            nodeDomain=decomposition.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                #set values to zero if wall node
                if(nodeNumber not in wallNodes[:]):
                    # get the geometric position for this node
                    for component in xrange(numberOfDimensions):
                        componentId = component + 1
                        value = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                     CMISS.FieldParameterSetTypes.VALUES,
                                                                     1,1,nodeNumber,componentId)
                        meshPosition[component] = value

                    radialPosition = []
                    radialResInc = 0.0
                    for i in range(3):
                        if (i != axialComponent):
                            radialPosition.append(meshPosition[i])
                            radialResInc += dataResolution[i]**2.
                    radialResInc = math.sqrt(radialResInc)
                    radialDistance = linalg.norm(radialPosition)
                    distanceFromWall = radius - radialDistance

                    if(distanceFromWall < vicinityFactor*radialResInc):
                       fittingPoint = numpy.size(numpy.nonzero(weight[nodeNumber,:]))
                       dataPoint = idwNumberOfDataPoints
                       dataList[nodeNumber].append(dataPoint)
                       weight[nodeNumber,fittingPoint] = 1.0/((distanceFromWall + zeroTolerance)**p)
                       sumWeights[nodeNumber] += weight[nodeNumber,fittingPoint]
                       idwNumberOfDataPoints += 1

    print("Added projection wall data points: " + str(idwNumberOfDataPoints - numberOfDataPoints))        

    t2 = time.time()        
    sys.stdout.write("\n")
    print("Time to do geometric fit (s): " + str(t2-timeIdwStart))
    print("total number of search failures: " + str(numberOfSearchFailures))
    print("Fitting Velocity data")

    # Use geometry weights to define time-dependent node based vector values
    initialVelocity = numpy.zeros((numberOfNodes,3))
    nodeData = numpy.zeros((numberOfNodes,3))
    outputData = numpy.zeros((numberOfNodes,3))
    for node in xrange(numberOfNodes):
        nodeId = node + 1
        nodeNumber = nodes.UserNumberGet(nodeId)
        nodeVector = numpy.zeros((3))
        velocityVector = numpy.zeros((3))
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            if (nodeNumber in wallNodes[:]):
                nodeVector[:] = 0.0
                nodeData[node,:] = 0.0
            else:
                for fittingPoint in xrange(len(dataList[nodeNumber])):
                    dataPoint = dataList[nodeNumber][fittingPoint]
                    if(setProjectionBoundariesIdw and dataPoint >= numberOfDataPoints):
                        velocityVector[:] = 0.0
                    else:
                        velocityVector[:] = idwVelocityData[dataPoint,:]
                    nodeVector[:] += weight[nodeNumber,fittingPoint]*velocityVector[:]/sumWeights[nodeNumber]

                # vector for this node is the sum of the contributing fitting point vectors
                nodeData[node,:] = nodeVector[:]

            for component in range(numberOfDimensions):
                componentId = component + 1
                independentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U1,
                                                          CMISS.FieldParameterSetTypes.VALUES,
                                                          1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                          nodeId,componentId,nodeVector[component])

    meanVel =  numpy.apply_along_axis(numpy.mean, 0, nodeData)
    idwMean = meanVel[axialComponent]
    print('IDW Mean: ' + str(idwMean))
    timeIdwStop = time.time()        
    timeIdw = timeIdwStop - timeIdwStart
    print("finished idw fit, time to fit: " + str(timeIdw))

    velocityErrors = nodeData - analyticData
    velocityNormErrors =  numpy.apply_along_axis(numpy.linalg.norm, 1, velocityErrors)
    idwRMSError = numpy.sqrt(numpy.mean(velocityNormErrors**2))
    sdIdw = numpy.std(velocityNormErrors)

    print("P = " + str(p))
    print("Vicinity factor = " + str(vicinityFactor))
    print("node data: ")
    print(nodeData)
    print("analytic data: ")
    print(analyticData)
    print("error data: ")
    print(errorData)
    print("RMS Error: " + str(idwRMSError))
    print("Error Standard Deviation: " + str(sdIdw))
    print("----------------------------------\n\n")
    return(idwRMSError);

if (fitIdw):
    if useExternalModule:
        nodeList = []
        nodeLocations = numpy.zeros((numberOfNodes,numberOfDimensions))
        nodeData = numpy.zeros((numberOfNodes,numberOfDimensions))
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

        idw.idwFit(p,vicinityFactor,dataPointLocations,velocityData,
                   nodeLocations,nodeData,nodeList,wallNodes,dataResolutionIncrement)

        for nodeNumberPython in nodeList:
            nodeNumberCmiss = nodeNumberPython + 1
            for component in xrange(numberOfDimensions):
                componentId = component + 1
                value = nodeData[nodeNumberPython,component]
                independentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U1,
                                                          CMISS.FieldParameterSetTypes.VALUES,
                                                          1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                          nodeNumberCmiss,componentId,value)
    else:
        if (optimise):
            ranges = (slice(1.0,3.0,0.1),slice(1.0,3.0,0.1))
            resultIdw = optimize.brute(solveIdwWithParameters,ranges)
            print(resultIdw)        
            p = resultIdw[0]
            vicinityFactor = resultIdw[1]
            idwVelRMSE = solveIdwWithParameters([p,vicinityFactor])

        else:
            timeIdwStart = time.time()        
            idwVelRMSE = solveIdwWithParameters([p,vicinityFactor])
            timeIdwStop = time.time()        
            timeIdw = timeIdwStop - timeIdwStart
            print("Complete. Solve time: "+str(time))


#=================================================================
# Material Field
#=================================================================

# Create material field (sobelov parameters)
materialField = CMISS.Field()
equationsSet.MaterialsCreateStart(materialFieldUserNumber,materialField)
materialField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Smoothing Parameters")
equationsSet.MaterialsCreateFinish()

# Set kappa and tau - Sobelov smoothing parameters
materialField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,tau)
materialField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,kappa)


#=================================================================
# Equations
#=================================================================

# Create equations
equations = CMISS.Equations()
equationsSet.EquationsCreateStart(equations)
equations.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
equations.outputType = CMISS.EquationsOutputTypes.NONE
equationsSet.EquationsCreateFinish()

#=================================================================
# Problem setup
#=================================================================

# Create fitting problem
problem = CMISS.Problem()
problem.CreateStart(problemUserNumber)
problem.SpecificationSet(CMISS.ProblemClasses.FITTING,
        CMISS.ProblemTypes.DATA_FITTING,
        CMISS.ProblemSubTypes.DATA_POINT_VECTOR_STATIC_FITTING)
problem.CreateFinish()

# Create control loops
problem.ControlLoopCreateStart()
problem.ControlLoopCreateFinish()

# Create problem solver
solver = CMISS.Solver()
problem.SolversCreateStart()
problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,solver)
solver.outputType = CMISS.SolverOutputTypes.NONE
solver.linearType = CMISS.LinearSolverTypes.ITERATIVE
solver.linearIterativeAbsoluteTolerance = 1.0E-5
solver.linearIterativeRelativeTolerance = 1.0E-5
solver.linearIterativeDivergenceTolerance = 1.0e5
solver.linearIterativeMaximumIterations = 1000
problem.SolversCreateFinish()

# Create solver equations and add equations set to solver equations
solver = CMISS.Solver()
solverEquations = CMISS.SolverEquations()
problem.SolverEquationsCreateStart()
problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,solver)
solver.SolverEquationsGet(solverEquations)
solverEquations.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
equationsSetIndex = solverEquations.EquationsSetAdd(equationsSet)
problem.SolverEquationsCreateFinish()

#=================================================================
# Boundary Conditions
#=================================================================

# Create boundary conditions and set first and last nodes to 0.0 and 1.0
boundaryConditions = CMISS.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)

zeroTolerance = 0.00001
version = 1
boundaryNodeList = []

if (setBoundaries):
    # Wall boundary nodes
    value=0.0
    for node in range(numberOfWallNodes):
        nodeNumber = wallNodes[node]
        if (nodeNumber <= numberOfNodes): # hack for linear mesh
            nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent)
            if (nodeDomain == computationalNodeNumber):
                if (quadraticMesh):
                    pass
                elif (nodeNumber > numberOfNodes):
                    continue
                for component in range(numberOfDimensions):
                    componentId = component + 1
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                               version,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                               nodeNumber,componentId,
                                               CMISS.BoundaryConditionsTypes.FIXED,value)
solverEquations.BoundaryConditionsCreateFinish()

#=================================================================
# Solve 
#=================================================================

if (fitFem):
    if (optimise):
#        result = optimize.minimize(solveFemWithParameters,[tau,kappa],method="L-BFGS-B",
#                           bounds=[[-10000.0,-10000.0],[10000,10000]])

#        resultIdw = optimize.minimize(solveIdwWithParameters,[p,vicinityFactor],method="Powell",tol=1)
#        print(resultIdw)
#        tau = -0.00625
#        kappa = 1.0
#        ranges = ((-20,20),(-20,20))
#        ranges = ((0.00001,0.001,0.00005),(0.0005,0.05,0.0025))
#        ranges = ((0.00001,10.0),(0.00001,10.0))
#        ranges = ((0.00001,100.0),(0.00001,100.0))
#        resultFem = optimize.minimize(solveFemWithParameters,[tau,kappa],method="Powell",tol=0.0001)
#        resultFem = optimize.fmin_powell(solveFemWithParameters,[tau,kappa])
#        resultFem = optimize.fmin(solveFemWithParameters,[tau,kappa])

        logspace = 10.**scipy.linspace(-7.0, 1.0, 20)                    
        
        currentMin = 10000.0
        logFile = './output/' + meshName + '/opt/fef/CDR_' + str(diameterResolution) + '.log'
        currentErrorLine = 0
        for tau in logspace:
            for kappa in logspace:
                errorFem = solveFemWithParameters([tau,kappa])   

                #DEBUG: check if solution diverged
                try:
                    with open(logFile):
                        f = open(logFile,"rb")
                        lineCounter = 0
                        for line in f:
                            lineCounter += 1
                            if "Linear iterative solver did not converge" in line:
                                if lineCounter > currentErrorLine:
                                    print("*** WARNING: Detected PETSc solver divergence")
                                    print('previous error line: ' + str(currentErrorLine))
                                    print('current error line: ' + str(lineCounter))
                                    print("------------------------------------")
                                    currentErrorLine = lineCounter
                                    errorFem = 100.0
                        f.close()
                except IOError:
                   print ('Could not open run log file: ' + filename)

                if errorFem < currentMin:
                    print('### New brute min found: ' +str(errorFem) + '\n\n')
                    currentMin = errorFem
                    tauMin = tau
                    kappaMin = kappa

        print('\n Brute search complete. Min: ' +str(currentMin) + '\n')
        print('-------------------------------------')
        print('tau = '+str(tauMin))
        print('kappa = '+str(kappaMin))
        print('-------------------------------------')
        print('Starting Nelder-Mead fmin search \n')
        tau = tauMin
        kappa = kappaMin
        resultFem = optimize.fmin(solveFemWithParameters,[tau,kappa], xtol=0.0001)

        print(resultFem)
        print('tau brute= ' + str(tau))
        tau = resultFem[0]
        print('tau fmin= ' + str(tau))
        print('kappa brute= ' + str(kappa))
        kappa = resultFem[1]
        print('kappa fmin= ' + str(kappa))

        # #resultFem = optimize.brute(solveFemWithParameters,ranges)
        # print(resultFem)
        # print('tau original= ' + str(tau))
        # tau = resultFem[0]
        # print('tau new= ' + str(tau))
        # print('kappa original= ' + str(kappa))
        # kappa = resultFem[1]
        # print('kappa new= ' + str(kappa))

        fefVelRMSE = solveFemWithParameters([tau,kappa])
        print('error Fem= ' + str(fefVelRMSE))


    elif fefIterations > 1:
        #tau = 100.0
        #kappa = 10.0
        factor = 2.0
        timeFemStart = time.time()        
        # Solve the FEM problem w/ sobolev relaxation
        for iteration in range(fefIterations):
            timeFemItStart = time.time()                    
            tau = tau/factor
            kappa = kappa/factor
            fefVelRMSE = solveFemWithParameters([tau,kappa])            
            timeFemItStop = time.time()        
            timeIt = timeFemItStop - timeFemItStart
            print('\n\n   Iteration Time: ' + str(timeIt) + '\n\n')

        timeFemStop = time.time()        
        timeFem = timeFemStop - timeFemStart

    else:
        # Solve the FEM problem
        print("Solving...")
        timeFemStart = time.time()        
        #problem.Solve()
        fefVelRMSE = solveFemWithParameters([tau,kappa])
        print('error Fem= ' + str(fefVelRMSE))
        timeFemStop = time.time()        
        timeFem = timeFemStop - timeFemStart
        print("Complete. Solve time: "+str(timeFem))

#=================================================================
# Export results
#=================================================================

print("Fitting complete- exporting results")

print("exporting CMGUI data")
# Export results
fields = CMISS.Fields()
fields.CreateRegion(region)
print('mesh name: ' + meshName)
outputDirectory = 'output/' + meshName + '/'
# Create a results directory if needed
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   

if(addNoise):
    fields.NodesExport(outputDirectory + meshName + 'AxialSNR' + str(SNR) + "DataFit" + str(dataResolution[0]) + "_" + str(dataResolution[1]) + "_" + str(dataResolution[2]),"FORTRAN")
else:
    if(optimise):
    #if(True):
        optFolder = 'opt/'
        if fitIdw:
            if addZeroLayer:
                optFolder += 'idw/zeroLayer/'
                optErrorFile = './output/errorData/opt/idwZeroLayer.txt'
            elif setProjectionBoundariesIdw:
                optFolder += 'idw/projection/'
                optErrorFile = './output/errorData/opt/idwProjection.txt'
            else:
                optFolder += 'idw/none/'
                optErrorFile = './output/errorData/opt/idwNone.txt'
        else:
                optFolder += 'fef/'
                optErrorFile = './output/errorData/opt/fef.txt'

        outputDirectory += optFolder
        try:
            os.makedirs(outputDirectory)
        except OSError, e:
            if e.errno != 17:
                raise   
        outputFile = outputDirectory +  "CDR_" + str(diameterResolution)
        print('outputing to file ' +outputFile)
        fields.NodesExport(outputFile,"FORTRAN")
        print('Writing error to csv file: ' + optErrorFile)
        try:
            with open(optErrorFile, 'a') as file:
                if fitIdw:
                    file.write(meshTypes[meshName]+ ', ' + str(diameterResolution) + ', ' + str(p) + ', ' + str(vicinityFactor) + ', ' + str(idwVelRMSE) + ', \n')            
                else:
                    file.write(meshTypes[meshName]+ ', ' + str(diameterResolution) + ', ' + str(tau) + ', ' + str(kappa) + ', ' + str(fefVelRMSE) + ', \n')            
        except IOError:
            print ('Could not open RMSE log file: ' + optErrorFile)
    else:
        if fitIdw:
            outputDirectory = outputDirectory + 'versus/idw/'
            error = idwVelRMSE
        elif fitFem:
            outputDirectory = outputDirectory + 'versus/fef/'
            error = fefVelRMSE
        else:
            print('please specify fitting type as idw or fef!')
            sys.exit(0)
        # Create a results directory if needed
        try:
            os.makedirs(outputDirectory)
        except OSError, e:
            if e.errno != 17:
                raise   
        rmseFile = outputDirectory+'RMSE.txt'
        print('Writing error to RMSE error file: ' + rmseFile)
        try:
            with open(rmseFile, 'a') as file:
                file.write(str(diameterResolution) + ', ' + str(error) + ', \n')            
        except IOError:
            print ('Could not open RMSE log file: ' + rmseFile)
        outputFile = outputDirectory + 'CDR' + str(diameterResolution)
        #outputFile = outputDirectory + meshName + "DataFit_" + str(dataResolution[0]) + "_" + str(dataResolution[1]) + "_" + str(dataResolution[2])
        print('outputting to file ' +outputFile)
        fields.NodesExport(outputFile,"FORTRAN")
fields.Finalise()

#print("exporting analytic analysis")
exportAnalytic1=False
if (exportAnalytic1):
    analysisFile = "output/analysisFem" + "Data" + str(dataResolution[0]) + "_" + str(dataResolution[1]) + "_" + str(dataResolution[2]) + "_Tau" + str(tau) + "_Kappa" + str(kappa)
#    CMISS.AnalyticAnalysisOutput(dependentField,analysisFile)
    AbsErrorFem = numpy.zeros((numberOfNodes,numberOfDimensions))
    ErrorMagFem = numpy.zeros((numberOfNodes))
    for node in range(numberOfNodes):
        nodeId = node + 1
        for component in range(numberOfDimensions):
            componentId = component+1
            AbsErrorFem[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(dependentField,CMISS.FieldVariableTypes.U,1,
                                                                                   CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                                   nodeId,componentId)
        ErrorMagFem[node] = linalg.norm(AbsErrorFem[node,:])

    SDFem = numpy.std(ErrorMagFem)    
    RMSErrorFem = numpy.sqrt(numpy.mean(ErrorMagFem**2))
    print("Tau: " + str(tau))
    print("Kappa: " + str(kappa))
    print("Errors: ")
    print(AbsErrorFem)
    print("RMS Error: " + str(RMSErrorFem))
    print("------------------------------------\n\n")
#    RMSErrorFem = numpy.zeros((numberOfDimensions))
    # for component in range(numberOfDimensions):
    #     RMSErrorFem[component] = numpy.sqrt(numpy.mean(AbsErrorFem[:,component]**2))
    # RMSErrorMagnitudeFem = linalg.norm(RMSErrorFem)
    #print("RMS Error FEM: " + str(RMSErrorFem))
    #print("Standard Deviation FEM: " + str(SDFem))

    # if (optimise == False):
    #     print("Solve time FEM: " + str(timeFem) + "\n")
    #     with open('output/logs' + meshName + 'batchTimes.txt', 'a') as file:
    #         file.write(str(dataResolution) + ', ' + str(timeFem) + ', ' + str(timeIdw) + ', ' + str(dataProjectionTime) + '\n')

exportAnalytic=False
#if (fitIdw and not optimise):
if (exportAnalytic):
    analysisFile = "output/analysisIdw" + "Data" + str(dataResolution[0]) + "_" + str(dataResolution[1]) + "_" + str(dataResolution[2])
#    CMISS.AnalyticAnalysisOutput(dependentField,analysisFile)
    AbsErrorIdw = numpy.zeros((numberOfNodes,numberOfDimensions))
#    PercentErrorIdw = numpy.zeros((numberOfNodes,numberOfDimensions))
    AbsErrorNoise = numpy.zeros((numberOfNodes,numberOfDimensions))
#    PercentErrorNoise = numpy.zeros((numberOfNodes,numberOfDimensions))
    difference = numpy.zeros((numberOfNodes,numberOfDimensions))
    ErrorMagIdw = numpy.zeros((numberOfNodes))
    for node in range(numberOfNodes):
        nodeId = node + 1
        for component in range(numberOfDimensions):
            componentId = component+1
            AbsErrorIdw[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(independentField,CMISS.FieldVariableTypes.U1,1,
                                                                                   CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                                   nodeId,componentId)
            # PercentErrorIdw[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(independentField,CMISS.FieldVariableTypes.U1,1,
            #                                                                        CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
            #                                                                        nodeId,componentId)
            if (addNoise):
                AbsErrorNoise[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(independentField,CMISS.FieldVariableTypes.U1,1,
                                                                                       CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                                       nodeId,componentId)
                # PercentErrorNoise[node,component]=CMISS.AnalyticAnalysisPercentageErrorGetNode(independentField,CMISS.FieldVariableTypes.U1,1,
                #                                                                              CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                #                                                                              nodeId,componentId)
        ErrorMagIdw[node] = linalg.norm(AbsErrorIdw[node,:])

    meanIdw = numpy.mean(AbsErrorIdw)
    SDIdw = numpy.std(ErrorMagIdw)
    RMSErrorIdw = numpy.sqrt(numpy.mean(ErrorMagIdw**2))

    print("Error IDW: " + str(AbsErrorIdw))
    print("RMS Error IDW: " + str(RMSErrorIdw))
    print("Standard Deviation IDW: " + str(SDIdw))
    print("------------------------")
    print("SNR: " + str(SNR))    

    if(addNoise):
        with open('output/idwSmooth/' + meshName + 'SNR' + str(int(SNR)) + 'RMSErrors.txt', 'a') as file:
            file.write(str(dataResolution[0]) + ', ' + str(dataResolution[1]) + ', ' + str(dataResolution[2]) + ', ' + str(dataPointNoiseErrorRMS) + ', ' + str(RMSErrorIdw) + ', ' + str(dataPointNoiseErrorSD) + ', ' + str(SDIdw) +'\n')

    if(fitFem):
        with open('output/logs/' + meshName + 'RMSErrors.txt', 'a') as file:
            file.write(str(dataResolution[0]) + ', ' + str(dataResolution[1]) + ', ' + str(dataResolution[2]) + ', ' + str(RMSErrorFem) + ', ' + str(RMSErrorIdw) + ', ' + str(SDFem) + ', ' + str(SDIdw) +'\n')
    else:
        with open('output/logs/idw/' + meshName + 'RMSErrors.txt', 'a') as file:
            file.write(str(dataResolution[0]) + ', ' + str(dataResolution[1]) + ', ' + str(dataResolution[2]) + ', ' + str(meanIdw) + ', ' + str(RMSErrorIdw) + ', ' + str(SDIdw) + '\n')

CMISS.Finalise()
