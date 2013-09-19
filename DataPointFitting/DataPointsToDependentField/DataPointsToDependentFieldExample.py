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
import math

# Intialise OpenCMISS
from opencmiss import CMISS

# Set mesh grid dimensions
height = 1.0
width = 1.0
length = 1.0
meshDimensions=[height,width,length]
meshOrigin=[0.0,0.0,0.0]

numberOfDimensions = 3
# Set data point grid dimensions

meshResolution = 8
dataResolution = 4
setBoundaries = True
quadraticMesh = True
# Embedded only
dataPointInit = []
dataPointRegionSize = []
meshRegionSize = []
for i in range(numberOfDimensions):
    meshRegionSize.append(1.0)
    dataPointRegionSize.append(0.7)
    dataPointInit.append(0.15)

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

# Set sobelov smoothing parameters

tau = 0.01
kappa = 0.5

numberCoordinateElements = []
numberCoordinateDataPoints = []
numberOfElements = 1
numberDataPoints = 1

# Mesh and data point resolution
for dimension in range(numberOfDimensions):
    numberCoordinateElements.append(meshResolution)
    numberCoordinateDataPoints.append(dataResolution)
    numberOfElements = numberOfElements*numberCoordinateElements[dimension]
    numberDataPoints = numberDataPoints*numberCoordinateDataPoints[dimension]

# if (numberOfDimensions == 3) :
#     # Set mesh resolution
#     numberCoordinateElements[0] = 5
#     numberCoordinateElements[1] = 5
#     numberCoordinateElements[2] = 5
#     numberOfElements = numberCoordinateElements[0]*numberCoordinateElements[1]*numberCoordinateElements[2]
#     # Set data point resolution
#     numberCoordinateDataPoints[0] = 10
#     numberCoordinateDataPoints[1] = 10
#     numberCoordinateDataPoints[2] = 10
#     numberDataPoints = numberCoordinateDataPoints[0]*numberCoordinateDataPoints[1]*numberCoordinateDataPoints[2]
# elif (numberOfDimensions == 2) :
#     # Set mesh resolution
#     numberCoordinateElements[0] = 5
#     numberCoordinateElements[1] = 5
#     numberOfElements = numberCoordinateElements[0]*numberCoordinateElements[1]
#     # Set data point resolution
#     numberCoordinateDataPoints[0] = 10
#     numberCoordinateDataPoints[1] = 10
#     numberDataPoints = numberCoordinateDataPoints[0]*numberCoordinateDataPoints[1]

CMISS.DiagnosticsSetOn(CMISS.DiagnosticTypes.IN,[1,2,3,4,5],"Diagnostics",["DOMAIN_MAPPINGS_LOCAL_FROM_GLOBAL_CALCULATE"])

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()

# Creation a RC coordinate system
coordinateSystem = CMISS.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.dimension = numberOfDimensions
coordinateSystem.CreateFinish()

# Create a region
region = CMISS.Region()
region.CreateStart(regionUserNumber,CMISS.WorldRegion)
region.label = "FittingRegion"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

#=================================================================
# Data Points
#=================================================================

# Create a numpy array of data point locations
dataPointLocations = numpy.zeros((numberDataPoints,numberOfDimensions))
i = 0
#dataPointCoordinate = []
dataPointX = dataPointInit[0]
dataPointY = dataPointInit[1]
if (numberOfDimensions==3):
    dataPointZ = dataPointInit[2]

print("Number of data points:")
print(numberCoordinateDataPoints)

for x in range (numberCoordinateDataPoints[0]):
    for y in range (numberCoordinateDataPoints[1]):
        if (numberOfDimensions == 3):
            for z in range (numberCoordinateDataPoints[2]):
                dataPointLocations[i,:] = [dataPointX,dataPointY,dataPointZ]
                dataPointZ += (dataPointRegionSize[2]/(numberCoordinateDataPoints[2]-1))
                i+=1
            dataPointZ = dataPointInit[2]
        dataPointY += (dataPointRegionSize[1]/(numberCoordinateDataPoints[1]-1))
        if (numberOfDimensions == 2):
            dataPointLocations[i,:] = [dataPointX,dataPointY]
            i+=1
    dataPointY = dataPointInit[1]
    dataPointX += (dataPointRegionSize[0]/(numberCoordinateDataPoints[0]-1))

# Set up data points with geometric values
dataPoints = CMISS.DataPoints()
dataPoints.CreateStart(region,numberDataPoints)
for dataPoint in range(numberDataPoints):
    dataPointId = dataPoint + 1
    dataList = dataPointLocations[dataPoint,:]
    dataPoints.ValuesSet(dataPointId,dataList)
#    dataPoints.LabelSet(dataPointId,"Data Points")
dataPoints.CreateFinish()

#=================================================================
# Mesh
#=================================================================

# Create a lagrange basis
basis = CMISS.Basis()
basis.CreateStart(basisUserNumber)
basis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
basis.numberOfXi = numberOfDimensions
if (quadraticMesh):
    basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfDimensions
else:
    basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfDimensions

basis.quadratureNumberOfGaussXi = [numberOfDimensions]*numberOfDimensions
basis.CreateFinish()

# Create a generated mesh
generatedMesh = CMISS.GeneratedMesh()
generatedMesh.CreateStart(generatedMeshUserNumber,region)
generatedMesh.type = CMISS.GeneratedMeshTypes.REGULAR
generatedMesh.basis = [basis]
generatedMesh.extent = meshRegionSize

generatedMesh.numberOfElements = numberCoordinateElements
mesh = CMISS.Mesh()
generatedMesh.CreateFinish(meshUserNumber,mesh)

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CreateFinish()

#=================================================================
# Geometric Field
#=================================================================

# Create a field for the geometry
geometricField = CMISS.Field()
geometricField.CreateStart(geometricFieldUserNumber,region)
geometricField.meshDecomposition = decomposition
for dimension in range(numberOfDimensions):
    geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimension+1,1)
geometricField.CreateFinish()

# Set geometry from the generated mesh
generatedMesh.GeometricParametersCalculate(geometricField)

# Export mesh geometry
fields = CMISS.Fields()
fields.CreateRegion(region)
fields.NodesExport("Geometry","FORTRAN")
fields.ElementsExport("Geometry","FORTRAN")
fields.Finalise()

#=================================================================
# Data Projection on Geometric Field
#=================================================================

# Set up data projection
dataProjection = CMISS.DataProjection()
dataProjection.CreateStart(dataProjectionUserNumber,dataPoints,mesh)
dataProjection.projectionType = CMISS.DataProjectionProjectionTypes.ALL_ELEMENTS
dataProjection.CreateFinish()

# Evaluate data projection based on geometric field
dataProjection.ProjectionEvaluate(geometricField)
# Create mesh topology for data projection
mesh.TopologyDataPointsCalculateProjection(dataProjection)
# Create decomposition topology for data projection
decomposition.TopologyDataProjectionCalculate()

print("Data projection finished")
#=================================================================
# Equations Set
#=================================================================

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
# Nodes
#=================================================================
nodes = CMISS.Nodes()
region.NodesGet(nodes)
numberOfNodes = nodes.numberOfNodes

#=================================================================
# Dependent Field
#=================================================================
# Create dependent field (fitted values from data points)
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Dependent")
equationsSet.DependentCreateFinish()
# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

# Analytic values to test against - based on geometry field
#-----------------------------------------------------------
geometricValue = 0.0
dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.U,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)
dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.DELUDELN,
                                  CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)

dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                          1,geometricValue)
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.DELUDELN,
                                          CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                          1,geometricValue)
#set node-based analytic field to function of geometric field
for node in range(numberOfNodes):
    nodeId = node + 1
    for component in range(numberOfDimensions):
        componentId = component+1
        geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                            CMISS.FieldParameterSetTypes.VALUES,
                                                            1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
        value = geometricValue
#        print(value)
        dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                               nodeId,componentId,value)

# Export Fields w/o independent
fields = CMISS.Fields()
fields.CreateRegion(region)
fields.ElementsExport("Dependent","FORTRAN")
fields.Finalise()

#=================================================================
# Independent Field
#=================================================================

# Create data point field (independent field, with vector values stored at the data points)
independentField = CMISS.Field()
equationsSet.IndependentCreateStart(independentFieldUserNumber,independentField)
independentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"data point vector")
independentField.VariableLabelSet(CMISS.FieldVariableTypes.V,"data point weight")
independentField.DataProjectionSet(dataProjection)
equationsSet.IndependentCreateFinish()
# Initialise data point vector field to 0
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)
# Initialise data point weight field to 1
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,1,1.0)

minDistance=0.0001 

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
                value = dataPointLocations[dataPointNumberIndex,component]
                independentField.ParameterSetUpdateElementDataPointDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,elementId,dataPointId,componentId,value)

            # Set data point weights based on distance
            distance = dataProjection.ResultDistanceGet(dataPointNumber)
#            value = 1/(distance+minDistance)**2
            value = 1.0
            independentField.ParameterSetUpdateElementDataPointDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,elementId,dataPointId,1,value)


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
solver.outputType = CMISS.SolverOutputTypes.SOLVER
solver.linearType = CMISS.LinearSolverTypes.ITERATIVE
solver.linearIterativeAbsoluteTolerance = 1.0E-10
solver.linearIterativeRelativeTolerance = 1.0E-10
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
    # first find which nodes are boundary nodes
    for node in range(numberOfNodes):
        nodeId = node + 1
        for component in range(numberOfDimensions):
            componentId = component+1
            geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                CMISS.FieldParameterSetTypes.VALUES,
                                                                1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                nodeId,componentId)
            if(geometricValue < meshOrigin[component] +zeroTolerance or 
               geometricValue > meshDimensions[component] - zeroTolerance):

                boundaryNodeList.append(nodeId)
                break

    # now set boundary conditions on boundary nodes
    for nodeId in boundaryNodeList:
        for component in range(numberOfDimensions):
            componentId = component+1
            geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                CMISS.FieldParameterSetTypes.VALUES,
                                                                1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                nodeId,componentId)
            value = geometricValue

#            print("node # " + str(nodeId) + "component # " + str(componentId))

            boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                       version,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                       nodeId,componentId,
                                       CMISS.BoundaryConditionsTypes.FIXED,value)

solverEquations.BoundaryConditionsCreateFinish()

#=================================================================
# Solve 
#=================================================================

# Solve the problem
print("Solving...")
problem.Solve()

#=================================================================
# Export results
#=================================================================

print("Fitting complete- exporting results")

print("exporting analytic analysis")

if (quadraticMesh):
    analysisFile = "quadraticMesh" + str(meshResolution) + "Data" + str(dataResolution) + "Tau" + str(tau) + "Kappa" + str(kappa)
else:
    analysisFile = "linearMesh" + str(meshResolution) + "Data" + str(dataResolution) + "Tau" + str(tau) + "Kappa" + str(kappa)

CMISS.AnalyticAnalysisOutput(dependentField,analysisFile)
AbsError = numpy.zeros((numberOfNodes,numberOfDimensions))
ErrorMag = numpy.zeros((numberOfNodes))
for node in range(numberOfNodes):
    nodeId = node + 1
    for component in range(numberOfDimensions):
        componentId = component+1
        AbsError[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(dependentField,CMISS.FieldVariableTypes.U,1,
                                                                            CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                            nodeId,componentId)
    ErrorMag[node] = linalg.norm(AbsError[node,:])

RMSError = numpy.sqrt(numpy.mean(ErrorMag**2))

print("Error: ")
print(RMSError)



print("exporting CMGUI data")
# Export results
fields = CMISS.Fields()
fields.CreateRegion(region)
fields.NodesExport("DataFit","FORTRAN")
fields.Finalise()

#-----------------------------------------------------------------
# Dummy data points mesh - for display only!
#-----------------------------------------------------------------

# Creation a RC coordinate system
dummyCoordinateSystem = CMISS.CoordinateSystem()
dummyCoordinateSystem.CreateStart(dummyCoordinateSystemUserNumber)
dummyCoordinateSystem.dimension = numberOfDimensions
dummyCoordinateSystem.CreateFinish()

# Create a region
dummyRegion = CMISS.Region()
dummyRegion.CreateStart(dummyRegionUserNumber,CMISS.WorldRegion)
dummyRegion.label = "DataRegion"
dummyRegion.coordinateSystem = dummyCoordinateSystem
dummyRegion.CreateFinish()

dummyGeneratedMesh = CMISS.GeneratedMesh()
dummyGeneratedMesh.CreateStart(dummyGeneratedMeshUserNumber,dummyRegion)
dummyGeneratedMesh.type = CMISS.GeneratedMeshTypes.REGULAR

dummyBasis = CMISS.Basis()
dummyBasis.CreateStart(dummyBasisUserNumber)
dummyBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
dummyBasis.numberOfXi = numberOfDimensions
dummyBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*numberOfDimensions
dummyBasis.quadratureNumberOfGaussXi = [numberOfDimensions]*numberOfDimensions
dummyBasis.CreateFinish()

dummyGeneratedMesh.basis = [dummyBasis]
dummyGeneratedMesh.extent = dataPointRegionSize
dummyGeneratedMesh.origin = dataPointInit

print("Number of data points:")
print(numberCoordinateDataPoints)
dummyGeneratedMesh.numberOfElements = [i-1 for i in numberCoordinateDataPoints]
dummyMesh = CMISS.Mesh()
dummyGeneratedMesh.CreateFinish(dummyMeshUserNumber,dummyMesh)

# Create a decomposition for the mesh
dummyDecomposition = CMISS.Decomposition()
dummyDecomposition.CreateStart(dummyDecompositionUserNumber,dummyMesh)
dummyDecomposition.type = CMISS.DecompositionTypes.CALCULATED
dummyDecomposition.numberOfDomains = numberOfComputationalNodes
dummyDecomposition.CalculateFacesSet(True)
dummyDecomposition.CreateFinish()

# Create a field for the geometry
dummyGeometricField = CMISS.Field()
dummyGeometricField.CreateStart(dummyGeometricFieldUserNumber,dummyRegion)
dummyGeometricField.meshDecomposition = dummyDecomposition
for dimension in range(numberOfDimensions):
    dummyGeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimension+1,1)
dummyGeometricField.CreateFinish()

# Set geometry from the generated mesh
dummyGeneratedMesh.GeometricParametersCalculate(dummyGeometricField)

# Export dummy display mesh
baseName = "dataPoints"
dataFormat = "PLAIN_TEXT"
fml = CMISS.FieldMLIO()
fml.OutputCreate(dummyMesh, "", baseName, dataFormat)
# Write geometric field
fml.OutputCreate(dummyMesh, "", baseName, dataFormat)
fml.OutputAddFieldNoType(baseName+".geometric", dataFormat, dummyGeometricField,
    CMISS.FieldVariableTypes.U, CMISS.FieldParameterSetTypes.VALUES)
fml.OutputWrite("DataPoints.xml")
fml.Finalise()

# else:
#     # Export results
#     fields = CMISS.Fields()
#     fields.CreateRegion(dummyRegion)
#     fields.NodesExport("DataPoints","FORTRAN")
#     fields.Finalise()

#Destroy dummy objects
dummyGeometricField.Destroy()
dummyDecomposition.Destroy()
dummyGeneratedMesh.Destroy()
dummyMesh.Destroy()

#-----------------------------------------------------------------

CMISS.Finalise()
