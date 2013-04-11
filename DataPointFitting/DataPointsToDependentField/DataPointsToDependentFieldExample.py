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
import math

# Intialise OpenCMISS
from opencmiss import CMISS

# Set mesh grid dimensions
height = 1.0
width = 1.0
length = 1.0

# Set data point grid dimensions
heightData = 1.5
widthData = 1.5
lengthData = 1.5
dataPointXInit = -0.25
dataPointYInit = -0.25
dataPointZInit = -0.25

(coordinateSystemUserNumber,
    regionUserNumber,
    basisUserNumber,
    generatedMeshUserNumber,
    meshUserNumber,
    decompositionUserNumber,
    geometricFieldUserNumber,
    dummyCoordinateSystemUserNumber,
    dummyRegionUserNumber,
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
    dependentDataFieldUserNumber,
    dataProjectionUserNumber,
    equationsSetUserNumber,
    problemUserNumber) = range(1,24)

numberOfDimensions = 3

# Set sobelov smoothing parameters
tau = 0.0001
kappa = 0.0005

# Set mesh resolution
numberGlobalXElements = 5
numberGlobalYElements = 5
numberGlobalZElements = 5

# Set data point resolution
numberGlobalXDataPoints = 3
numberGlobalYDataPoints = 3
numberGlobalZDataPoints = 3

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
# Data Point/Projection 
#=================================================================

numberDataPoints = numberGlobalXDataPoints*numberGlobalYDataPoints*numberGlobalZDataPoints

# Create a numpy array of data point locations
dataPointLocations = numpy.zeros((numberDataPoints,3))
i = 0
dataPointX = dataPointXInit
dataPointY = dataPointYInit
dataPointZ = dataPointZInit
for x in range (numberGlobalXDataPoints):
    for y in range (numberGlobalYDataPoints):
        for z in range (numberGlobalZDataPoints):
            dataPointLocations[i,:] = [dataPointX,dataPointY,dataPointZ]
            dataPointZ += (lengthData/(numberGlobalZDataPoints-1))
            i+=1
        dataPointZ = dataPointZInit
        dataPointY += (heightData/(numberGlobalYDataPoints-1))
    dataPointY = dataPointYInit
    dataPointX += (widthData/(numberGlobalXDataPoints-1))

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
# Setup Mesh
#=================================================================

# Create a bi-linear lagrange basis
basis = CMISS.Basis()
basis.CreateStart(basisUserNumber)
basis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
basis.numberOfXi = 3
basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*3
basis.quadratureNumberOfGaussXi = [3]*3
basis.CreateFinish()

# Create a generated mesh
generatedMesh = CMISS.GeneratedMesh()
generatedMesh.CreateStart(generatedMeshUserNumber,region)
generatedMesh.type = CMISS.GeneratedMeshTypes.REGULAR
generatedMesh.basis = [basis]
generatedMesh.extent = [width,height,length]
generatedMesh.numberOfElements = [numberGlobalXElements,numberGlobalYElements,numberGlobalZElements]
mesh = CMISS.Mesh()
generatedMesh.CreateFinish(meshUserNumber,mesh)

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
#decomposition.CalculateFacesSet(True)
decomposition.CreateFinish()

#=================================================================
# Setup Geometric Field
#=================================================================

# Create a field for the geometry
geometricField = CMISS.Field()
geometricField.CreateStart(geometricFieldUserNumber,region)
geometricField.meshDecomposition = decomposition
geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,1)
geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,3,1)
geometricField.CreateFinish()

# Set geometry from the generated mesh
generatedMesh.GeometricParametersCalculate(geometricField)

#=================================================================
# Setup Data Projection on Geometric Field
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

#=================================================================
# Equations Set
#=================================================================

# Create vector fitting equations set
equationsSetField = CMISS.Field()
equationsSet = CMISS.EquationsSet()
equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
        CMISS.EquationsSetClasses.FITTING,
        CMISS.EquationsSetTypes.DATA_FITTING_EQUATION,
        CMISS.EquationsSetSubtypes.DATA_POINT_VECTOR_DATA_FITTING,
        equationsSetFieldUserNumber, equationsSetField)
equationsSet.CreateFinish()

#=================================================================
# Dependent Field
#=================================================================

# Create dependent field (fitted values from data points)
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Dependent")
#dependentField.DOFOrderTypeSet(CMISS.FieldVariableTypes.U,CMISS.FieldDOFOrderTypes.SEPARATED)
equationsSet.DependentCreateFinish()
# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

#=================================================================
# Independent Field
#=================================================================

# Create data point field (independent field, with vector values stored at the data points)
independentField = CMISS.Field()
equationsSet.IndependentCreateStart(independentFieldUserNumber,independentField)
independentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Independent")
independentField.DataProjectionSet(dataProjection)
equationsSet.IndependentCreateFinish()
# Initialise dataPoint field
independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

# set data point field values
for dataPoint in range(numberDataPoints):
    dataPointId = dataPoint + 1
    projectedElementNumber = dataProjection.ElementGet(dataPointId)
    elementDomain = decomposition.ElementDomainGet(projectedElementNumber)
    if (elementDomain == computationalNodeNumber):
        for component in range(numberOfDimensions):
            componentId = component + 1
            value = dataPointLocations[dataPoint,component]
            independentField.ParameterSetUpdateDataPointDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,dataPointId,componentId,value)

#=================================================================
# Material Field
#=================================================================

# Create material field (fitted values from data points)
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
        CMISS.ProblemSubTypes.DATA_POINT_VECTOR_DATA_FITTING)
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
solverEquations.BoundaryConditionsCreateFinish()

#=================================================================
# Solve 
#=================================================================

# Solve the problem
problem.Solve()


#=================================================================
# Export results
#=================================================================
exportFieldml = True

if (exportFieldml):
    # Export geometric field (fieldML)
    baseName = "dataFit"
    dataFormat = "PLAIN_TEXT"
    fml = CMISS.FieldMLIO()
    fml.OutputCreate(mesh, "", baseName, dataFormat)
    fml.OutputAddFieldNoType(baseName+".geometric", dataFormat, geometricField,
        CMISS.FieldVariableTypes.U, CMISS.FieldParameterSetTypes.VALUES)
    # ouput dependent field results
    fml.OutputAddFieldNoType(baseName+".dependent", dataFormat, dependentField,
        CMISS.FieldVariableTypes.U, CMISS.FieldParameterSetTypes.VALUES)
    fml.OutputWrite("DataFitExample.xml")
    fml.Finalise()
else:
    # Export results
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("DataFit","FORTRAN")
    fields.ElementsExport("DataFit","FORTRAN")
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
dummyGeneratedMesh.basis = [basis]
dummyGeneratedMesh.extent = [widthData,heightData,lengthData]
dummyGeneratedMesh.origin = [dataPointXInit,dataPointYInit,dataPointZInit]
dummyGeneratedMesh.numberOfElements = [numberGlobalXDataPoints,numberGlobalYDataPoints,numberGlobalZDataPoints]
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
dummyGeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
dummyGeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,1)
dummyGeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,3,1)
dummyGeometricField.CreateFinish()

# Set geometry from the generated mesh
dummyGeneratedMesh.GeometricParametersCalculate(dummyGeometricField)

if (exportFieldml):
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
else:
    # Export results
    fields = CMISS.Fields()
    fields.CreateRegion(dummyRegion)
    fields.NodesExport("DataPoints","FORTRAN")
    fields.Finalise()

#Destroy dummy objects
dummyGeometricField.Destroy()
dummyDecomposition.Destroy()
dummyGeneratedMesh.Destroy()
dummyMesh.Destroy()

#-----------------------------------------------------------------

CMISS.Finalise()
