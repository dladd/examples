#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an example to fit a generated cube mesh to a sphere.
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

#> \example /Fitting/GeometricFitting/GeometricFittingExample.py
## Example script to fit a generated cube mesh to a sphere using openCMISS calls in python.
## \par Latest Builds:
#<

# Add Python bindings directory to PATH
import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import exfile
import numpy
from numpy import linalg
import math
import random

# Intialise OpenCMISS
from opencmiss import CMISS

def writeExdataFile(filename,dataPointLocations):
    "Writes 3D data points to an exdata file"

    try:
        f = open(filename,"w")    
        header = '''Group name: DataPoints
 #Fields=1
 1) data_coordinates, coordinate, rectangular cartesian, #Components=3
  1.  Value index=1, #Derivatives=0, #Versions=1
  2.  Value index=2, #Derivatives=0, #Versions=1
  3.  Value index=3, #Derivatives=0, #Versions=1
'''
        f.write(header)

        numberOfDataPoints = len(dataPointLocations)
        for i in range(numberOfDataPoints):
            line = " Node: " + str(i+1) + '\n'
            f.write(line)
            for j in range (3):
                line = ' ' + str(dataPointLocations[i,j]) + '\n'
                f.write(line)
        f.close()
            
    except IOError:
        print ('Could not open file: ' + filename)
        

#=================================================================
# Control Panel
#=================================================================

# Set generated mesh grid dimensions
numberOfDimensions = 3
height = 1.0
width = 1.0
length = 1.0
meshDimensions=[height,width,length]
meshOrigin=[-height/2.0,-width/2.0,-length/2.0]
meshResolution = [2,2,2]

# Set data point resolution
numberOfDataPoints = 1000
radius = 1.0
origin = [0.,0.,0.]

# fix interior nodes so that fitting only applies to surface
fixInterior = True

# analyse fitting error against the analytic solution
analyticAnalysis = False

iteration = 1
if iteration > 1:
    exfileMesh = True
    exnode = exfile.Exnode("DeformedGeometry" + str(iteration-1) + ".part0.exnode")
    exelem = exfile.Exelem("DeformedGeometry" + str(iteration-1) + ".part0.exelem")
else:
    exfileMesh = False

# Set sobelov smoothing parameters
tau = 0.0
kappa = 0.0

numberOfGaussXi = 3
zeroTolerance = 0.00001

#=================================================================

(coordinateSystemUserNumber,
    regionUserNumber,
    basisUserNumber,
    generatedMeshUserNumber,
    meshUserNumber,
    decompositionUserNumber,
    geometricFieldUserNumber,
    equationsSetFieldUserNumber,
    dependentFieldUserNumber,
    independentFieldUserNumber,
    dataPointFieldUserNumber,
    materialFieldUserNumber,
    analyticFieldUserNumber,
    dependentDataFieldUserNumber,
    dataProjectionUserNumber,
    equationsSetUserNumber,
    problemUserNumber) = range(1,18)

# Diagnostics
CMISS.DiagnosticsSetOn(CMISS.DiagnosticTypes.IN,[1,2,3,4,5],"Diagnostics",["DOMAIN_MAPPINGS_LOCAL_FROM_GLOBAL_CALCULATE"])

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()

# Create a RC coordinate system
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
dataPointLocations = numpy.zeros((numberOfDataPoints,numberOfDimensions))

print("Number of data points: " + str(numberOfDataPoints))

# Calculate locations of points on a sphere
for i in range(numberOfDataPoints):
    theta = 2.0*math.pi*random.uniform(0.0,radius)
    phi = math.acos(2.0*random.uniform(0.0,radius)-radius)
    x = math.cos(theta)*math.sin(phi)
    y = math.sin(theta)*math.sin(phi)
    z = math.cos(phi)
    dataPointLocations[i,:] = [x,y,z]

# Set up CMISS data points with geometric values
dataPoints = CMISS.DataPoints()
dataPoints.CreateStart(region,numberOfDataPoints)
for dataPoint in range(numberOfDataPoints):
    dataPointId = dataPoint + 1
    dataList = dataPointLocations[dataPoint,:]
    dataPoints.ValuesSet(dataPointId,dataList)
dataPoints.CreateFinish()

print("Writing data points file")
writeExdataFile("DataPoints.exdata",dataPointLocations)

#=================================================================
# Mesh
#=================================================================
# Calc number of elements
numberOfElements = 1
for dimension in range(numberOfDimensions):
    numberOfElements = numberOfElements*meshResolution[dimension]

# Create a lagrange basis
basis = CMISS.Basis()
basis.CreateStart(basisUserNumber)
basis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
basis.numberOfXi = numberOfDimensions
#basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfDimensions
basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*numberOfDimensions

basis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfDimensions
basis.CreateFinish()

if (exfileMesh):
    mesh = CMISS.Mesh()
    mesh.CreateStart(meshUserNumber, region, numberOfDimensions)
    mesh.NumberOfComponentsSet(1)
    mesh.NumberOfElementsSet(exelem.num_elements)
    # Define nodes for the mesh
    nodes = CMISS.Nodes()
    nodes.CreateStart(region, exnode.num_nodes)
    nodes.CreateFinish()
    # Define elements for the mesh
    elements = CMISS.MeshElements()
    meshComponentNumber = 1
    elements.CreateStart(mesh, meshComponentNumber, basis)
    for elem in exelem.elements:
        elements.NodesSet(elem.number, elem.nodes)
    elements.CreateFinish()
    mesh.CreateFinish()
else:
    # Create a generated mesh
    generatedMesh = CMISS.GeneratedMesh()
    generatedMesh.CreateStart(generatedMeshUserNumber,region)
    generatedMesh.type = CMISS.GeneratedMeshTypes.REGULAR
    generatedMesh.basis = [basis]
    generatedMesh.extent = meshDimensions
    generatedMesh.origin = meshOrigin

    generatedMesh.numberOfElements = meshResolution
    mesh = CMISS.Mesh()
    generatedMesh.CreateFinish(meshUserNumber,mesh)

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CalculateFacesSet(True)
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

# Get or calculate geometric parameters
if (exfileMesh):
    # Read the geometric field from the exnode file
    geometricField.ParameterSetUpdateStart(
            CMISS.FieldVariableTypes.U, CMISS.FieldParameterSetTypes.VALUES)
    for node_num in range(1, exnode.num_nodes + 1):
        version = 1
        derivative = 1
        for component in range(1, numberOfDimensions + 1):
            component_name = ["x", "y", "z"][component - 1]
            value = exnode.node_value("Coordinate", component_name, node_num, derivative)
            geometricField.ParameterSetUpdateNode(
                    CMISS.FieldVariableTypes.U,
                    CMISS.FieldParameterSetTypes.VALUES,
                    version, derivative, node_num, component, value)
    geometricField.ParameterSetUpdateFinish(
            CMISS.FieldVariableTypes.U, CMISS.FieldParameterSetTypes.VALUES)
else:
    # Create undeformed geometry from the generated mesh
    generatedMesh.GeometricParametersCalculate(geometricField)
    # Export undeformed mesh geometry
    print("Writing undeformed geometry")
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    fields.NodesExport("UndeformedGeometry","FORTRAN")
    fields.ElementsExport("UndeformedGeometry","FORTRAN")
    fields.Finalise()

#=================================================================
# Data Projection on Geometric Field
#=================================================================

print("Projecting data points onto geometric field")
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
print("Projection complete")

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

# Create dependent field (will be deformed fitted values based on data point locations)
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Dependent")
equationsSet.DependentCreateFinish()
# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

# #=================================================================
# # Analytic Field
# #=================================================================

# # Create analytic field (mainly used for output- to comparewill be deformed fitted values based on data point locations)
# analyticField = CMISS.Field()
# equationsSet.AnalyticCreateStart(CMISS.EquationsSetLaplaceAnalyticFunctionTypes.THREE_DIM_1,analyticFieldUserNumber,analyticField)
# analyticField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Analytic")
# equationsSet.AnalyticCreateFinish()
# # Initialise analytic field
# analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

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

# loop over each element's data points and set independent field values to data point locations on surface of the sphere
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

version = 1
surfaceNodeList = []
interiorList = []
meshComponent = decomposition.MeshComponentGet()
# Fix the interior nodes- use to only apply fit to surface nodes
if (fixInterior):
    # first find which nodes are non-surface nodes
    for node in range(numberOfNodes):
        nodeId = node + 1
        nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            for component in range(numberOfDimensions):
                componentId = component+1
                geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                    CMISS.FieldParameterSetTypes.VALUES,
                                                                    1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                    nodeId,componentId)
                if (geometricValue < meshOrigin[component] + zeroTolerance or
                    geometricValue > meshOrigin[component] + meshDimensions[component] - zeroTolerance):
                    surfaceNodeList.append(nodeId)
                    break

    # set fixed conditions on interior nodes
#    print(boundaryNodeList)
    for nodeId in range(1,numberOfNodes+1):
        nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            if nodeId not in surfaceNodeList:
                interiorList.append(nodeId)
                for component in range(numberOfDimensions):
                    componentId = component+1
                    geometricValue=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                        CMISS.FieldParameterSetTypes.VALUES,
                                                                        1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                        nodeId,componentId)
                    value = geometricValue
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                               version,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                               nodeId,componentId,
                                               CMISS.BoundaryConditionsTypes.FIXED,value)

#print(interiorList)
solverEquations.BoundaryConditionsCreateFinish()

#=================================================================
# Solve 
#=================================================================

# Solve the problem
print("Solving fitting problem...")
problem.Solve()

#=================================================================
# Calculate error
#=================================================================

if (analyticAnalysis):
    # Error will be distance from sphere surface
    #-----------------------------------------------------------
    value = 0.0
    dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.U,
                                      CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)
    dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.DELUDELN,
                                      CMISS.FieldParameterSetTypes.ANALYTIC_VALUES)
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                              CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                              1,value)
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.DELUDELN,
                                              CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                              1,value)

    undeformedValue = numpy.zeros((numberOfDimensions))
    deformedValue = numpy.zeros((numberOfDimensions))
    distFromOrigin = numpy.zeros((numberOfDimensions))
    #set projected analytic values
    for node in range(numberOfNodes):
        nodeId = node + 1
        nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            for component in range(numberOfDimensions):
                componentId = component+1
                undeformedValue[component]=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                               CMISS.FieldParameterSetTypes.VALUES,
                                                                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
                deformedValue[component]=dependentField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                              CMISS.FieldParameterSetTypes.VALUES,
                                                                              1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
                distFromOrigin[component] = origin[component] - deformedValue[component] 

            # calc nearest point on the sphere
            for component in range(numberOfDimensions):
                value = radius*(distFromOrigin[component])/numpy.linalg.norm(distFromOrigin)
                if (fixInterior and nodeId not in surfaceNodeList):
                    value = undeformedValue[component]

                # Set analytic values 
                dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                        CMISS.FieldParameterSetTypes.ANALYTIC_VALUES,
                                                        1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                        nodeId,componentId,value)            
                # analyticField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                #                                        CMISS.FieldParameterSetTypes.VALUES,
                #                                        1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                #                                        nodeId,componentId,value)            


#=================================================================
# Copy dependent field to geometric
#=================================================================
for component in range(1,numberOfDimensions+1):
    dependentField.ParametersToFieldParametersComponentCopy(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,component,geometricField,CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,component)

print("Writing deformed geometry")
# Export mesh geometry
fields = CMISS.Fields()
fields.CreateRegion(region)
fields.NodesExport("DeformedGeometry" + str(iteration),"FORTRAN")
fields.ElementsExport("DeformedGeometry" + str(iteration),"FORTRAN")
fields.Finalise()

#=================================================================
# Export results
#=================================================================

print("Fitting complete- exporting results")

if (analyticAnalysis):
    print("Exporting analytic analysis")
    analysisFile = "mesh" + str(meshResolution[0]) + "x" + str(meshResolution[1]) + "x" + str(meshResolution[2]) + "Data" + str(numberOfDataPoints) + "Tau" + str(tau) + "Kappa" + str(kappa)

    CMISS.AnalyticAnalysisOutput(dependentField,analysisFile)
    AbsError = numpy.zeros((numberOfNodes,numberOfDimensions))
    ErrorMag = numpy.zeros((numberOfNodes))
    for node in range(numberOfNodes):
        nodeId = node + 1
        nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            for component in range(numberOfDimensions):
                componentId = component+1
                AbsError[node,component]=CMISS.AnalyticAnalysisAbsoluteErrorGetNode(dependentField,CMISS.FieldVariableTypes.U,1,
                                                                                    CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                                    nodeId,componentId)
            ErrorMag[node] = linalg.norm(AbsError[node,:])

    RMSError = numpy.sqrt(numpy.mean(ErrorMag**2))

    print("Error: ")
    print(RMSError)

#-----------------------------------------------------------------

CMISS.Finalise()
