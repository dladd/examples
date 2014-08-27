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

def writeIpdataFile(filename,dataPointLocations):
    "Writes 3D data points to an ipdata file"

    try:
        f = open(filename,"w")    
        header = 'cube\n'
        f.write(header)

        numberOfDataPoints = len(dataPointLocations)
        for i in range(numberOfDataPoints):
            line = str(i+1)
            for j in range (3):
                line += ' ' + str(dataPointLocations[i,j])
            for j in range (3):
                line += ' 1.0'
            line += '\n'
            f.write(line)
        f.close()
            
    except IOError:
        print ('Could not open file: ' + filename)
        

#=================================================================
# Control Panel
#=================================================================

# Set generated mesh grid dimensions
numberOfDimensions = 3
height = 1.25
width = 1.25
length = 1.25
meshDimensions=[height,width,length]
meshOrigin=[-height/2.0,-width/2.0,-length/2.0]
meshResolution = [2,2,2]

# Set data point resolution
numberOfDataPoints = 1000
radius = 1.0
origin = [0.,0.,0.]

# fix interior nodes so that fitting only applies to surface
fixInterior = True

numberOfIterations = 5

iteration = 1
if iteration > 1:
    exfileMesh = True
    exnode = exfile.Exnode("DeformedGeometry" + str(iteration-1) + ".part0.exnode")
#    exelem = exfile.Exelem("DeformedGeometry" + str(iteration-1) + ".part0.exelem")
    exelem = exfile.Exelem("UndeformedGeometry.part0.exelem")
else:
    exfileMesh = False

# Set sobelov smoothing parameters
tau = 0.01
kappa = 0.01

numberOfGaussXi = 4
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
#writeIpdataFile("cube.ipdata",dataPointLocations)

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
basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.CUBIC_HERMITE]*numberOfDimensions

basis.quadratureNumberOfGaussXi = [numberOfGaussXi]*numberOfDimensions
basis.CreateFinish()

if (exfileMesh):
    # Read previous mesh
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
geometricField.ScalingTypeSet(CMISS.FieldScalingTypes.UNIT)
geometricField.CreateFinish()

# Get nodes
nodes = CMISS.Nodes()
region.NodesGet(nodes)
numberOfNodes = nodes.numberOfNodes

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

writeIpnode = True
if (writeIpnode):
    meshComponent = decomposition.MeshComponentGet()
    try:
        f = open("cube.ipnode","w")    
        header = ''' CMISS Version 1.21 ipnode File Version 2
 Heading: 27 nodes in a cube
 
 The number of nodes is [    1]:     27
 Number of coordinates [3]: 3
 Do you want prompting for different versions of nj=1 [N]? N
 Do you want prompting for different versions of nj=2 [N]? N
 Do you want prompting for different versions of nj=3 [N]? N
 The number of derivatives for coordinate 1 is [0]: 0
 The number of derivatives for coordinate 2 is [0]: 0
 The number of derivatives for coordinate 3 is [0]: 0'''
        f.write(header)

        undeformedValue = numpy.zeros((numberOfDimensions))
        for node in range(numberOfNodes):
            nodeId = node + 1
            nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
            if (nodeDomain == computationalNodeNumber):
                nodeHeader = "\n\n Node number [    " + str(nodeId) + "]:     " + str(nodeId)
                f.write(nodeHeader)
                for component in range(numberOfDimensions):
                    componentId = component+1
                    undeformedValue[component]=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                                    CMISS.FieldParameterSetTypes.VALUES,
                                                                                    1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeId,componentId)
                    nodeComponent = "\n The Xj(" + str(componentId) + ") coordinate is [ 0.00000E+00]: " + str(undeformedValue[component])
                    f.write(nodeComponent)
        f.close()
    except IOError:
        print ('Could not open ipnode file')
            

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
# Dependent Field
#=================================================================

# Create dependent field (will be deformed fitted values based on data point locations)
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
dependentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"Dependent")
dependentField.ScalingTypeSet(CMISS.FieldScalingTypes.UNIT)
equationsSet.DependentCreateFinish()
# Initialise dependent field
dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

# Initialise dependent field to undeformed geometric field
for component in range (1,numberOfDimensions+1):
    geometricField.ParametersToFieldParametersComponentCopy(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                            component, dependentField, CMISS.FieldVariableTypes.U,
                                                            CMISS.FieldParameterSetTypes.VALUES, component)

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
solver.outputType = CMISS.SolverOutputTypes.NONE # NONE
solver.linearType = CMISS.LinearSolverTypes.ITERATIVE
solver.LibraryTypeSet(CMISS.SolverLibraries.UMFPACK) # UMFPACK/SUPERLU
solver.linearIterativeAbsoluteTolerance = 1.0E-10
solver.linearIterativeRelativeTolerance = 1.0E-05
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
meshComponent = decomposition.MeshComponentGet()
# Fix the interior nodes- use to only apply fit to surface nodes
if (fixInterior):
    # first find which nodes are non-surface nodes
    for node in range(numberOfNodes):
        nodeId = node + 1
        nodeDomain = decomposition.NodeDomainGet(nodeId,meshComponent)
        if (nodeDomain == computationalNodeNumber):
            geometricValue = numpy.zeros((numberOfDimensions))
            for component in range(numberOfDimensions):
                componentId = component+1
                geometricValue[component]=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                               CMISS.FieldParameterSetTypes.VALUES,
                                                                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                               nodeId,componentId)

            derivList = []
            deriv = 0
            if (abs(geometricValue[0]) < (abs(meshOrigin[0]) - zeroTolerance)) and (abs(geometricValue[1]) < (abs(meshOrigin[1]) - zeroTolerance)) and (abs(geometricValue[2]) < (abs(meshOrigin[2]) - zeroTolerance)):
                # Interior node
                derivList = [1,2,3,4,5,6,7,8]
            # Radial nodes
            elif abs(geometricValue[0]) < zeroTolerance and abs(geometricValue[1]) < zeroTolerance:
                deriv = 5
            elif abs(geometricValue[1]) < zeroTolerance and abs(geometricValue[2]) < zeroTolerance:
                deriv = 2
            elif abs(geometricValue[0]) < zeroTolerance and abs(geometricValue[2]) < zeroTolerance:
                deriv = 3

            if deriv > 0 and deriv not in derivList:
                derivList.append(deriv)


            print("BC node " + str(nodeId))
            for globalDeriv in derivList: 
                print("    deriv " + str(globalDeriv))
                for component in range(1,numberOfDimensions+1):
                    print("        component " + str(component))
                    value=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                               CMISS.FieldParameterSetTypes.VALUES,
                                                               version,globalDeriv,nodeId,component)
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                               version,globalDeriv,nodeId,component,
                                               CMISS.BoundaryConditionsTypes.FIXED,value)

solverEquations.BoundaryConditionsCreateFinish()


for iteration in range (1,numberOfIterations+1):

    #=================================================================
    # Solve 
    #=================================================================

    # Solve the problem
    print("Solving fitting problem, iteration: " + str(iteration))

    problem.Solve()

    #=================================================================
    # Copy dependent field to geometric & Export fields
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

#-----------------------------------------------------------------

CMISS.Finalise()
