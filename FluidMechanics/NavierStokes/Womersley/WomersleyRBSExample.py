#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an OpenCMISS script to solve Navier-Stokes oscillatory flow through a cylinder and validate the solution against Womersley's analytic solution for the velocity profile. 
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

#> \example FluidMechanics/NavierStokes/Womersley/WomersleyExample.py
## Python OpenCMISS script to solve Navier-Stokes oscillatory flow through a cylinder and validate the solution against Womersley's analytic solution for the velocity profile.
## \par Latest Builds:
#<


# Add Python bindings directory to PATH
import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy
import gzip
import time
import re
import math
import contextlib
import womersleyAnalytic

# Intialise OpenCMISS
from opencmiss import CMISS

@contextlib.contextmanager
def ChangeDirectory(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(prev_cwd)

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()
if numberOfComputationalNodes == 1:
    logfile = 'serial'
else:
    logfile = 'mpi'
CMISS.OutputSetOn(logfile)

#==========================================================
# P r o b l e m     C o n t r o l
#==========================================================

# Problem parameters
offset = 0.0
density = 1.0
amplitude = 1.0
period = math.pi/2.
timeIncrement = period/1000.0 #[period/400.] 10,25,50,200
theta = 1.0
womersleyNumber = 10.0
startTime = 0.0
stopTime = timeIncrement + 0.00001 #period + 0.000001
outputFrequency = 1
initialiseAnalytic = True
beta = 0.0

# Mesh parameters
quadraticMesh = True
equalOrder = True
meshName = 'hexCylinder13' # 140, 12
inputDir = './input/' + meshName +'/'
length = 10.0
radius = 0.5
axialComponent = 1

#==========================================================
print('dynamic theta: ' + str(theta))
viscosity = density/(womersleyNumber**2.0)
if quadraticMesh:
    meshType = 'Quadratic'
else:
    meshType = 'Linear'
outputDirectory = ("./output/Wom" + str(womersleyNumber) + 'Dt' + str(round(timeIncrement,5)) +
                   '_'+ meshName + meshType + "_theta"+str(theta)+ '_Beta'+str(beta)+"/")
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   
print('results output to : ' + outputDirectory)

# -----------------------------------------------
#  Get the mesh information from FieldML data
# -----------------------------------------------
# Read xml file
fieldmlInput = inputDir + meshName + meshType + '.xml'
print('FieldML input file: ' + fieldmlInput)

#Wall boundary nodes
filename=inputDir + 'bc/wallNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfWallNodes=int(f.readline())
        wallNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Wall boundary node file: ' + filename)

#Inlet boundary nodes
filename=inputDir + 'bc/inletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletNodes=int(f.readline())
        inletNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary node file: ' + filename)
print('inlet nodes: ' + str(inletNodes))
#Inlet boundary elements
filename=inputDir + 'bc/inletElements.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletElements=int(f.readline())
        inletElements=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary element file: ' + filename)
#Inlet boundary info
normalInlet=[0.0,-1.0,0.0]

#Outlet boundary nodes
filename=inputDir + 'bc/outletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfOutletNodes=int(f.readline())
        outletNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Outlet boundary node file: ' + filename)
#Outlet boundary elements
filename=inputDir + 'bc/outletElements.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfOutletElements=int(f.readline())
        outletElements=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Outlet boundary element file: ' + filename)
#Outlet boundary info
normalOutlet=[0.0,1.0,0.0]

# -----------------------------------------------
#  Set up problem fields
# -----------------------------------------------

(coordinateSystemUserNumber,
 regionUserNumber,
 linearBasisUserNumber,
 quadraticBasisUserNumber,
 generatedMeshUserNumber,
 meshUserNumber,
 decompositionUserNumber,
 geometricFieldUserNumber,
 equationsSetFieldUserNumber,
 dependentFieldUserNumber,
 materialsFieldUserNumber,
 analyticFieldUserNumber,
 equationsSetUserNumber,
 problemUserNumber) = range(1,15)

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
region.label = "Cylinder"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Create nodes
nodes=CMISS.Nodes()
fieldmlInfo.InputNodesCreateStart("CylinderMesh.nodes.argument",region,nodes)
nodes.CreateFinish()
numberOfNodes = nodes.numberOfNodes
print("number of nodes: " + str(numberOfNodes))

# Create bases
basisNumberLinear = 1
if quadraticMesh:
    basisNumberQuadratic = 1
    basisNumberLinear = 2
    gaussQuadrature = [3,3,3]
    fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.triquadratic_lagrange",basisNumberQuadratic)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)

gaussQuadrature = [2,2,2]
fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.trilinear_lagrange",basisNumberLinear)
CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberLinear,gaussQuadrature)
CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberLinear,True)
CMISS.Basis_CreateFinishNum(basisNumberLinear)

# Create Mesh
numberOfMeshComponents=2
meshComponent1 = 1
meshComponent2 = 2
mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("CylinderMesh.mesh.argument",mesh,meshUserNumber,region)
mesh.NumberOfComponentsSet(numberOfMeshComponents)

if equalOrder:
    if quadraticMesh:
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"CylinderMesh.template.triquadratic")
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"CylinderMesh.template.triquadratic")
    else:
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"CylinderMesh.template.trilinear")
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"CylinderMesh.template.trilinear")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"CylinderMesh.template.triquadratic")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"CylinderMesh.template.trilinear")

mesh.CreateFinish()
numberOfElements = mesh.numberOfElements
print("number of elements: " + str(numberOfElements))

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CalculateFacesSet(True)
decomposition.CreateFinish()

# Create a field for the geometry
geometricField = CMISS.Field()
fieldmlInfo.InputFieldCreateStart(region,decomposition,geometricFieldUserNumber,
                                  geometricField,CMISS.FieldVariableTypes.U,
                                  "CylinderMesh.coordinates")
geometricField.CreateFinish()
fieldmlInfo.InputFieldParametersUpdate(geometricField,"CylinderMesh.node.coordinates",
                                       CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
fieldmlInfo.Finalise()


# -----------------------------------------------
#  Solve problem with provided settings
# -----------------------------------------------
""" Sets up the problem and solve with the provided parameter values


    Oscillatory flow through a rigid cylinder

                                 u=0
              ------------------------------------------- R = 0.5
                                         >
                                         ->  
    p = A*cos(frequency*t)               --> u(r,t)        p = 0
                                         ->
                                         >
              ------------------------------------------- L = 10
                                 u=0
"""
angularFrequency = 2.0*math.pi/period
if computationalNodeNumber == 0:
    print("-----------------------------------------------")
    print("Setting up problem for Womersley number: " + str(womersleyNumber))
    print("-----------------------------------------------")

# Create standard Navier-Stokes equations set
equationsSetField = CMISS.Field()
equationsSet = CMISS.EquationsSet()
equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
        CMISS.EquationsSetClasses.FLUID_MECHANICS,
        CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
        CMISS.EquationsSetSubtypes.TRANSIENT_RBS_NAVIER_STOKES,
        equationsSetFieldUserNumber, equationsSetField)
equationsSet.CreateFinish()
# Set max CFL number (default 1.0, choose 0 to skip)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,2,0.0)
# Set time increment (default 0.0)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,3,timeIncrement)
# Set stabilisation type (default 1.0 = RBS)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,4,1.0)

# Set boundary retrograde flow stabilisation scaling factor (default 0.2)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                              CMISS.FieldParameterSetTypes.VALUES, 
                                              1,beta)

# Create dependent field
dependentField = CMISS.Field()
equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
# velocity
for component in range(1,4):
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,component,meshComponent1)        
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,component,meshComponent1) 
# pressure
dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,4,meshComponent2)        
dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,4,meshComponent2) 
dependentField.DOFOrderTypeSet(CMISS.FieldVariableTypes.U,CMISS.FieldDOFOrderTypes.SEPARATED)
dependentField.DOFOrderTypeSet(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldDOFOrderTypes.SEPARATED)
equationsSet.DependentCreateFinish()
# Initialise dependent field to 0
for component in range(1,5):
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,component,0.0)
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,component,0.0)

# Initialise dependent field to analytic values
if initialiseAnalytic:
    phaseShift = math.pi/2.0
    for node in range(1,numberOfNodes+1):
        sumPositionSq = 0.
        nodeNumber = nodes.UserNumberGet(node)
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            for component in range(1,4):
                if component != axialComponent+1:
                    value=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                  CMISS.FieldParameterSetTypes.VALUES,
                                                                  1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,component)
                    sumPositionSq += value**2
                else:
                    # Pressure values
                    axialPosition=geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                                       CMISS.FieldParameterSetTypes.VALUES,
                                                                       1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,component)
                    #pressureValue = amplitude - amplitude*(axialPosition)/length
                    pressureValue = womersleyAnalytic.womersleyPressure(startTime,angularFrequency,offset,amplitude,axialPosition,length)
                    if meshName == 'hexCylinder8':
                        if node in outletNodes:
                            pressureValue = 0.0
                        elif node in inletNodes:
                            pressureValue = 1.0
                    dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                            1,1,nodeNumber,4,pressureValue)
            radialNodePosition=math.sqrt(sumPositionSq)
                    
            
            velocityValue = womersleyAnalytic.womersleyAxialVelocity(startTime,offset,amplitude,radius,
                                                                     radialNodePosition,period,viscosity,
                                                                     womersleyNumber,length)
            # Velocity value
            for component in range(1,4):
                if component == axialComponent+1:
                    value = velocityValue
                else:
                    value = 0.0
                dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        1,1,nodeNumber,component,value)
else:
    phaseShift = 0.0
dependentField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
dependentField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

# Create materials field
materialsField = CMISS.Field()
equationsSet.MaterialsCreateStart(materialsFieldUserNumber,materialsField)
equationsSet.MaterialsCreateFinish()
# Initialise materials field parameters
materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,viscosity)
materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,density)

# Create analytic field (allows for time-dependent calculation of sinusoidal pressure waveform during solve)
analytic = True
if analytic:
    analyticField = CMISS.Field()
    equationsSet.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SINUSOID,analyticFieldUserNumber,analyticField)
    equationsSet.AnalyticCreateFinish()
    for node in range(1,numberOfNodes+1):
        nodeNumber = nodes.UserNumberGet(node)
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            if nodeNumber in inletNodes:
                nodeAmplitude = 1.0
                pNorm = 1.0
            # elif nodeNumber in outletNodes:
            #     nodeAmplitude = 0.0
            #     pNorm = 1.0
            else:
                nodeAmplitude = 0.0
                pNorm = 0.0
            analyticParameters = [0.0,0.0,0.0,pNorm,nodeAmplitude,offset,angularFrequency,phaseShift,startTime,stopTime]
            parameterNumber = 0
            for parameter in analyticParameters:
                parameterNumber += 1
                analyticField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1,
                                                       nodeNumber,parameterNumber,parameter)

# Create equations
equations = CMISS.Equations()
equationsSet.EquationsCreateStart(equations)
equations.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
equations.outputType = CMISS.EquationsOutputTypes.NONE
equationsSet.EquationsCreateFinish()

# Create Navier-Stokes problem
problem = CMISS.Problem()
problem.CreateStart(problemUserNumber)
problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                         CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                         CMISS.ProblemSubTypes.TRANSIENT_RBS_NAVIER_STOKES)
problem.CreateFinish()

# Create control loops
problem.ControlLoopCreateStart()
controlLoop = CMISS.ControlLoop()
problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],controlLoop)
controlLoop.TimesSet(startTime,stopTime,timeIncrement)
controlLoop.TimeOutputSet(outputFrequency)
problem.ControlLoopCreateFinish()

# Create problem solver
dynamicSolver = CMISS.Solver()
problem.SolversCreateStart()
problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver)
dynamicSolver.outputType = CMISS.SolverOutputTypes.NONE
dynamicSolver.dynamicTheta = [theta]
nonlinearSolver = CMISS.Solver()
dynamicSolver.DynamicNonlinearSolverGet(nonlinearSolver)
nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
nonlinearSolver.outputType = CMISS.SolverOutputTypes.NONE
nonlinearSolver.newtonAbsoluteTolerance = 1.0E-7
nonlinearSolver.newtonRelativeTolerance = 1.0E-7
nonlinearSolver.newtonSolutionTolerance = 1.0E-7
nonlinearSolver.newtonMaximumFunctionEvaluations = 10000
nonlinearSolver.newtonLineSearchType = CMISS.NewtonLineSearchTypes.QUADRATIC
linearSolver = CMISS.Solver()
nonlinearSolver.NewtonLinearSolverGet(linearSolver)
linearSolver.outputType = CMISS.SolverOutputTypes.NONE
linearSolver.linearType = CMISS.LinearSolverTypes.DIRECT
linearSolver.libraryType = CMISS.SolverLibraries.MUMPS
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

# Define fixed pressure at a reference node
referenceNode = 0 #outletNodes[0]
# Create boundary conditions
boundaryConditions = CMISS.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)
print('setting up boundary conditions')
# Wall boundary nodes u = 0 (no-slip)
value=0.0
for nodeNumber in wallNodes:
    nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
    if (nodeDomain == computationalNodeNumber):
        for component in range(numberOfDimensions):
            componentId = component + 1
            boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                       1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                       nodeNumber,componentId,CMISS.BoundaryConditionsTypes.FIXED,value)

#========================
# O u t l e t
#========================
# outlet boundary nodes p = 0 
value=0.0
for nodeNumber in outletNodes:
    if (nodeNumber <= numberOfNodes): 
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent2)
        if (nodeDomain == computationalNodeNumber):
            if (nodeNumber == referenceNode):
                refPressure = value
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.FIXED,refPressure)
            else:
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.PRESSURE,value)
# Outlet boundary elements
for element in range(numberOfOutletElements):
    elementNumber = outletElements[element]
    elementDomain=decomposition.ElementDomainGet(elementNumber)
    boundaryID = 3.0
    if (elementDomain == computationalNodeNumber):
        # Boundary ID: used to identify common faces for flowrate calculation
        equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,8,boundaryID)
        # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
        equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,9,CMISS.BoundaryConditionsTypes.PRESSURE)
        # Boundary normal
        for component in range(numberOfDimensions):
            componentId = component + 5
            value = normalOutlet[component]
            equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                          elementNumber,componentId,value)

#========================
# I n l e t
#========================
# Inlet boundary nodes p = f(t) - will be updated in pre-solve
value = 0.0
for nodeNumber in inletNodes:
    if (nodeNumber <= numberOfNodes): 
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            if (nodeNumber == referenceNode):
                value = 1.0
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.FIXED_INLET,value)
            else:
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.PRESSURE,value)

# Inlet boundary elements
for element in range(numberOfInletElements):
    elementNumber = inletElements[element]
    elementDomain=decomposition.ElementDomainGet(elementNumber)
    boundaryID = 2.0
    if (elementDomain == computationalNodeNumber):
        # Boundary ID
        equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,8,boundaryID)
        # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
        equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,9,CMISS.BoundaryConditionsTypes.PRESSURE)
        # Boundary normal
        for component in range(numberOfDimensions):
            componentId = component + 5
            value = normalInlet[component]
            equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                          elementNumber,componentId,value)
solverEquations.BoundaryConditionsCreateFinish()
# Make sure fields are updated to distribute on ghost elements
equationsSetField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
equationsSetField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
# Allocate some extra MUMPS factorisation space
linearSolver.MumpsSetIcntl(14,150)

# Solve the problem
print("solving problem...")
preSolveTime = time.time()
# change to new directory and solve problem (note will return to original directory on exit)
with ChangeDirectory(outputDirectory):
    problem.Solve()
print("Problem successfully solved. Time to solve (seconds): " + str(time.time()-preSolveTime))    
CMISS.Finalise()





