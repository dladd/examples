#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an OpenCMISS script to solve Navier-Stokes flow through an aortic aneurysm using velocities fitted from phase contrast MRI. 
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
import scipy
import bisect
from scipy import interpolate
from numpy import linalg
import fittingUtils as fit
#from mpi4py import MPI


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
    logfile = 'AA_serial'
else:
    logfile = 'AA_mpi'
CMISS.OutputSetOn(logfile)


#==========================================================
# P r o b l e m     C o n t r o l
#==========================================================
#-------------------------------------------
# General parameters
#-------------------------------------------
# Set the material parameters
density  = 1050.0     # Density     (kg/m3)
viscosity= 0.0035      # Viscosity   (Pa.s)
startTime = 0.0
stopTime = 0.0100001 
timeIncrement = 0.01

# Boundary Type selection:
# -------------------------
# 0:  fit aortic inlet and subclavian, zero traction on descending aorta
# 1:  fit aortic inlet and descending aorta, zero traction on subclavian
# 2:  fit subclavian and descending aorta, zero traction on aortic inlet
boundaryType = 0

# Note problem scales are Length:mm, Time:ms, Mass:g
#==========================================================
# Material parameter scaling factors
Ls = 1000.0              # Length   (m -> mm)
Ts = 1000.0                # Time     (s -> ms)
Ms = 1000.0                # Mass     (kg -> g)
# Calculated parameters
Rhos  = Ms/(Ls**3.0)     # Density          (kg/m3)
Mus   = Ms/(Ls*Ts)       # Viscosity        (kg/(ms))
density = density*Rhos
viscosity  = viscosity*Mus
stopTime = stopTime*Ts
timeIncrement = timeIncrement*Ts
#==========================================================

outputFrequency = 1
theta = 1.0
beta = 0.0
sineInflow = False

# Mesh parameters
quadraticMesh = False
equalOrder = True
doNothingBounds = True
initialiseVelocity = True
#inputDir = './input/' + meshName +'/'
meshName = 'AorticAneurysm'
inputDir = './input/'

# PCV Data
fitData = False
solveNSE = True
pcvDataDir = inputDir + 'pcvData/'
pcvTimeIncrement = 40.0
velocityScaleFactor = 1.0/Ts # mm/s to mm/ms
numberOfPcvDataFiles = 20
dataPointResolution = [2.5,1.8,2.6]
startFit = 0
addZeroLayer = True
p = 2.5
vicinityFactor = 1.1

#==========================================================
print('dynamic theta: ' + str(theta))
print('boundary stabilisation beta: ' + str(beta))
print('viscosity = '+str(viscosity))
print('density = '+str(density))

if quadraticMesh:
    meshType = 'Quadratic'
else:
    meshType = 'Linear'
outputDirectory = ("./output/Dt" + str(round(timeIncrement,5)) +
                   '_'+ meshName + meshType +"_BoundaryType"+ str(boundaryType)+"/")
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   
print('results output to : ' + outputDirectory)
interpolatedDir = './output/interpolatedData/'
try:
    os.makedirs(interpolatedDir)
except OSError, e:
    if e.errno != 17:
        raise   
print('interpolated data output to : ' + interpolatedDir)

# -----------------------------------------------
#  Get the mesh information from FieldML data
# -----------------------------------------------
# Read xml file
fieldmlInput = inputDir + meshName + '.xml'
print('FieldML input file: ' + fieldmlInput)

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

# Aortic arch
filename=inputDir + 'bc/aorticArchNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfNodes=int(f.readline())
        aorticArchNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open aortic arch boundary node file: ' + filename)

# Descending aorta
filename=inputDir + 'bc/descendingAortaNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfNodes=int(f.readline())
        descendingAortaNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open descending aorta boundary node file: ' + filename)

# Subclavian
filename=inputDir + 'bc/subclavianNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfNodes=int(f.readline())
        subclavianNodes=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open subclavian boundary node file: ' + filename)

if boundaryType == 0:
    fittedNodes = aorticArchNodes + subclavianNodes
    numberOfFittedNodes = len(fittedNodes)
    referenceNode = descendingAortaNodes[0]
elif boundaryType == 1:
    fittedNodes = aorticArchNodes + descendingAortaNodes
    numberOfFittedNodes = len(fittedNodes)
    referenceNode = subclavianNodes[0]
elif boundaryType == 2:
    fittedNodes = subclavianNodes + descendingAortaNodes
    numberOfFittedNodes = len(fittedNodes)
    referenceNode = aorticArchNodes[0]
print('fitted nodes: '+str(fittedNodes))

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
 independentFieldUserNumber,
 materialsFieldUserNumber,
 analyticFieldUserNumber,
 equationsSetUserNumber,
 problemUserNumber) = range(1,16)

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
region.label = "AorticAneurysm"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Create nodes
nodes=CMISS.Nodes()
fieldmlInfo.InputNodesCreateStart("ArteryMesh.nodes.argument",region,nodes)
nodes.CreateFinish()
numberOfNodes = nodes.numberOfNodes
print("number of nodes: " + str(numberOfNodes))

# Create bases
basisNumberLinear = 1
quadratureOrder= 5
if quadraticMesh:
    basisNumberQuadratic = 1
    basisNumberLinear = 2
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.triquadratic_simplex",basisNumberQuadratic)
    #CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
    CMISS.Basis_QuadratureOrderSetNum(basisNumberQuadratic,quadratureOrder)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
else:
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.trilinear_simplex",basisNumberLinear)
    #CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberLinear,gaussQuadrature)
    CMISS.Basis_QuadratureOrderSetNum(basisNumberLinear,quadratureOrder)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberLinear,True)
    CMISS.Basis_CreateFinishNum(basisNumberLinear)

print('Basis creation finished')

# Create Mesh
numberOfMeshComponents=2
meshComponent1 = 1
meshComponent2 = 2
mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("ArteryMesh.mesh.argument",mesh,meshUserNumber,region)
mesh.NumberOfComponentsSet(numberOfMeshComponents)

if equalOrder:
    if quadraticMesh:
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"ArteryMesh.template.triquadratic")
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"ArteryMesh.template.triquadratic")
    else:
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"ArteryMesh.template.trilinear")
        fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"ArteryMesh.template.trilinear")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent1,"ArteryMesh.template.triquadratic")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent2,"ArteryMesh.template.trilinear")

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
print('decomposition finalised')

# Create a field for the geometry
geometricField = CMISS.Field()
fieldmlInfo.InputFieldCreateStart(region,decomposition,geometricFieldUserNumber,
                                  geometricField,CMISS.FieldVariableTypes.U,
                                  "ArteryMesh.coordinates")
geometricField.CreateFinish()
fieldmlInfo.InputFieldParametersUpdate(geometricField,"ArteryMesh.node.coordinates",
                                       CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
fieldmlInfo.Finalise()

print('fieldml finalised')

# ================================================
#  3D Navier-Stokes solve
# ================================================
""" Sets up the problem and solve with the provided parameter values


    Flow through a tet Artery

"""
if computationalNodeNumber == 0:
    print("-----------------------------------------------")
    print("      Setting up 3D Navier-Stokes problem ")
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
# Set boundary retrograde flow stabilisation scaling factor (default 0.0)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,1,beta)#DEBUG
# Set max CFL number (default 1.0, choose 0 to skip)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,2,0.0)
# Set time increment (default 0.0)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,3,timeIncrement)
# Set stabilisation type (default 1.0 = RBS)
equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                              CMISS.FieldParameterSetTypes.VALUES,4,1.0)

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

dependentField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
dependentField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

if not sineInflow:
    #=================================================================
    # Independent Field
    #=================================================================
    # Create independent field for Navier Stokes - will hold fitted values on the NSE side
    independentField = CMISS.Field()
    independentField.CreateStart(independentFieldUserNumber,region)
    independentField.LabelSet("Fitted data")
    independentField.TypeSet(CMISS.FieldTypes.GENERAL)
    independentField.MeshDecompositionSet(decomposition)
    independentField.GeometricFieldSet(geometricField)
    independentField.DependentTypeSet(CMISS.FieldDependentTypes.INDEPENDENT)
    independentField.NumberOfVariablesSet(1)
    # PCV values field
    independentField.VariableTypesSet([CMISS.FieldVariableTypes.U])
    independentField.DimensionSet(CMISS.FieldVariableTypes.U,CMISS.FieldDimensionTypes.VECTOR)
    independentField.NumberOfComponentsSet(CMISS.FieldVariableTypes.U,numberOfDimensions)
    independentField.VariableLabelSet(CMISS.FieldVariableTypes.U,"FittedData")
    for dimension in range(numberOfDimensions):
        dimensionId = dimension + 1
        independentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimensionId,1)
    independentField.CreateFinish()
    equationsSet.IndependentCreateStart(independentFieldUserNumber,independentField)
    equationsSet.IndependentCreateFinish()
    # Initialise data point vector field to 0
    independentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)
    
# Create materials field
materialsField = CMISS.Field()
equationsSet.MaterialsCreateStart(materialsFieldUserNumber,materialsField)
equationsSet.MaterialsCreateFinish()
# Initialise materials field parameters
materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,viscosity)#DEBUG
materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,density)

if sineInflow:
    analyticField = CMISS.Field()
    equationsSet.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SINUSOID,analyticFieldUserNumber,analyticField)
    equationsSet.AnalyticCreateFinish()
    for node in range(1,numberOfNodes+1):
        nodeNumber = nodes.UserNumberGet(node)
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            if nodeNumber in fittedNodes:
                nodeAmplitude = 1.0
            else:
                nodeAmplitude = 0.0
            rampPeriod = 0.25*Ts
            inletValue = 0.5
            frequency = math.pi/(rampPeriod)
            inflowAmplitude = 0.5*inletValue
            yOffset = 0.5*inletValue
            phaseShift = -math.pi/2.0 
            startSine = 0.0
            stopSine = stopTime#rampPeriod
            analyticParameters = [1.0,0.0,0.0,0.0,inflowAmplitude,yOffset,frequency,phaseShift,startSine,stopSine]
            parameterNumber = 0
            for parameter in analyticParameters:
                parameterNumber += 1
                analyticField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1,
                                                       nodeNumber,parameterNumber,parameter)

    analyticField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
    analyticField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

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

# Create boundary conditions
boundaryConditions = CMISS.BoundaryConditions()
solverEquations.BoundaryConditionsCreateStart(boundaryConditions)
print('setting up boundary conditions')

#=============================
# F i t t e d   N o d e s
#=============================
# fitted nodes - will be updated in pre-solve
value = 0.0
for nodeNumber in fittedNodes:
    if (nodeNumber <= numberOfNodes): 
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            for component in range(1,4):
                value = 0.0
                if sineInflow:
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                               nodeNumber,component,CMISS.BoundaryConditionsTypes.FIXED_INLET,value)
                else:
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                               nodeNumber,component,CMISS.BoundaryConditionsTypes.FIXED_FITTED,value)

#========================
# W a l l
#========================
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

#====================================
# R e f e r e n c e   P r e s s u r e
#====================================
value=0.0
nodeNumber = referenceNode
nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
if (nodeDomain == computationalNodeNumber):
    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                               1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                               nodeNumber,4,CMISS.BoundaryConditionsTypes.FIXED,value)

solverEquations.BoundaryConditionsCreateFinish()
# Make sure fields are updated to distribute on ghost elements
equationsSetField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
equationsSetField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
# Allocate some extra MUMPS factorisation space
linearSolver.MumpsSetIcntl(14,150)

if fitData:
    #=======================================================================
    # D a t a    P o i n t   T i m e   F i t t i n g
    #=======================================================================
    if computationalNodeNumber == 0:
        print("-----------------------------------------------")
        print("      Setting up fitting problem ")
        print("-----------------------------------------------")
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
        dataPoints.LabelSet(dataPointId,"PCVDataPoints")
    dataPoints.CreateFinish()

    #--------------------------------------------
    # Read in velocity data
    #--------------------------------------------

    numberOfTimesteps = int((stopTime - startTime)/timeIncrement) + 1
    print('number of timesteps to solve: ' + str(numberOfTimesteps))
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

        outputExtras = False
        if outputExtras:
            outputFile = outputDirectory + "pcv/dataPoints" + str(pcvTimestep) + ".C"
            exnodeFile = outputDirectory + "pcv/dataPoints" + str(pcvTimestep) + ".exnode"
            fit.writeDataPoints(pcvGeometry,pcvData[pcvTimestep],outputFile)
            os.system("perl $scripts/meshConversion/dataPointsConversion.pl "+ outputFile + " 1000000 " + exnodeFile)

    print(pcvData.shape)

    startTempInterpTime = time.time()
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
    #print(velocityDataPoints)
    endTempInterpTime = time.time()
    tempInterpTime = endTempInterpTime - startTempInterpTime 

    #=================================================================
    # Inverse Distance Weighting
    #=================================================================
    startIdwTime = time.time()
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
        if timestep == startFit:
            # Update Field
            if initialiseVelocity:
                print('Initialising velocity field to PCV solution')
                for nodeNumberPython in nodeList:
                    nodeNumberCmiss = nodeNumberPython + 1
                    for component in xrange(numberOfDimensions):
                        componentId = component + 1
                        value = velocityNodes[nodeNumberPython,component]
                        dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                                CMISS.FieldParameterSetTypes.VALUES,
                                                                1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                                nodeNumberCmiss,componentId,value)
            else:
                velocityNodes = numpy.zeros((numberOfNodes,3))

        if(numberOfComputationalNodes == 1):
            filename = interpolatedDir + "fitData" + str(timestep) + ".dat"
            print("writing: " + filename)
            numpy.savetxt(filename,velocityNodes[:,:], delimiter=' ')
        else:    
            # keeping in a temp dir keeps node0 below from reading in a file before it is finished writing
            filename = interpolatedDir + "temp/fitData" + str(timestep) + ".part" + str(computationalNodeNumber) + ".dat"
            if (os.path.exists(filename)):
                print("replacing file " + filename)
                numpy.savetxt(filename,velocityNodes[:,:], delimiter=' ')
                os.rename(filename,interpolatedDir + "fitData" + str(timestep) + ".part" + str(computationalNodeNumber) + ".dat")
            else:
                print("writing: " + filename)
                numpy.savetxt(filename,velocityNodes[:,:], delimiter=' ')
                os.rename(filename,interpolatedDir + "fitData" + str(timestep) + ".part" + str(computationalNodeNumber) + ".dat")

            allCompNodes = False
            while (allCompNodes == False):
                gathered = []
                #Gather process velocityNodes on node 0
                if (computationalNodeNumber == 0):
                    outputData = numpy.zeros((numberOfNodes,3))
                    for compNode in xrange(numberOfComputationalNodes):
                        filename = interpolatedDir + "fitData" + str(timestep) + ".part" + str(compNode) + ".dat"
                        if (os.path.exists(filename)):
                            if(compNode not in gathered):
                                gathered.append(compNode)
                                velocityNodes = numpy.loadtxt(filename, delimiter=' ')
                                outputData = outputData + velocityNodes
                                if(len(gathered) == numberOfComputationalNodes):
                                    filename = interpolatedDir + "fitData" + str(timestep) + ".dat"
                                    print("Writing " + filename)
                                    f = open(filename,"w")
                                    numpy.savetxt(f,outputData[:,:], delimiter=' ')
                                    f.close()
                                    for i in xrange(numberOfComputationalNodes):
                                        filename = interpolatedDir + "fitData" + str(timestep) + ".part" + str(i) + ".dat"
                                        print("Deleting: " + filename)
                                        os.remove(filename)
                                    allCompNodes = True
                                    break
                            else:
                                break
                else:
                    # Don't let other processors get too far ahead of control node
                    filename = interpolatedDir + "fitData" + str(timestep-1) + ".dat"
                    if (os.path.exists(filename)):
                        break
                    else:
                        if (timestep == 0):
                            break
                        else:
                            time.sleep(10)
                        # break
        
        # Export results
        if outputExtras:
            timeStart = time.time()            
            print("exporting CMGUI data")
            # Export results
            fields = CMISS.Fields()
            fields.CreateRegion(region)
            print('mesh name: ' + meshName)
            fields.NodesExport( outputDirectory +'/fittingResults/' + meshName + "_t"+ str(timestep),"FORTRAN")
            fields.Finalise()
            timeStop = time.time()            
            print("Finished CMGUI data export, time: " + str(timeStop-timeStart))

    endIdwTime = time.time()
    idwTime = endIdwTime - startIdwTime 
    print("time for idw solve: " + str(idwTime))

# Let all computational nodes catch up here:
#MPI.COMM_WORLD.Barrier()
if initialiseVelocity and not fitData:
    filename = interpolatedDir + "fitData0.dat"
    velocityNodes = numpy.loadtxt(filename)
    for node in range(1,numberOfNodes+1):
        nodeNumber = nodes.UserNumberGet(node)
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,meshComponent1)
        if (nodeDomain == computationalNodeNumber):
            if nodeNumber not in wallNodes:
                for component in xrange(numberOfDimensions):
                    componentId = component + 1
                    value = velocityNodes[nodeNumber-1,component]
                    dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                            CMISS.FieldParameterSetTypes.VALUES,
                                                            1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                                            nodeNumber,componentId,value)

# Finish the parameter update
dependentField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
dependentField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)   
if solveNSE:
    # Solve the problem
    print("solving problem...")
    preSolveTime = time.time()
    # change to new directory and solve problem (note will return to original directory on exit)
    with ChangeDirectory(outputDirectory):
        problem.Solve()
    print("CFD problem successfully solved. Time to solve (seconds): " + str(time.time()-preSolveTime))    
if fitData:
    print("time for idw solve: " + str(idwTime))
CMISS.Finalise()





