#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an OpenCMISS script to solve Navier-Stokes flow through
#>  a 3D cylinder coupled with a straight 1D conduit via an interface condition. 
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

#> \example FluidMechanics/NavierStokes/Coupled3D1D/Python/Coupled3D1DExample.py
## Python OpenCMISS script to solve Navier-Stokes flow through a 3D cylinder coupled with a straight 1D conduit via an interface condition.
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
import FluidExamples1DUtilities as Utilities1D

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


#==========================================================
# P r o b l e m     C o n t r o l
#==========================================================

#-------------------------------------------
# General parameters
#-------------------------------------------
# Set the material parameters
density  = 1050.0     # Density     (kg/m3)
viscosity= 0.004      # Viscosity   (Pa.s)
G0   = 0.0        # Gravitational acceleration (m/s2)
Pext = 0.0        # External pressure (Pa)

couplingTolerance3D1D = 1.0E-7
startTime = 0.0
stopTime = 0.200001 
timeIncrement = 0.001
outputFrequency = 1
totalLength = 20.0
radius = 0.5

#-------------------------------------------
# 3D parameters
#-------------------------------------------
length3D = 10.0
radius3D = radius
couplingNode3D = 184
quadraticMesh = True
inputDir = "./input/"
meshName = "hexCylinder140"
analyticInflow = True
inletValue = 50.0#0.001#1.0
inletCoeff = [0.0,1.0,0.0,0.0]

#-------------------------------------------
# 1D parameters
#-------------------------------------------
numberOfNodes1D = 5
numberOfElements1D = 2
checkTimestepStability = False
couplingTolerance1D = 1.0E10

Q0 = 0.0
dQ = 0.0
dA = 0.0

# Materials parameters
A0   = math.pi*(radius/100.0)**2.0  # reference Area (m2)
H    = 0.00163                      # vessel thickness (m)
E    = 400000                       # Elasticity Pa    (kg/(ms2)
Alpha = 1.0                         # Flow profile type

# Material parameter scaling factors
Ls = 100.0              # Length   (m -> cm)
Ts = 1.0                # Time     (s -> s)
Ms = 1.0                # Mass     (kg -> kg)

# calculate specific scale factors
Qs    = (Ls**3.0)/Ts     # Flow             (m3/s)  
As    = Ls**2.0          # Area             (m2)
Hs    = Ls               # vessel thickness (m)
Es    = Ms/(Ls*Ts**2.0)  # Elasticity Pa    (kg/(ms2)
Rhos  = Ms/(Ls**3.0)     # Density          (kg/m3)
Mus   = Ms/(Ls*Ts)       # Viscosity        (kg/(ms))
Ps    = Ms/(Ls*Ts**2.0)  # Pressure         (kg/(ms2))
Gs    = Ls/(Ts**2.0)     # Acceleration    (m/s2)


# Apply scale factors        
density = density*Rhos
viscosity  = viscosity*Mus
P   = Pext*Ps
A0  = A0*As
E   = E*Es
H   = H*Hs
G0  = G0*Gs

#==========================================================
# Setup
#==========================================================

if computationalNodeNumber == 0:
    print(""" Setting up the problem and solving


        Coupled flow from a 3D rigid cylinder to a 1D conduit

                           u=0         |
        -------------------------------| R = 0.5
        >                              |
        ->                             |
        --> u(r,t)                     |------------------------------- P(A)=0
        ->                             |
        >                              |
        -------------------------------| L = 10                         L = 20
                           u=0         |
    """)

startRunTime = time.time()

# -----------------------------------------------
#  3D: Get the mesh information from FieldML data
# -----------------------------------------------
# Read xml file
if quadraticMesh:
    meshType = 'Quadratic'
else:
    meshType = 'Linear'

fieldmlInput = inputDir + meshName + '/'+meshName+ meshType + '.xml'
print('FieldML input file: ' + fieldmlInput)

#Wall boundary nodes
filename=inputDir + meshName + '/bc/wallNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfWallNodes3D=int(f.readline())
        wallNodes3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Wall boundary node file: ' + filename)

#Inlet boundary nodes
filename=inputDir + meshName + '/bc/inletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletNodes3D=int(f.readline())
        inletNodes3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary node file: ' + filename)
#Inlet boundary elements
filename=inputDir + meshName + '/bc/inletElements.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletElements3D=int(f.readline())
        inletElements3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary element file: ' + filename)
#Inlet boundary info
normalInlet3D=[0.0,-1.0,0.0]

#Outlet boundary nodes
filename=inputDir + meshName + '/bc/outletNodes'+meshType+'.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfOutletNodes3D=int(f.readline())
        outletNodes3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Outlet boundary node file: ' + filename)
#Outlet boundary elements
filename=inputDir + meshName + '/bc/outletElements.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfOutletElements3D=int(f.readline())
        outletElements3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Outlet boundary element file: ' + filename)
#Outlet boundary info
normalOutlet3D=[0.0,1.0,0.0]

# -----------------------------------------------
#  Set up Fields and equations
# -----------------------------------------------

(coordinateSystem3DUserNumber,
 region3DUserNumber,
 linearBasis3DUserNumber,
 quadraticBasis3DUserNumber,
 mesh3DUserNumber,
 decomposition3DUserNumber,
 geometricField3DUserNumber,
 equationsSetField3DUserNumber,
 dependentField3DUserNumber,
 materialsField3DUserNumber,
 analyticField3DUserNumber,
 equationsSet3DUserNumber,
 coordinateSystem1DUserNumber,
 region1DUserNumber,
 quadraticBasis1DUserNumber,
 mesh1DUserNumber,
 decomposition1DUserNumber,
 geometricField1DUserNumber,
 equationsSetField1DNSUserNumber,
 equationsSetField1DCUserNumber,
 dependentField1DUserNumber,
 independentField1DUserNumber,
 materialsField1DUserNumber,
 analyticField1DUserNumber,
 equationsSet1DCUserNumber,
 equationsSet1DNSUserNumber,
 problemUserNumber) = range(1,28)

# ========================================================
#  Set up coordinate systems, regions, and geometry
# ========================================================

#-----------------------------------
# 3D 
#-----------------------------------
#Initialise fieldML IO
fieldmlInfo=CMISS.FieldMLIO()
fieldmlInfo.InputCreateFromFile(fieldmlInput)

# Create a RC coordinate system
coordinateSystem3D = CMISS.CoordinateSystem()
fieldmlInfo.InputCoordinateSystemCreateStart("CylinderMesh.coordinates",coordinateSystem3D,coordinateSystem3DUserNumber)
coordinateSystem3D.CreateFinish()
numberOfDimensions = coordinateSystem3D.DimensionGet()

# Create a region
region3D = CMISS.Region()
region3D.CreateStart(region3DUserNumber,CMISS.WorldRegion)
region3D.label = "3DCylinder"
region3D.coordinateSystem = coordinateSystem3D
region3D.CreateFinish()

# Create nodes
nodes3D=CMISS.Nodes()
fieldmlInfo.InputNodesCreateStart("CylinderMesh.nodes.argument",region3D,nodes3D)
nodes3D.CreateFinish()
numberOfNodes3D = nodes3D.numberOfNodes
print("number of nodes 3D: " + str(numberOfNodes3D))

# Create bases
if (quadraticMesh):
    basisNumberQuadratic = 1
    gaussQuadrature = [3,3,3]
    fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.triquadratic_lagrange",basisNumberQuadratic)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
else:
    basisNumberLinear = 1
    gaussQuadrature = [2,2,2]
    fieldmlInfo.InputBasisCreateStartNum("CylinderMesh.trilinear_lagrange",basisNumberLinear)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberLinear,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberLinear,True)
    CMISS.Basis_CreateFinishNum(basisNumberLinear)

# Create Mesh
meshComponent3DVelocity=1
meshComponent3DPressure=2
mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("CylinderMesh.mesh.argument",mesh,mesh3DUserNumber,region3D)
mesh.NumberOfComponentsSet(2)

if (quadraticMesh):
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DVelocity,"CylinderMesh.template.triquadratic")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DPressure,"CylinderMesh.template.triquadratic")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DVelocity,"CylinderMesh.template.trilinear")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DPressure,"CylinderMesh.template.trilinear")

mesh.CreateFinish()
numberOfElements = mesh.numberOfElements
print("number of elements: " + str(numberOfElements))

# Create a decomposition for the mesh
decomposition3D = CMISS.Decomposition()
decomposition3D.CreateStart(decomposition3DUserNumber,mesh)
decomposition3D.type = CMISS.DecompositionTypes.CALCULATED
decomposition3D.numberOfDomains = numberOfComputationalNodes
decomposition3D.CalculateFacesSet(True)
decomposition3D.CreateFinish()

# Create a field for the geometry
geometricField3D = CMISS.Field()
fieldmlInfo.InputFieldCreateStart(region3D,decomposition3D,geometricField3DUserNumber,
                                  geometricField3D,CMISS.FieldVariableTypes.U,
                                  "CylinderMesh.coordinates")
geometricField3D.CreateFinish()
fieldmlInfo.InputFieldParametersUpdate(geometricField3D,"CylinderMesh.node.coordinates",
                                       CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField3D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
geometricField3D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,
                                       CMISS.FieldParameterSetTypes.VALUES)
fieldmlInfo.Finalise()

#-----------------------------------
# 1D
#-----------------------------------
#couplingNode1D = numberOfNodes3D + 1
#outletNode1D = couplingNode1D + numberOfNodes1D - 1

# Create a RC coordinate system
sameCoordinates = True
if sameCoordinates:
    coordinateSystem1D = coordinateSystem3D
else:
    coordinateSystem1D = CMISS.CoordinateSystem()
    coordinateSystem1D.CreateStart(coordinateSystem1DUserNumber)
    # Embed 1D equations in 3D space
    coordinateSystem1D.DimensionSet(numberOfDimensions) 
    coordinateSystem1D.CreateFinish()

# Create a region
sameRegion = False
if sameRegion:
    region1D = region3D
    couplingNode1D = numberOfNodes3D + 1
    outletNode1D = numberOfNodes3D + numberOfNodes1D
else:
    couplingNode1D = 1
    outletNode1D = numberOfNodes1D
    region1D = CMISS.Region()
    region1D.CreateStart(region1DUserNumber,CMISS.WorldRegion)
    region1D.label = "1DConduit"
    region1D.coordinateSystem = coordinateSystem1D
    region1D.CreateFinish()

# Create a basis
basisXiGauss = 3
basis1D = CMISS.Basis()
basis1D.CreateStart(quadraticBasis1DUserNumber)
basis1D.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
basis1D.numberOfXi = 1
basis1D.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]
basis1D.quadratureNumberOfGaussXi = [basisXiGauss]
basis1D.CreateFinish()

# Create nodes
nodes1D = CMISS.Nodes()
nodes1D.CreateStart(region1D,numberOfNodes1D)
nodes1D.CreateFinish()

# Create mesh
mesh1D = CMISS.Mesh()
mesh1D.CreateStart(mesh1DUserNumber,region1D,numberOfDimensions)
mesh1D.NumberOfElementsSet(numberOfElements1D)
# Specify the mesh components
mesh1D.NumberOfComponentsSet(1)
# Specify the mesh components
meshElements = CMISS.MeshElements()
meshComponentNumber = 1
meshElements.CreateStart(mesh1D,meshComponentNumber,basis1D)
nodeIdx = couplingNode1D
for element in range(1,numberOfElements1D+1):
    elementNodes = []
    for elementNode in range(3):
        elementNodes.append(elementNode+nodeIdx)
    meshElements.NodesSet(element,elementNodes)
    nodeIdx+=2
meshElements.CreateFinish()
mesh1D.CreateFinish() 

# Create a decomposition for the mesh
decomposition1D = CMISS.Decomposition()
decomposition1D.CreateStart(decomposition1DUserNumber,mesh1D)
decomposition1D.type = CMISS.DecompositionTypes.EVERY
decomposition1D.numberOfDomains = numberOfComputationalNodes
decomposition1D.CreateFinish()

# Create a field for the geometry
geometricField1D = CMISS.Field()
geometricField1D.CreateStart(geometricField1DUserNumber,region1D)
geometricField1D.MeshDecompositionSet(decomposition1D)
geometricField1D.TypeSet(CMISS.FieldTypes.GEOMETRIC)
geometricField1D.VariableLabelSet(CMISS.FieldVariableTypes.U,"Geometry")
geometricField1D.fieldScalingType = CMISS.FieldScalingTypes.NONE
geometricField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
geometricField1D.CreateFinish()

# Set node positions along Y axis
length1D = totalLength - length3D
lengthIncrement = length1D/(numberOfNodes1D-1)
for node in range(numberOfNodes1D):
    nodeNumber = node+couplingNode1D
    yPosition = length3D+node*lengthIncrement
    geometricField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1,
                                              nodeNumber,2,yPosition)
# Finish the parameter update
geometricField1D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
geometricField1D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)     

# ========================================================
#  Set up equations sets
# ========================================================

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create standard Navier-Stokes equations set
equationsSetField3D = CMISS.Field()
equationsSet3D = CMISS.EquationsSet()
equationsSet3D.CreateStart(equationsSet3DUserNumber,region3D,geometricField3D,
        CMISS.EquationsSetClasses.FLUID_MECHANICS,
        CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
        CMISS.EquationsSetSubtypes.TRANSIENT_RBS_NAVIER_STOKES,
                           equationsSetField3DUserNumber, equationsSetField3D)
equationsSet3D.CreateFinish()

# Set boundary retrograde flow stabilisation scaling factor (default 0.0)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,1,-1.0)
# Set max CFL number (default 1.0)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,2,1.0E20)
# Set time increment (default 0.0)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,3,timeIncrement)
# Set stabilisation type (default 1.0 = RBS)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,4,1.0)

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic equations set
equationsSetField1DC = CMISS.Field()
equationsSet1DC = CMISS.EquationsSet()
equationsSet1DC.CreateStart(equationsSet1DCUserNumber,region1D,geometricField1D,
                           CMISS.EquationsSetClasses.FLUID_MECHANICS,
                           CMISS.EquationsSetTypes.CHARACTERISTIC_EQUATION,
                           CMISS.EquationsSetSubtypes.CHARACTERISTIC,
                             equationsSetField1DCUserNumber, equationsSetField1DC)
equationsSet1DC.CreateFinish()
# Navier-Stokes equations set
equationsSetField1DNS = CMISS.Field()
equationsSet1DNS = CMISS.EquationsSet()
equationsSet1DNS.CreateStart(equationsSet1DNSUserNumber,region1D,geometricField1D,
                           CMISS.EquationsSetClasses.FLUID_MECHANICS,
                           CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                           CMISS.EquationsSetSubtypes.TRANSIENT1D_NAVIER_STOKES,
                             equationsSetField1DNSUserNumber,equationsSetField1DNS)
equationsSet1DNS.CreateFinish()


# ========================================================
#  Dependent Field
# ========================================================

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create dependent field
dependentField3D = CMISS.Field()
equationsSet3D.DependentCreateStart(dependentField3DUserNumber,dependentField3D)
# velocity
for component in range(1,4):
    dependentField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,component,meshComponent3DVelocity)        
    dependentField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,component,meshComponent3DVelocity) 
# pressure
dependentField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,4,meshComponent3DPressure)        
dependentField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,4,meshComponent3DPressure) 
dependentField3D.DOFOrderTypeSet(CMISS.FieldVariableTypes.U,CMISS.FieldDOFOrderTypes.SEPARATED)
dependentField3D.DOFOrderTypeSet(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldDOFOrderTypes.SEPARATED)
equationsSet3D.DependentCreateFinish()
# Initialise dependent field to 0
for component in range(1,5):
    dependentField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,component,0.0)
    dependentField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,component,0.0)

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic equations
dependentField1DNS = CMISS.Field()
equationsSet1DC.DependentCreateStart(dependentField1DUserNumber,dependentField1DNS)
dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.U,'Flow_and_Area')
dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Derivatives')
dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.V,'Characteristics')
dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.U2,'Pressure_Stress_Flow')
# Flow & Area
meshComponentNumber=1
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumber)
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumber)
# Derivatives
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,meshComponentNumber)
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,2,meshComponentNumber)
# Riemann
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,1,meshComponentNumber)
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,2,meshComponentNumber)
# Pressure
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,1,meshComponentNumber)
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,2,meshComponentNumber)
equationsSet1DC.DependentCreateFinish()

# Navier-Stokes
equationsSet1DNS.DependentCreateStart(dependentField1DUserNumber,dependentField1DNS)
equationsSet1DNS.DependentCreateFinish()
dependentField1DNS.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES)

# Initialise the dependent field variables
versionIdx = 1
for nodeIdx in range (couplingNode1D,couplingNode1D+numberOfNodes1D):
    #nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    #if (nodeDomain == computationalNodeNumber):
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,1,Q0)
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,2,A0)
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                versionIdx,1,nodeIdx,1,Q0)
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                versionIdx,1,nodeIdx,2,A0)
    # delUdelN variables
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,1,dQ)
    dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,2,dA)
# Finish the parameter update
dependentField1DNS.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
dependentField1DNS.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)   

# ========================================================
#  Materials Field
# ========================================================

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create materials field
materialsField3D = CMISS.Field()
equationsSet3D.MaterialsCreateStart(materialsField3DUserNumber,materialsField3D)
equationsSet3D.MaterialsCreateFinish()
# Initialise materials field parameters
materialsField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,viscosity)
materialsField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,density)

# Create analytic field (allows for time-dependent calculation of sinusoidal pressure waveform during solve)
if analyticInflow:
    # yOffset + amplitude*sin(frequency*time + phaseShift))))
    # Set the time it takes to ramp pressure to fully developed
    rampPeriod = 0.1
    frequency = math.pi/(rampPeriod)
    inflowAmplitude = 0.5*inletValue
    yOffset = 0.5*inletValue
    phaseShift = -math.pi/2.0 
    startSine = 0.0
    stopSine = stopTime#rampPeriod

    analyticField = CMISS.Field()
    equationsSet3D.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SINUSOID,analyticField3DUserNumber,analyticField)
    equationsSet3D.AnalyticCreateFinish()
    analyticParameters = [inletCoeff[0],inletCoeff[1],inletCoeff[2],inletCoeff[3],
                          inflowAmplitude,yOffset,frequency,phaseShift,startSine,stopSine]
    parameterNumber = 0
    for parameter in analyticParameters:
        parameterNumber += 1
        for nodeNumber in inletNodes3D:
            nodeDomain = decomposition3D.NodeDomainGet(nodeNumber,meshComponentNumber)
            if (nodeDomain == computationalNodeNumber):
                analyticField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1,
                                                       nodeNumber,parameterNumber,parameter)

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
# Create the equations set materials field variables 
materialsField1DNS = CMISS.Field()
equationsSet1DC.MaterialsCreateStart(materialsField1DUserNumber,materialsField1DNS)
materialsField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.U,'MaterialsConstants')
materialsField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.V,'MaterialsVariables')
# Set the mesh component to be used by the field components.
for componentNumber in range(1,4):
    materialsField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,componentNumber,meshComponentNumber)
equationsSet1DC.MaterialsCreateFinish()

# Navier-Stokes
equationsSet1DNS.MaterialsCreateStart(materialsField1DUserNumber,materialsField1DNS)
equationsSet1DNS.MaterialsCreateFinish()
# Set the materials field constants
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,1,viscosity)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,2,density)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,3,Alpha)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,4,Pext)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,5,Ls)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,6,Ts)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,7,Ms)
materialsField1DNS.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                               CMISS.FieldParameterSetTypes.VALUES,8,G0)
# Set node-based materials
versionIdx = 1
for nodeIdx in range (couplingNode1D,couplingNode1D+numberOfNodes1D):
    #nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    #if (nodeDomain == computationalNodeNumber):
    materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,1,A0)
    materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,2,E)
    materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,1,nodeIdx,3,H)

# Finish the parameter update
materialsField1DNS.ParameterSetUpdateStart(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
materialsField1DNS.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)

# ========================================================
#  Independent Field
# ========================================================

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
# Create the equations set independent field variables  
independentField1D = CMISS.Field()
equationsSet1DC.IndependentCreateStart(independentField1DUserNumber,independentField1D)
independentField1D.VariableLabelSet(CMISS.FieldVariableTypes.U,'Normal Wave Direction')
# Set the mesh component to be used by the field components.
independentField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumber)
independentField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumber)
equationsSet1DC.IndependentCreateFinish()

# Navier-Stokes
equationsSet1DNS.IndependentCreateStart(independentField1DUserNumber,independentField1D)
equationsSet1DNS.IndependentCreateFinish()

# Set the normal wave direction for inlet and outlet
# Inlet
versionIdx = 1
#nodeDomain = decomposition1D.NodeDomainGet(couplingNode1D,meshComponentNumber)
#if (nodeDomain == computationalNodeNumber):
# Incoming (parent) only - reflected wave will be 0
independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                            versionIdx,1,couplingNode1D,2,-1.0)
# Outlet
versionIdx = 1
#nodeDomain = decomposition1D.NodeDomainGet(outletNode1D,meshComponentNumber)
#if (nodeDomain == computationalNodeNumber):
# Outgoing (parent) only - reflected wave will be 0
independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                            versionIdx,1,outletNode1D,1,1.0)
# Finish the parameter update
independentField1D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
independentField1D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

# ========================================================
#  Equations
# ========================================================

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create equations
equations3D = CMISS.Equations()
equationsSet3D.EquationsCreateStart(equations3D)
equations3D.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
equations3D.outputType = CMISS.EquationsOutputTypes.NONE
equationsSet3D.EquationsCreateFinish()

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
equations1DC = CMISS.Equations()
equationsSet1DC.EquationsCreateStart(equations1DC)
equations1DC.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
equations1DC.outputType = CMISS.EquationsOutputTypes.NONE
equationsSet1DC.EquationsCreateFinish()

# Navier-Stokes
equations1DNS = CMISS.Equations()
equationsSet1DNS.EquationsCreateStart(equations1DNS)
equations1DNS.sparsityType = CMISS.EquationsSparsityTypes.FULL
equations1DNS.lumpingType = CMISS.EquationsLumpingTypes.UNLUMPED
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
equations1DNS.outputType = CMISS.EquationsOutputTypes.NONE
equationsSet1DNS.EquationsCreateFinish()


# ========================================================
#  Problem
# ========================================================
# Create coupled (multiscale) problem
problem = CMISS.Problem()
problem.CreateStart(problemUserNumber)
problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                         CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                         CMISS.ProblemSubTypes.MULTISCALE_NAVIER_STOKES)
problem.CreateFinish()

# ========================================================
#  Control Loops
# ========================================================
#   L1                 L2                         L3                                 L4
#
#
#  Time Loop --- | Iterative 3D-1D Coupling --- | 1D NS/C coupling subloop --- | CharacteristicSolver (solver 1)
#                  (while loop, subloop 1)      | (while loop, subloop 1)      | 1DNavierStokesSolver (solver 2)
#                                               |
#                                               | 3D Navier-Stokes subloop --- | 3DNavierStokesSolver (solver 1)
#                                               | (simple, subloop 2)

problem.ControlLoopCreateStart()

# Create time loop
timeLoop = CMISS.ControlLoop()
problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],timeLoop)
timeLoop.LabelSet('Time Loop')
timeLoop.TimesSet(startTime,stopTime,timeIncrement)
timeLoop.TimeOutputSet(outputFrequency)

# Create iterative 3D-1D coupling loop
iterative3D1DLoop = CMISS.ControlLoop()
problem.ControlLoopGet([1,CMISS.ControlLoopIdentifiers.NODE],iterative3D1DLoop)
iterative3D1DLoop.AbsoluteToleranceSet(couplingTolerance3D1D)

# Create iterative 1D NS/C coupling loop
iterative1DLoop = CMISS.ControlLoop()
problem.ControlLoopGet([1,1,CMISS.ControlLoopIdentifiers.NODE],iterative1DLoop)
iterative1DLoop.AbsoluteToleranceSet(couplingTolerance1D)

# Create simple 3D NS loop
simple3DLoop = CMISS.ControlLoop()
problem.ControlLoopGet([1,2,CMISS.ControlLoopIdentifiers.NODE],simple3DLoop)

problem.ControlLoopCreateFinish()

# ========================================================
#  Solvers
# ========================================================
problem.SolversCreateStart()

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
nonlinearSolver1DC = CMISS.Solver()
problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],1,nonlinearSolver1DC)
nonlinearSolver1DC.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS)
nonlinearSolver1DC.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
nonlinearSolver1DC.newtonAbsoluteTolerance = 1.0E-10
nonlinearSolver1DC.newtonSolutionTolerance = 1.0E-10
nonlinearSolver1DC.newtonRelativeTolerance = 1.0E-10
# Get the nonlinear linear solver
linearSolver1DC = CMISS.Solver()
nonlinearSolver1DC.NewtonLinearSolverGet(linearSolver1DC)
linearSolver1DC.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
linearSolver1DC.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
linearSolver1DC.LinearIterativeMaximumIterationsSet(100000)
linearSolver1DC.LinearIterativeDivergenceToleranceSet(1.0E+10)
linearSolver1DC.LinearIterativeRelativeToleranceSet(1.0E-10)
linearSolver1DC.LinearIterativeAbsoluteToleranceSet(1.0E-10)
linearSolver1DC.LinearIterativeGMRESRestartSet(3000)

# Navier-Stokes
dynamicSolver1DNS = CMISS.Solver()
problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],2,dynamicSolver1DNS)
dynamicSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
dynamicSolver1DNS.dynamicTheta = [1.0]
# Get the dynamic nonlinear solver
nonlinearSolver1DNS = CMISS.Solver()
dynamicSolver1DNS.DynamicNonlinearSolverGet(nonlinearSolver1DNS)
# Set the nonlinear Jacobian type
nonlinearSolver1DNS.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS)
nonlinearSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
nonlinearSolver1DNS.NewtonAbsoluteToleranceSet(1.0E-8)
nonlinearSolver1DNS.NewtonSolutionToleranceSet(1.0E-8)
nonlinearSolver1DNS.NewtonRelativeToleranceSet(1.0E-8)
# Get the dynamic nonlinear linear solver
linearSolver1DNS = CMISS.Solver()
nonlinearSolver1DNS.NewtonLinearSolverGet(linearSolver1DNS)
linearSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
linearSolver1DNS.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
linearSolver1DNS.LinearIterativeMaximumIterationsSet(100000)
linearSolver1DNS.LinearIterativeDivergenceToleranceSet(1.0E+10)
linearSolver1DNS.LinearIterativeRelativeToleranceSet(1.0E-8)
linearSolver1DNS.LinearIterativeAbsoluteToleranceSet(1.0E-8)
linearSolver1DNS.LinearIterativeGMRESRestartSet(3000)

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create problem solver
dynamicSolver3D = CMISS.Solver()
problem.SolverGet([1,2,CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver3D)
dynamicSolver3D.outputType = CMISS.SolverOutputTypes.NONE
dynamicSolver3D.dynamicTheta = [1.0]
nonlinearSolver3D = CMISS.Solver()
dynamicSolver3D.DynamicNonlinearSolverGet(nonlinearSolver3D)
nonlinearSolver3D.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
nonlinearSolver3D.outputType = CMISS.SolverOutputTypes.NONE
nonlinearSolver3D.newtonAbsoluteTolerance = 1.0E-8
nonlinearSolver3D.newtonRelativeTolerance = 1.0E-8
nonlinearSolver3D.newtonSolutionTolerance = 1.0E-8
nonlinearSolver3D.newtonMaximumFunctionEvaluations = 10000
linearSolver3D = CMISS.Solver()
nonlinearSolver3D.NewtonLinearSolverGet(linearSolver3D)
linearSolver3D.outputType = CMISS.SolverOutputTypes.NONE
linearSolver3D.linearType = CMISS.LinearSolverTypes.DIRECT
linearSolver3D.libraryType = CMISS.SolverLibraries.MUMPS

# Finish the creation of the problem solvers
problem.SolversCreateFinish()

# ========================================================
#  Solver Equations
# ========================================================
problem.SolverEquationsCreateStart()

# -----------------------------------------------
#  1D
# -----------------------------------------------
solver1DC = CMISS.Solver()
solver1DNS = CMISS.Solver()
solverEquations1DC = CMISS.SolverEquations()
solverEquations1DNS = CMISS.SolverEquations()
# Characteristic
problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],1,solver1DC)
solver1DC.SolverEquationsGet(solverEquations1DC)
solverEquations1DC.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
equationsSet1DC = solverEquations1DC.EquationsSetAdd(equationsSet1DC)
# Navier-Stokes
problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],2,solver1DNS)
solver1DNS.SolverEquationsGet(solverEquations1DNS)
solverEquations1DNS.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
equationsSet1DNS = solverEquations1DNS.EquationsSetAdd(equationsSet1DNS)

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create solver equations and add equations set to solver equations
solver3D = CMISS.Solver()
solverEquations3D = CMISS.SolverEquations()

problem.SolverGet([1,2,CMISS.ControlLoopIdentifiers.NODE],1,solver3D)
solver3D.SolverEquationsGet(solverEquations3D)
solverEquations3D.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
equationsSet3D = solverEquations3D.EquationsSetAdd(equationsSet3D)

problem.SolverEquationsCreateFinish()


# ========================================================
#  Boundary Conditions
# ========================================================

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create boundary conditions
boundaryConditions3D = CMISS.BoundaryConditions()
solverEquations3D.BoundaryConditionsCreateStart(boundaryConditions3D)

# W a l l
#--------------
# Wall boundary nodes u = 0 (no-slip)
value=0.0
for nodeNumber in wallNodes3D:
    nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DVelocity)
    if (nodeDomain == computationalNodeNumber):
        for component in range(numberOfDimensions):
            componentId = component + 1
            boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                         1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                         nodeNumber,componentId,CMISS.BoundaryConditionsTypes.FIXED,value)
# O u t l e t
#--------------
# Outlet boundary nodes: Coupling stress (traction)
value=0.0
for nodeNumber in outletNodes3D:
    nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DPressure)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                     1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                     nodeNumber,4,CMISS.BoundaryConditionsTypes.COUPLING_STRESS,value)
        #dependentField3D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PRESSURE_VALUES,
        #                                          1,1,nodeNumber,4,1.0)
# Outlet boundary elements
for element in range(numberOfOutletElements3D):
    elementNumber = outletElements3D[element]
    elementDomain=decomposition3D.ElementDomainGet(elementNumber)
    boundaryID = 3.0
    if (elementDomain == computationalNodeNumber):
        # Boundary ID: used to identify common faces for flowrate calculation
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,8,boundaryID)
        # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        elementNumber,9,CMISS.BoundaryConditionsTypes.COUPLING_STRESS)
                                                        #elementNumber,9,CMISS.BoundaryConditionsTypes.PRESSURE)
        # 1D Coupling node
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,11,couplingNode1D)
        # Boundary normal
        for component in range(numberOfDimensions):
            componentId = component + 5
            value = normalOutlet3D[component]
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,componentId,value)

# I n l e t
#--------------
# inlet boundary nodes p = f(t) - will be updated in pre-solve
value = 0.0
if abs(inletCoeff[3]) > 0.01:
    boundaryType = CMISS.BoundaryConditionsTypes.PRESSURE
else:
    boundaryType = CMISS.BoundaryConditionsTypes.FIXED_INLET

for nodeNumber in inletNodes3D:
    nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DVelocity)
    if (nodeDomain == computationalNodeNumber):
        for component in range(4):
            if abs(inletCoeff[component]) > 0.01:
                componentIdx = component + 1
                if (componentIdx < 4) and (nodeNumber in wallNodes3D):
                    # skip wall nodes if specifying velocity BC
                    break
                boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                             1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                             nodeNumber,componentIdx,boundaryType,value)
# Inlet boundary elements
for element in range(numberOfInletElements3D):
    elementNumber = inletElements3D[element]
    elementDomain=decomposition3D.ElementDomainGet(elementNumber)
    boundaryID = 2.0
    if (elementDomain == computationalNodeNumber):
        # Boundary ID
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                      elementNumber,8,boundaryID)
        # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        elementNumber,9,boundaryType)
        # Boundary normal
        for component in range(numberOfDimensions):
            componentId = component + 5
            value = normalInlet3D[component]
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                          elementNumber,componentId,value)
solverEquations3D.BoundaryConditionsCreateFinish()

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
boundaryConditions1DC = CMISS.BoundaryConditions()
solverEquations1DC.BoundaryConditionsCreateStart(boundaryConditions1DC)
# outlet boundary: nonreflecting
boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                              1,1,outletNode1D,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A0)
solverEquations1DC.BoundaryConditionsCreateFinish()

# Navier-Stokes
boundaryConditions1DNS = CMISS.BoundaryConditions()
solverEquations1DNS.BoundaryConditionsCreateStart(boundaryConditions1DNS)
# inlet boundary: coupling flow (Q)
boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                              1,1,couplingNode1D,1,CMISS.BoundaryConditionsTypes.COUPLING_FLOW,Q0)
# outlet boundary: nonreflecting
boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                              1,1,outletNode1D,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A0)
solverEquations1DNS.BoundaryConditionsCreateFinish()

if (checkTimestepStability):
    QMax = 430.0
    maxTimestep = Utilities1D.GetMaxStableTimestep(elementNodes,QMax,nodeCoordinates,H,E,A0,Rho)
    if (timeIncrement > maxTimestep):
        sys.exit('Timestep size '+str(timeIncrement)+' above maximum allowable size of '+str(maxTimestep)+'. Please reduce step size and re-run')

#=========================================================
# S O L V E
#=========================================================

# Solve the coupled problem
print("solving problem...")
# make a new output directory if necessary
outputDirectory = "./output/Dt" + str(round(timeIncrement,5)) + meshName + "/"
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   
# change to new directory and solve problem (note will return to original directory on exit)
with ChangeDirectory(outputDirectory):
    problem.Solve()
#========================================================

print("Finished. Time to solve (seconds): " + str(time.time()-startRunTime))    

CMISS.Finalise()
