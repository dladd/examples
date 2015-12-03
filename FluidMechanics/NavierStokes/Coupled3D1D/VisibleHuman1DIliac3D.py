#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an OpenCMISS script to solve Navier-Stokes flow through
#>  an 3D iliac bifurcation with descending arteries in 1D.
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

#> \example FluidMechanics/NavierStokes/Coupled3D1D/Python/IliacExample.py
## Python OpenCMISS script to solve Navier-Stokes flow through a 3D iliac bifurcation with descending arteries in 1D.
## \par Latest Builds:
#<


# Add Python bindings directory to PATH
import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy as np
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
# Setup
#==========================================================

if computationalNodeNumber == 0:
    print(""" Setting up the problem and solving

        1D Visible Human arterial tree with embedded 3D iliac bifurcation

            1D           3D          1D
                   |------------|----------
         ----------|       -----|    
                   |------------|----------
                                    
    """)

startRunTime = time.time()

#==========================================================
# P r o b l e m     C o n t r o l
#==========================================================

#-------------------------------------------
# General parameters
#-------------------------------------------
# Set the material parameters
density  = 1050.0     # Density     (kg/m3)
viscosity= 0.004      # Viscosity   (Pa.s)
G0   = 0.0            # Gravitational acceleration (m/s2)
Pext = 0.0            # External pressure (Pa)

# Material parameter scaling factors
Ls = 1000.0             # Length    (m -> mm)
Ts = 1000.0             # Time      (s -> ms)
Ms = 1000.0             # Mass      (kg -> g)

absoluteCouplingTolerance3D1D = 1.0E-3
relativeCouplingTolerance3D1D = 0.01

# Set the time parameters
numberOfPeriods = 1.0
timePeriod      = 790.0
timeIncrement   = 0.2
startTime       = 0.0
stopTime  = numberOfPeriods*timePeriod + 1.0E-6
outputFrequency = 5
dynamicSolverTheta = 1.0

#-------------------------------------------
# 3D parameters
#-------------------------------------------
numberOfElements3D = 540#3840 #540
quadraticMesh3D = True
outletArteries3D = ['LIA','RIA']
normalInlet3D = np.zeros((3))
normalOutlets3D = np.zeros((2,3))
if numberOfElements3D == 3840:
    couplingNodes3D = [31,4210,4468]
    inputDir3D = "./input/iliac3D/3840Elem/normal/"
    meshName3D = "iliac3840"
    normalInlet3D = [-0.0438947,-0.999036,-6.80781e-6]
    normalOutlets3D[0,:] = [-0.42607713,0.87736536,0.22065429]
    normalOutlets3D[1,:] = [ 0.43452153,0.8646826,0.25202191]
elif numberOfElements3D == 540:
    couplingNodes3D = [14,643,695]
    inputDir3D = "./input/iliac3D/540Elem/normal/"
    meshName3D = "iliac540"
    normalInlet3D = [-0.0438947,-0.999036,-6.80781e-6]
    normalOutlets3D[0,:] = [-0.39840638, 0.89000052, 0.22174634]#[-0.267634, 0.963521, -5.91711e-7]
    normalOutlets3D[1,:] = [ 0.42891156, 0.86122539,  0.27262741] # [0.396259, 0.918139, -6.62205e-6]
else:
    sys.exit('Unknown mesh specified')    

analyticInflow3D = False
inletValue3D = 0.0#500.0#0.001#1.0
inletCoeff3D = [-normalInlet3D[0],-normalInlet3D[1],-normalInlet3D[2],0.0]

#-------------------------------------------
# 1D parameters
#-------------------------------------------
Alpha = 1.0                         # Flow profile type
couplingNodes1D = [115,116,149]
couplingTolerance1D = 1.0E15
nodesUpper1D = range(1,115)
nodesUpper1D.append(182)
nodesUpper1D.append(183)
nodesLeft1D = range(116,148)
nodesRight1D = range(149,181)
nodesCouplingGroups1D = [nodesUpper1D,nodesLeft1D,nodesRight1D]
elementsUpper1D = range(1,57)
elementsUpper1D.append(90)
elementsLeft1D = range(74,89)
elementsRight1D = range(58,73)
elementsCouplingGroups1D = [elementsUpper1D,elementsLeft1D,elementsRight1D]

RCRBoundaries            = False   # Set to use coupled 0D Windkessel models (from CellML) at model outlet boundaries
nonReflecting            = True    # Set to use non-reflecting outlet boundaries
checkTimestepStability   = False   # Set to do a basic check of the stability of the hyperbolic problem based on the timestep size
initialiseFromFile       = False   # Set to initialise values
if(nonReflecting and RCRBoundaries):
    sys.exit('Please set either RCR or non-reflecting boundaries- not both.')


#==========================================================
# R e a d    M e s h    D a t a
#==========================================================
# -----------------------------------------------
#  3D: Get the mesh information from FieldML data
# -----------------------------------------------

if quadraticMesh3D:
    meshType = 'Quadratic'
else:
    meshType = 'Linear'
# Read xml file
fieldmlInput = inputDir3D + meshName3D + '.xml'
print('FieldML input file: ' + fieldmlInput)

#Wall boundary nodes
filename=inputDir3D + '/bc/wallNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfWallNodes3D=int(f.readline())
        wallNodes3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Wall boundary node file: ' + filename)

#Inlet boundary nodes
filename=inputDir3D + '/bc/inletNodes.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletNodes3D=int(f.readline())
        inletNodes3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary node file: ' + filename)
#Inlet boundary elements
filename=inputDir3D + '/bc/inletElements.dat'
try:
    with open(filename):
        f = open(filename,"r")
        numberOfInletElements3D=int(f.readline())
        inletElements3D=map(int,(re.split(',',f.read())))
        f.close()
except IOError:
   print ('Could not open Inlet boundary element file: ' + filename)

#Outlet boundary nodes
outletNodes3D = []
outletElements3D = []
branch = 0
for outletArtery in outletArteries3D:
    filename=inputDir3D + '/bc/' + outletArtery + '_Nodes.dat'
    try:
        with open(filename):
            f = open(filename,"r")
            numberOfOutletNodes3D=int(f.readline())
            nodes=map(int,(re.split(',',f.read())))
            outletNodes3D.append(nodes)
            f.close()
    except IOError:
       print ('Could not open Outlet boundary node file: ' + filename)
    #Outlet boundary elements
    filename=inputDir3D + '/bc/'+ outletArtery + '_Elements.dat'
    try:
        with open(filename):
            f = open(filename,"r")
            numberOfOutletElements3D=int(f.readline())
            elements=map(int,(re.split(',',f.read())))
            outletElements3D.append(elements)
            f.close()
    except IOError:
       print ('Could not open Outlet boundary element file: ' + filename)

# -----------------------------------------------
#  1D: Get the mesh from 1D csv files
# -----------------------------------------------
#  1 D   N o d e s
# ----------------------------
# Read nodes
inputNodeNumbers1D        = []
bifurcationNodeNumbers1D  = []
trifurcationNodeNumbers1D = []
terminalNodeNumbers1D      = []
arteryLabels1D            = []
filename = './input/visibleHuman1D/Node.csv'
numberOfNodes1D = Utilities1D.GetNumberOfNodes(filename)
nodeCoordinates1D = np.zeros([numberOfNodes1D,4,3])
Utilities1D.CsvNodeReader(filename,inputNodeNumbers1D,bifurcationNodeNumbers1D,trifurcationNodeNumbers1D,
                          terminalNodeNumbers1D,nodeCoordinates1D,arteryLabels1D)
numberOfInputNodes1D     = len(inputNodeNumbers1D)
numberOfBifurcations1D   = len(bifurcationNodeNumbers1D)
numberOfTrifurcations1D  = len(trifurcationNodeNumbers1D)
numberOfTerminalNodes1D  = len(terminalNodeNumbers1D)

#  1 D   E l e m  e n t s 
# ----------------------------
# Read elements
elementNodes1D = []
elementNodes1D.append([0,0,0])
bifurcationElements1D = (numberOfBifurcations1D+1)*[3*[0]]
trifurcationElements1D = (numberOfTrifurcations1D+1)*[4*[0]]
Utilities1D.CsvElementReader('input/visibleHuman1D/Element.csv',elementNodes1D,bifurcationElements1D,
                             trifurcationElements1D,numberOfBifurcations1D,numberOfTrifurcations1D)
numberOfElements1D = len(elementNodes1D)-1

Q  = np.zeros((numberOfNodes1D+1,4))
A  = np.zeros((numberOfNodes1D+1,4))
dQ = np.zeros((numberOfNodes1D+1,4))
dA = np.zeros((numberOfNodes1D+1,4))

#  1 D   M a t e r i a l s
# ----------------------------
# Initialise the node-based parameters
A0   = np.zeros((numberOfNodes1D+1,4))  # Area        (m2)
H    = np.zeros((numberOfNodes1D+1,4))  # Thickness   (m)
E    = np.zeros((numberOfNodes1D+1,4))  # Elasticity  (Pa)
# Read the MATERIAL csv file
Utilities1D.CsvMaterialReader('input/visibleHuman1D/Material.csv',A0,E,H)

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

for bifIdx in range(1,numberOfBifurcations1D+1):
    nodeIdx = bifurcationNodeNumbers1D[bifIdx-1]
    for versionIdx in range(1,3):
        A0[nodeIdx][versionIdx] = A0[elementNodes1D[bifurcationElements1D[bifIdx][versionIdx]][1]][0]
        E [nodeIdx][versionIdx] = E [elementNodes1D[bifurcationElements1D[bifIdx][versionIdx]][1]][0]
        H [nodeIdx][versionIdx] = H [elementNodes1D[bifurcationElements1D[bifIdx][versionIdx]][1]][0]
for trifIdx in range(1,numberOfTrifurcations1D+1):
    nodeIdx = trifurcationNodeNumbers1D[trifIdx-1]
    for versionIdx in range(1,4):
        A0[nodeIdx][versionIdx] = A0[elementNodes1D[trifurcationElements1D[trifIdx][versionIdx]][1]][0]
        E [nodeIdx][versionIdx] = E [elementNodes1D[trifurcationElements1D[trifIdx][versionIdx]][1]][0]
        H [nodeIdx][versionIdx] = H [elementNodes1D[trifurcationElements1D[trifIdx][versionIdx]][1]][0]

# Start with Q=0, A=A0 state
A = A0

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
 region1DDummyUserNumber,
 quadraticBasis1DUserNumber,
 mesh1DUserNumber,
 mesh1DDummyUserNumber,
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
 problemUserNumber) = range(1,30)

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
fieldmlInfo.InputCoordinateSystemCreateStart("ArteryMesh.coordinates",coordinateSystem3D,coordinateSystem3DUserNumber)
coordinateSystem3D.CreateFinish()
numberOfDimensions = coordinateSystem3D.DimensionGet()

# Create a region
region3D = CMISS.Region()
region3D.CreateStart(region3DUserNumber,CMISS.WorldRegion)
region3D.label = "3DIliac"
region3D.coordinateSystem = coordinateSystem3D
region3D.CreateFinish()

# Create nodes
nodes3D=CMISS.Nodes()
fieldmlInfo.InputNodesCreateStart("ArteryMesh.nodes.argument",region3D,nodes3D)
nodes3D.CreateFinish()
numberOfNodes3D = nodes3D.numberOfNodes
print("number of nodes 3D: " + str(numberOfNodes3D))

# Create bases
if (quadraticMesh3D):
    basisNumberQuadratic = 1
    gaussQuadrature = [3,3,3]
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.triquadratic_lagrange",basisNumberQuadratic)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberQuadratic,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberQuadratic,True)
    CMISS.Basis_CreateFinishNum(basisNumberQuadratic)
else:
    basisNumberLinear = 1
    gaussQuadrature = [2,2,2]
    fieldmlInfo.InputBasisCreateStartNum("ArteryMesh.trilinear_lagrange",basisNumberLinear)
    CMISS.Basis_QuadratureNumberOfGaussXiSetNum(basisNumberLinear,gaussQuadrature)
    CMISS.Basis_QuadratureLocalFaceGaussEvaluateSetNum(basisNumberLinear,True)
    CMISS.Basis_CreateFinishNum(basisNumberLinear)

# Create Mesh
meshComponent3DVelocity=1
meshComponent3DPressure=2
mesh = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("ArteryMesh.mesh.argument",mesh,mesh3DUserNumber,region3D)
mesh.NumberOfComponentsSet(2)

if (quadraticMesh3D):
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DVelocity,"ArteryMesh.template.triquadratic")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DPressure,"ArteryMesh.template.triquadratic")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DVelocity,"ArteryMesh.template.trilinear")
    fieldmlInfo.InputCreateMeshComponent(mesh,meshComponent3DPressure,"ArteryMesh.template.trilinear")

mesh.CreateFinish()
numberOfElements3D = mesh.numberOfElements
print("Number of 3D elements: " + str(numberOfElements3D))

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
                                  "ArteryMesh.coordinates")
geometricField3D.CreateFinish()
fieldmlInfo.InputFieldParametersUpdate(geometricField3D,"ArteryMesh.node.coordinates",
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
else:
    region1D = CMISS.Region()
    region1D.CreateStart(region1DUserNumber,CMISS.WorldRegion)
    region1D.label = "VisibleHuman1D"
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
# Note there is a hack here for setting up a dummy mesh so decomposition can have elements on 
#  each MPI process
if numberOfComputationalNodes > 1:
    nodes1D.CreateStart(region1D,numberOfNodes1D+numberOfComputationalNodes*2+1)
    print("Number of nodes 1D: "+str(numberOfNodes1D)+" + "+str(numberOfComputationalNodes*2+1)+" dummy nodes")
else:
    nodes1D.CreateStart(region1D,numberOfNodes1D)
    print("Number of nodes 1D: "+str(numberOfNodes1D))
nodes1D.CreateFinish()

# Create mesh
mesh1D = CMISS.Mesh()
mesh1D.CreateStart(mesh1DUserNumber,region1D,numberOfDimensions)
if numberOfComputationalNodes > 1:
    mesh1D.NumberOfElementsSet(numberOfElements1D+numberOfComputationalNodes)
    print("Number of elements 1D: "+str(numberOfElements1D)+" + "+ str(numberOfComputationalNodes) +" dummy elements")
else:
    mesh1D.NumberOfElementsSet(numberOfElements1D)
    print("Number of elements 1D: "+str(numberOfElements1D))
# Specify the mesh components
mesh1D.NumberOfComponentsSet(1)
# Specify the mesh components
meshElements = CMISS.MeshElements()
meshComponentNumber = 1
meshElements.CreateStart(mesh1D,meshComponentNumber,basis1D)

# Set up 1D nodes
for elemIdx in range(1,numberOfElements1D+1):
    meshElements.NodesSet(elemIdx,elementNodes1D[elemIdx])
for bifIdx in range(1,numberOfBifurcations1D+1):
    meshElements.LocalElementNodeVersionSet(int(bifurcationElements1D[bifIdx][0]),1,1,3)
    meshElements.LocalElementNodeVersionSet(int(bifurcationElements1D[bifIdx][1]),2,1,1) 
    meshElements.LocalElementNodeVersionSet(int(bifurcationElements1D[bifIdx][2]),3,1,1) 
for trifIdx in range(1,numberOfTrifurcations1D+1):
    meshElements.LocalElementNodeVersionSet(int(trifurcationElements1D[trifIdx][0]),1,1,3)
    meshElements.LocalElementNodeVersionSet(int(trifurcationElements1D[trifIdx][1]),2,1,1) 
    meshElements.LocalElementNodeVersionSet(int(trifurcationElements1D[trifIdx][2]),3,1,1) 
    meshElements.LocalElementNodeVersionSet(int(trifurcationElements1D[trifIdx][3]),4,1,1) 
# Set up dummy nodes
if numberOfComputationalNodes > 1:
    for cnode in range(numberOfComputationalNodes):
        nodeIdx = numberOfNodes1D + 2*cnode + 1
        elementNodes = []
        for elementNode in range(3):
            elementNodes.append(elementNode+nodeIdx)
        dummyElement=numberOfElements1D+cnode+1
        print("Dummy element "+ str(dummyElement)+ " dummy nodes: " + str(elementNodes))
        meshElements.NodesSet(dummyElement,elementNodes)

meshElements.CreateFinish()
mesh1D.CreateFinish() 

# Create a decomposition for the 1D mesh- attach 1D branches to 3D branches with the corresponding coupling node
decomposition1D = CMISS.Decomposition()
decomposition1D.CreateStart(decomposition1DUserNumber,mesh1D)
decomposition1D.type = CMISS.DecompositionTypes.USER_DEFINED
branch = 0
for nodeNumber3D in couplingNodes3D:
    nodeDomain = decomposition3D.NodeDomainGet(nodeNumber3D,meshComponentNumber)
    for element in elementsCouplingGroups1D[branch]:
        decomposition1D.ElementDomainSet(element,nodeDomain)
    branch+=1
# Dummy elements
if numberOfComputationalNodes > 1:
    for cnode in range(numberOfComputationalNodes):
        decomposition1D.ElementDomainSet(numberOfElements1D+cnode+1,cnode)
        print("Rank " + str(computationalNodeNumber) + " dummy element number " + str(numberOfElements1D+cnode+1)+" on "+str(cnode))

decomposition1D.numberOfDomains = numberOfComputationalNodes
decomposition1D.CreateFinish()

# Create a field for the geometry
geometricField1D = CMISS.Field()
geometricField1D.CreateStart(geometricField1DUserNumber,region1D)
geometricField1D.MeshDecompositionSet(decomposition1D)
geometricField1D.TypeSet(CMISS.FieldTypes.GEOMETRIC)
geometricField1D.VariableLabelSet(CMISS.FieldVariableTypes.U,"Geometry")
geometricField1D.fieldScalingType = CMISS.FieldScalingTypes.NONE
for componentIdx in range(1,4):
    geometricField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,componentIdx,1)
geometricField1D.CreateFinish()

# Set the geometric field values
for node in range(numberOfNodes1D):
    nodeNumber = node+1
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        for version in range(4):
            versionNumber = version + 1
            # If the version is undefined for this node (not a bi/trifurcation), continue to next node
            if (np.isnan(nodeCoordinates1D[node,version,0])):
                break
            else:
                for component in range(3):
                    componentNumber = component+1
                    geometricField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                              versionNumber,1,nodeNumber,componentNumber,
                                                              nodeCoordinates1D[node,version,component])

# Set dummy node positions
if numberOfComputationalNodes > 1:
    yLocation = 0.0
    lengthIncrement=(10.0*Ls)/(float(numberOfNodes1D)/2.-1.)
    A0_dummy = A0[0][0]
    E_dummy = E[0][0]
    H_dummy = H[0][0]
    for dummyNodeNumber in range(numberOfNodes1D+1,numberOfNodes1D+2+numberOfComputationalNodes*2):
        nodeDomain = decomposition1D.NodeDomainGet(dummyNodeNumber,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            value = dummyNodeNumber*lengthIncrement
            geometricField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
                                                      CMISS.FieldParameterSetTypes.VALUES,1,1,
                                                      dummyNodeNumber,1,value)                

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
                                                CMISS.FieldParameterSetTypes.VALUES,1,0.0)
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
dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.U,'Flow_Area')
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
for nodeIdx in range (1,numberOfNodes1D+1):
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nodeIdx in trifurcationNodeNumbers1D):
            versions = [1,2,3,4]
        elif (nodeIdx in bifurcationNodeNumbers1D):
            versions = [1,2,3]
        else:
            versions = [1]
        for versionIdx in versions:
            # U variables
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,Q[nodeIdx][versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,A[nodeIdx][versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,1,Q[nodeIdx][versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,2,A[nodeIdx][versionIdx-1])
            # delUdelN variables
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,dQ[nodeIdx][versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,dA[nodeIdx][versionIdx-1])

versionIdx=1
# Update dummy values
if numberOfComputationalNodes > 1:
    for nodeIdx in range (numberOfNodes1D+1,numberOfNodes1D+2+numberOfComputationalNodes*2):
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,0.0)
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,A0_dummy)
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,1,0.0)
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,2,A0_dummy)
            # delUdelN variables
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,0.0)
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,0.0)

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
if analyticInflow3D:
    # yOffset + amplitude*sin(frequency*time + phaseShift))))
    # Set the time it takes to ramp to max value
    rampPeriod = 0.25
    frequency = math.pi/(rampPeriod)
    inflowAmplitude = 0.5*inletValue3D
    yOffset = 0.5*inletValue3D
    phaseShift = -math.pi/2.0 
    startSine = 0.0
    stopSine = stopTime#rampPeriod

    analyticField = CMISS.Field()
    equationsSet3D.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SINUSOID,analyticField3DUserNumber,analyticField)
    equationsSet3D.AnalyticCreateFinish()
    analyticParameters = [inletCoeff3D[0],inletCoeff3D[1],inletCoeff3D[2],inletCoeff3D[3],
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
# Initialise the materials field variables (A0,E,H)
bifIdx = 0
trifIdx = 0
for nodeIdx in range(1,numberOfNodes1D+1):
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nodeIdx in trifurcationNodeNumbers1D):
            versions = [1,2,3,4]
        elif (nodeIdx in bifurcationNodeNumbers1D):
            versions = [1,2,3]
        else:
            versions = [1]
        for versionIdx in versions:
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,A0[nodeIdx][versionIdx-1])
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,E[nodeIdx][versionIdx-1])
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,3,H[nodeIdx][versionIdx-1])

# dummy materials values
if numberOfComputationalNodes > 1:
    versionIdx = 1
    for nodeIdx in range (numberOfNodes1D+1,numberOfNodes1D+2+numberOfComputationalNodes*2):
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,A0_dummy)
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,E_dummy)
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,3,H_dummy)

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

# Set the normal wave direction for bifurcations
for nodeIdx in bifurcationNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        # Incoming(parent)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    1,1,nodeIdx,1,1.0)
        # Outgoing(branches)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    2,1,nodeIdx,2,-1.0)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    3,1,nodeIdx,2,-1.0)
# Set the normal wave direction for trifurcations
for nodeIdx in trifurcationNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        # Incoming(parent)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    1,1,nodeIdx,1,1.0)
        # Outgoing(branches)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    2,1,nodeIdx,2,-1.0)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    3,1,nodeIdx,2,-1.0)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    4,1,nodeIdx,2,-1.0)

# Set the normal wave direction for inlets, outlets, and coupling nodes
versionIdx = 1
branch = 0
# Coupling nodes
for nodeIdx in couplingNodes1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if branch == 0:
            normalWaveValue = 1.0
            component = 1
        else:
            normalWaveValue = -1.0
            component = 2
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                    versionIdx,1,nodeIdx,component,normalWaveValue)
    branch+=1

# Outlets
versionIdx = 1
# Outgoing (parent) only - reflected wave will be 0
for terminalIdx in terminalNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(terminalIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                    versionIdx,1,terminalIdx,1,1.0)

# Finish the parameter update
independentField1D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
independentField1D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

#========================================================
# Analytic Field
#========================================================

analyticField1D = CMISS.Field()
equationsSet1DNS.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SPLINT_FROM_FILE,analyticField1DUserNumber,
                                     analyticField1D)
analyticField1D.VariableLabelSet(CMISS.FieldVariableTypes.U,'SplineInterpolatedInlet')
equationsSet1DNS.AnalyticCreateFinish()

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
iterative3D1DLoop.AbsoluteToleranceSet(absoluteCouplingTolerance3D1D)
iterative3D1DLoop.RelativeToleranceSet(relativeCouplingTolerance3D1D)

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
nonlinearSolver1DC.newtonAbsoluteTolerance = 1.0E-8
nonlinearSolver1DC.newtonSolutionTolerance = 1.0E-5
nonlinearSolver1DC.newtonRelativeTolerance = 1.0E-5
# Get the nonlinear linear solver
linearSolver1DC = CMISS.Solver()
nonlinearSolver1DC.NewtonLinearSolverGet(linearSolver1DC)
linearSolver1DC.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
linearSolver1DC.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
linearSolver1DC.LinearIterativeMaximumIterationsSet(100000)
linearSolver1DC.LinearIterativeDivergenceToleranceSet(1.0E+10)
linearSolver1DC.LinearIterativeRelativeToleranceSet(1.0E-5)
linearSolver1DC.LinearIterativeAbsoluteToleranceSet(1.0E-8)
linearSolver1DC.LinearIterativeGMRESRestartSet(3000)

# Navier-Stokes
dynamicSolver1DNS = CMISS.Solver()
problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],2,dynamicSolver1DNS)
dynamicSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
dynamicSolver1DNS.dynamicTheta = [dynamicSolverTheta]
# Get the dynamic nonlinear solver
nonlinearSolver1DNS = CMISS.Solver()
dynamicSolver1DNS.DynamicNonlinearSolverGet(nonlinearSolver1DNS)
# Set the nonlinear Jacobian type
nonlinearSolver1DNS.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS)
nonlinearSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
nonlinearSolver1DNS.NewtonAbsoluteToleranceSet(1.0E-8)
nonlinearSolver1DNS.NewtonSolutionToleranceSet(1.0E-8)
nonlinearSolver1DNS.NewtonRelativeToleranceSet(1.0E-5)
# Get the dynamic nonlinear linear solver
linearSolver1DNS = CMISS.Solver()
nonlinearSolver1DNS.NewtonLinearSolverGet(linearSolver1DNS)
linearSolver1DNS.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
linearSolver1DNS.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
linearSolver1DNS.LinearIterativeMaximumIterationsSet(100000)
linearSolver1DNS.LinearIterativeDivergenceToleranceSet(1.0E+10)
linearSolver1DNS.LinearIterativeRelativeToleranceSet(1.0E-5)
linearSolver1DNS.LinearIterativeAbsoluteToleranceSet(1.0E-8)
linearSolver1DNS.LinearIterativeGMRESRestartSet(3000)

# -----------------------------------------------
#  3D
# -----------------------------------------------
# Create problem solver
dynamicSolver3D = CMISS.Solver()
problem.SolverGet([1,2,CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver3D)
dynamicSolver3D.outputType = CMISS.SolverOutputTypes.NONE
dynamicSolver3D.dynamicTheta = [dynamicSolverTheta]
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
# O u t l e t s
#--------------
boundaryType = CMISS.BoundaryConditionsTypes.COUPLING_STRESS
# Outlet boundary nodes: Coupling stress (traction)
value=0.0
for branch in range(2):
    for nodeNumber in outletNodes3D[branch]:
        nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DPressure)
        if (nodeDomain == computationalNodeNumber):
            boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                         1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                         nodeNumber,4,boundaryType,value)
    # Outlet boundary elements
    for elementNumber in outletElements3D[branch]:
        elementDomain=decomposition3D.ElementDomainGet(elementNumber)
        if (elementDomain == computationalNodeNumber):
            boundaryID = 3.0 + float(branch)
            # Boundary ID: used to identify common faces for flowrate calculation
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,8,boundaryID)
            # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,9,boundaryType)
            # 1D Coupling node
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,11,couplingNodes1D[branch])
            # Boundary normal
            for component in range(numberOfDimensions):
                componentId = component + 5
                value = normalOutlets3D[branch,component]
                equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                elementNumber,componentId,value)

# I n l e t
#--------------
boundaryType = CMISS.BoundaryConditionsTypes.COUPLING_STRESS
for nodeNumber in inletNodes3D:
    nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DVelocity)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                     1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                     nodeNumber,4,boundaryType,value)
# Inlet boundary elements
for elementNumber in inletElements3D:
    elementDomain=decomposition3D.ElementDomainGet(elementNumber)
    boundaryID = 2.0
    if (elementDomain == computationalNodeNumber):
        # Boundary ID
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        elementNumber,8,boundaryID)
        # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        elementNumber,9,boundaryType)
        # 1D Coupling node
        equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        elementNumber,11,couplingNodes1D[0])
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
versionIdx=1
for nodeNumber in terminalNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,
                                      A[nodeNumber][0])

# Set dummy nodes to fixed reference state
if numberOfComputationalNodes > 1:
    dummyNodeNumber = numberOfNodes1D+1
    nodeDomain = decomposition1D.NodeDomainGet(dummyNodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        print("Setting dummy Q BC for characteristic boundary node " + str(dummyNodeNumber))
        boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      1,1,dummyNodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,0.0)
    dummyNodeNumber = numberOfNodes1D+numberOfComputationalNodes*2+1
    nodeDomain = decomposition1D.NodeDomainGet(dummyNodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        print("Setting dummy A BC for characteristic boundary node " + str(dummyNodeNumber))
        boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      1,1,dummyNodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,A0_dummy)
solverEquations1DC.BoundaryConditionsCreateFinish()

# Navier-Stokes
boundaryConditions1DNS = CMISS.BoundaryConditions()
solverEquations1DNS.BoundaryConditionsCreateStart(boundaryConditions1DNS)

# Inlet (Flow)
versionIdx = 1
for nodeNumber in inputNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                       versionIdx,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED_FITTED,
                                       Q[nodeIdx][0])

# Outlet boundary: nonreflecting (Area)
for nodeIdx in terminalNodeNumbers1D:
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                       1,1,nodeIdx,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,
                                       A0[nodeIdx][0])

# Coupled 3D-to-1D boundary: coupling flow (Q)
for nodeIdx in couplingNodes1D[1:3]:
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                       1,1,nodeIdx,1,CMISS.BoundaryConditionsTypes.COUPLING_FLOW,
                                       Q[nodeIdx][0])
# Coupled 1D-to-3D boundary: coupling stress (A)
nodeIdx = couplingNodes1D[0]
nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
if (nodeDomain == computationalNodeNumber):
    boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                   1,1,nodeIdx,1,CMISS.BoundaryConditionsTypes.COUPLING_STRESS,
                                   A0[nodeIdx][0])

# Set dummy nodes to fixed
if numberOfComputationalNodes > 1:
    dummyNodeNumber = numberOfNodes1D+1
    nodeDomain = decomposition1D.NodeDomainGet(dummyNodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        print("Setting dummy Q BC for 1D Navier-Stokes boundary node " + str(dummyNodeNumber))
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      1,1,dummyNodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,0.0)
    dummyNodeNumber = numberOfNodes1D+numberOfComputationalNodes*2+1
    nodeDomain = decomposition1D.NodeDomainGet(dummyNodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        print("Setting dummy A BC for 1D Navier-Stokes boundary node " + str(dummyNodeNumber))
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                       1,1,dummyNodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,A0_dummy)

solverEquations1DNS.BoundaryConditionsCreateFinish()

if (checkTimestepStability):
    QMax = 430.0
    maxTimestep = Utilities1D.GetMaxStableTimestep(elementNodes,QMax,nodeCoordinates,H,E,A0,Rho)
    if (timeIncrement > maxTimestep):
        sys.exit('Timestep size '+str(timeIncrement)+' above maximum allowable size of '+str(maxTimestep)+'. Please reduce step size and re-run')

#=========================================================
# S O L V E
#=========================================================

# Export Geometry
print("Exporting geometry...")
Fields = CMISS.Fields()
Fields.CreateRegion(region1D)
Fields.NodesExport("Geometry","FORTRAN")
Fields.ElementsExport("Geometry","FORTRAN")
Fields.Finalise()
print("done!")

# Solve the coupled problem
print("solving problem...")
# make a new output directory if necessary
outputDirectory = "./output/VisibleHuman_Dt" + str(round(timeIncrement/Ts,5)) + meshName3D + "/"
print('output in directory: '+outputDirectory)
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
