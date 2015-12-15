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

import numpy
import gzip
import time
import re
import math
import contextlib
import FluidExamples1DUtilities as Utilities1D
import scipy
import bisect
from scipy import interpolate
from numpy import linalg
import fittingUtils as fit

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
viscosity= 0.0035      # Viscosity   (Pa.s)
G0   = 0.0            # Gravitational acceleration (m/s2)
Pext = 0.0            # External pressure (Pa)

# Material parameter scaling factors
Ls = 1000.0             # Length    (m -> mm)
Ts = 1000.0                # Time      (s -> ms)
Ms = 1000.0                # Mass      (kg -> g)

# Set the time parameters
numberOfPeriods = 0.1#4.0
timePeriod      = 1020.0
timeIncrement   = 0.5
startTime       = 0.0
stopTime  = numberOfPeriods*timePeriod + timeIncrement*0.01 
outputFrequency = 1
dynamicSolverNavierStokesTheta = [1.0]
#couplingTolerance3D1D = 1.0E-4
couplingTolerance1D0D = 1.0E-4
flowTolerance3D1D = 1.0E-4
stressTolerance3D1D = 1.0E-4
relativeTolerance3D1D = 0.001

#-------------------------------------------
# Fit parameters
#-------------------------------------------
fitData = False
initialiseVelocity = False
pcvDataDir = './input/pcvData/'
pcvTimeIncrement = 40.8
velocityScaleFactor = 1.0/Ts
numberOfPcvDataFiles = 25
dataPointResolution = [2.08,2.08,2.29]
startFit = 0
addZeroLayer = False
p = 2.5
vicinityFactor = 1.1
interpolatedDir = './output/interpolatedData/'
try:
    os.makedirs(interpolatedDir)
except OSError, e:
    if e.errno != 17:
        raise   
print('interpolated data output to : ' + interpolatedDir)

#-------------------------------------------
# 3D parameters
#-------------------------------------------
couplingNodes3D = [643,695]
quadraticMesh = False
#inputDir3D = "./input/iliac3D/540Elem/normal/"
inputDir3D = "./input/iliac3D/540Elem/stenosis/"
meshName = "iliac540"
outletArteries = ['LIA','RIA']
analyticInflow = True
#inletValue = 1000.0#0.001#1.0
inletValue = 0.5
normalInlet3D = numpy.zeros((3))
normalOutlets3D = numpy.zeros((2,3))
normalInlet3D = [-0.0438947,-0.999036,-6.80781e-6]
normalOutlets3D[0,:] = [-0.39840586,  0.89000103,  0.2217452]
normalOutlets3D[1,:] = [.42891316,  0.8612245 ,  0.27262769]
#normalOutlets3D[0,:] = [-0.267634, 0.963521, -5.91711e-7]
#normalOutlets3D[1,:] = [0.396259, 0.918139, -6.62205e-6]
#inletCoeff = [0.0,1.0,0.0,0.0]
inletCoeff = [-normalInlet3D[0],-normalInlet3D[1],-normalInlet3D[2],0.0]

#-------------------------------------------
# 1D parameters
#-------------------------------------------
RCRBoundaries            = True   # Set to use coupled 0D Windkessel models (from CellML) at model outlet boundaries
nonReflecting            = False    # Set to use non-reflecting outlet boundaries
if(nonReflecting and RCRBoundaries):
    sys.exit('Please set either RCR or non-reflecting boundaries- not both.')

inputDir1D = "./input/Reymond1DIliacs/"
couplingNodes1D = [1,2]
checkTimestepStability = False
couplingTolerance1D = 1.0E15
inletLocations1D = numpy.zeros((2,3))
inletLocations1D[0,:] = [319.02,286.946,423.482]
inletLocations1D[1,:] = [361.766,291.798,428.677]
elementsLeft1D = range(1,8)
elementsRight1D = range(8,15)
elementsCouplingGroups1D = [elementsLeft1D,elementsRight1D]


# make a new output directory if necessary
outputDirectory = "./output/"#ReymondIliac_Dt" + str(round(timeIncrement,5)) + meshName + "/"
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   

#================================================================================================================================
#  1 D  M e s h   
#================================================================================================================================
ProgressDiagnostics=True
if (ProgressDiagnostics):
    print " == >> Reading geometry from files... << == "

# Read nodes
inputNodeNumbers1D = []
branchNodeNumbers1D = []
terminalNodeNumbers1D = []
nodeCoordinates1D = []
branchNodeElements1D = []
terminalArteryNames1D = []
RCRParameters = []
filename=inputDir1D+'Reymond2009_Leg_Nodes.csv'
Utilities1D.CsvNodeReader2(filename,inputNodeNumbers1D,branchNodeNumbers1D,terminalNodeNumbers1D,
                           nodeCoordinates1D,branchNodeElements1D,terminalArteryNames1D,RCRParameters)
numberOfInputNodes1D     = len(inputNodeNumbers1D)
numberOfNodes1D          = len(nodeCoordinates1D)
numberOfBranches1D       = len(branchNodeNumbers1D)
numberOfTerminalNodes1D  = len(terminalNodeNumbers1D)

# Read elements
elementNodes1D = []
elementArteryNames1D = []
elementNodes1D.append([0,0,0])
filename=inputDir1D+'Reymond2009_Leg_Elements.csv'
Utilities1D.CsvElementReader2(filename,elementNodes1D,elementArteryNames1D)
numberOfElements1D = len(elementNodes1D)-1
        
if (ProgressDiagnostics):
    print " Number of nodes: " + str(numberOfNodes1D)
    print " Number of elements: " + str(numberOfElements1D)
    print " Input at nodes: " + str(inputNodeNumbers1D)
    print " Branches at nodes: " + str(branchNodeNumbers1D)
    print " Terminal at nodes: " + str(terminalNodeNumbers1D)
    print " == >> Finished reading geometry... << == "

Q0 = 0.0
dQ = 0.0
dA = 0.0

# Constant materials parameters
Alpha = 4.0/3.0                    # Flow profile type
# calculate specific scale factors
Qs    = (Ls**3.0)/Ts     # Flow             (m3/s)  
As    = Ls**2.0          # Area             (m2)
Hs    = Ls               # vessel thickness (m)
Es    = Ms/(Ls*Ts**2.0)  # Elasticity Pa    (kg/(ms2)
Rhos  = Ms/(Ls**3.0)     # Density          (kg/m3)
Mus   = Ms/(Ls*Ts)       # Viscosity        (kg/(ms))
Ps    = Ms/(Ls*Ts**2.0)  # Pressure         (kg/(ms2))
Gs    = Ls/(Ts**2.0)     # Acceleration    (m/s2)
# Initialise the node-based parameters
A0   = []
H    = []
E    = []
# Read the MATERIAL csv file
filename = inputDir1D+'Reymond2009_Leg_Materials.csv'
print('Reading materials from: '+filename)
Utilities1D.CsvMaterialReader2(filename,A0,E,H)
for i in range(len(A0[:])):
    for j in range(len(A0[i][:])):
        A0[i][j] = A0[i][j]*As
        E[i][j] = E[i][j]*Es
        H[i][j] = H[i][j]*Hs

# Initial conditions
# Zero reference state: Q=0, A=A0 state
Q  = numpy.zeros((numberOfNodes1D,4))
A  = numpy.zeros((numberOfNodes1D,4))
dQ = numpy.zeros((numberOfNodes1D,4))
dA = numpy.zeros((numberOfNodes1D,4))
for i in range(len(A0[:])):
    for j in range(len(A0[i][:])):
            A[i,j] = A0[i][j]

# Apply scale factors        
density = density*Rhos
viscosity  = viscosity*Mus

#==========================================================
# Setup
#==========================================================

if computationalNodeNumber == 0:
    print(""" Setting up the problem and solving


        Coupled flow from a 3D iliac bifurcation to 1D-0D iliacs

                                    
        -------------------------------| 
        >                              |---------------|/\//\/\\
        ->                   ----------|          
        --> u(r,t)          |           
        ->                   ----------|          
        >                              |---------------|/\/\/\/\/
        -------------------------------| 
                                    
    """)

startRunTime = time.time()

# -----------------------------------------------
#  3D: Get the mesh information from FieldML data
# -----------------------------------------------
# Read xml file
if quadraticMesh:
    meshType = 'Quadratic'
    fieldmlInput = inputDir3D + meshName + '.xml'
else:
    meshType = 'Linear'
    fieldmlInput = inputDir3D + meshName + 'Linear.xml'

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
for outletArtery in outletArteries:
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
print(outletNodes3D)
print(outletElements3D)

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
 independentField3DUserNumber,
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
 CellMLUserNumber,
 CellMLModelsFieldUserNumber,
 CellMLStateFieldUserNumber,
 CellMLIntermediateFieldUserNumber,
 CellMLParametersFieldUserNumber,
 problemUserNumber) = range(1,34)

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
if (quadraticMesh):
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
mesh3D = CMISS.Mesh()
fieldmlInfo.InputMeshCreateStart("ArteryMesh.mesh.argument",mesh3D,mesh3DUserNumber,region3D)
mesh3D.NumberOfComponentsSet(2)

if (quadraticMesh):
    fieldmlInfo.InputCreateMeshComponent(mesh3D,meshComponent3DVelocity,"ArteryMesh.template.triquadratic")
    fieldmlInfo.InputCreateMeshComponent(mesh3D,meshComponent3DPressure,"ArteryMesh.template.triquadratic")
else:
    fieldmlInfo.InputCreateMeshComponent(mesh3D,meshComponent3DVelocity,"ArteryMesh.template.trilinear")
    fieldmlInfo.InputCreateMeshComponent(mesh3D,meshComponent3DPressure,"ArteryMesh.template.trilinear")

mesh3D.CreateFinish()
numberOfElements3D = mesh3D.numberOfElements
print("number of elements 3D: " + str(numberOfElements3D))

# Create a decomposition for the mesh
decomposition3D = CMISS.Decomposition()
decomposition3D.CreateStart(decomposition3DUserNumber,mesh3D)
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
    region1D.label = "1D_LegArteries"
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
#nodes1D.CreateStart(region1D,numberOfNodes1D)
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
#mesh1D.NumberOfElementsSet(numberOfElements1D)
# Specify the mesh components
mesh1D.NumberOfComponentsSet(1)
meshComponentNumber = 1
# Specify the  mesh component
meshElements1D = CMISS.MeshElements()
meshElements1D.CreateStart(mesh1D,meshComponentNumber,basis1D)
for elemIdx in range(1,numberOfElements1D+1):
    meshElements1D.NodesSet(elemIdx,elementNodes1D[elemIdx])
for nodeIdx in range(len(branchNodeNumbers1D)):
    versionIdx = 1
    for element in branchNodeElements1D[nodeIdx]:
        if versionIdx == 1:
            meshElements1D.LocalElementNodeVersionSet(element,versionIdx,1,3)            
        else:
            meshElements1D.LocalElementNodeVersionSet(element,versionIdx,1,1)            
        versionIdx+=1
# Set up dummy nodes
if numberOfComputationalNodes > 1:
    for cnode in range(numberOfComputationalNodes):
        nodeIdx = numberOfNodes1D + 2*cnode + 1
        elementNodes = []
        for elementNode in range(3):
            elementNodes.append(elementNode+nodeIdx)
        dummyElement=numberOfElements1D+cnode+1
        print("Dummy element "+ str(dummyElement)+ " dummy nodes: " + str(elementNodes))
        meshElements1D.NodesSet(dummyElement,elementNodes)

meshElements1D.CreateFinish()
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
geometricField1D.VariableLabelSet(CMISS.FieldVariableTypes.U,"Geometry1D")
geometricField1D.fieldScalingType = CMISS.FieldScalingTypes.NONE
geometricField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
for componentIdx in range(1,4):
    geometricField1D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,componentIdx,1)
geometricField1D.CreateFinish()

# Set the geometric field values
for node in range(numberOfNodes1D):
    nodeNumber = node+1
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeNumber in branchNodeNumbers1D:
            branchNodeIdx = branchNodeNumbers1D.index(nodeNumber)
            numberOfVersions = len(branchNodeElements1D[branchNodeIdx])
        for component in range(3):
            componentNumber = component+1
            for versionIdx in range(1,numberOfVersions+1):
                geometricField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                          versionIdx,1,nodeNumber,componentNumber,
                                                          nodeCoordinates1D[node][component])
        # for version in range(4):
        #     versionNumber = version + 1
        #     # If the version is undefined for this node (not a bi/trifurcation), continue to next node
        #     if (numpy.isnan(nodeCoordinates1D[node,version,0])):
        #         break
        #     else:
        #         for component in range(3):
        #             componentNumber = component+1
        #             geometricField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
        #                                                       versionNumber,1,nodeNumber,componentNumber,
        #                                                       nodeCoordinates1D[node,version,component])

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
# Set max CFL number (default 0.0- do not calc)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,2,0.0)
# Set time increment (default 0.0)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,3,timeIncrement)
# Set stabilisation type (default 1.0 = RBS)
equationsSetField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U1,
                                                CMISS.FieldParameterSetTypes.VALUES,4,1.0)

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Navier-Stokes solver
if(RCRBoundaries):
    EquationsSetSubtype1D = CMISS.EquationsSetSubtypes.COUPLED1D0D_NAVIER_STOKES
    ProblemSubtype1D = CMISS.ProblemSubTypes.COUPLED1D0D_NAVIER_STOKES
else:
    EquationsSetSubtype1D = CMISS.EquationsSetSubtypes.TRANSIENT1D_NAVIER_STOKES
    ProblemSubtype1D = CMISS.ProblemSubTypes.TRANSIENT1D_NAVIER_STOKES
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
                             EquationsSetSubtype1D,
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
if (RCRBoundaries):
    dependentField1DNS.VariableLabelSet(CMISS.FieldVariableTypes.U1,'CellML Q and P')
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
# qCellML & pCellml
if (RCRBoundaries):
    dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,1,meshComponentNumber)
    dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,2,meshComponentNumber)
# Pressure
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,1,meshComponentNumber)
dependentField1DNS.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,2,meshComponentNumber)
equationsSet1DC.DependentCreateFinish()

# Navier-Stokes
equationsSet1DNS.DependentCreateStart(dependentField1DUserNumber,dependentField1DNS)
equationsSet1DNS.DependentCreateFinish()
dependentField1DNS.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES)

# Initialise the dependent field variables
# Initialise the dependent field variables
for nodeIdx in range (1,numberOfNodes1D+1):
    nodeDomain =  decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeIdx in branchNodeNumbers1D:
            branchNodeIdx = branchNodeNumbers1D.index(nodeIdx)
            numberOfVersions = len(branchNodeElements1D[branchNodeIdx])    
        for versionIdx in range(1,numberOfVersions+1):
            # U variables
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,Q[nodeIdx-1,versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,A[nodeIdx-1,versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,1,Q[nodeIdx-1,versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                        versionIdx,1,nodeIdx,2,A[nodeIdx-1,versionIdx-1])
            # delUdelN variables
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,dQ[nodeIdx-1,versionIdx-1])
            dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,dA[nodeIdx-1,versionIdx-1])

versionIdx = 1
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

# for nodeIdx in range (1,numberOfNodes1D+1):
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
#                                                 versionIdx,1,nodeIdx,1,Q0)
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
#                                                 versionIdx,1,nodeIdx,2,A0)
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
#                                                 versionIdx,1,nodeIdx,1,Q0)
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
#                                                 versionIdx,1,nodeIdx,2,A0)
#     # delUdelN variables
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
#                                                 versionIdx,1,nodeIdx,1,dQ)
#     dependentField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
#                                                 versionIdx,1,nodeIdx,2,dA)
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
    # Set the time it takes to ramp to max value
    rampPeriod = 10.0
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
# Initialise the materials field variables (A0,E,H)
for node in range(numberOfNodes1D):
    nodeIdx = node+1
    nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeIdx in branchNodeNumbers1D:
            branchNodeIdx = branchNodeNumbers1D.index(nodeIdx)
            numberOfVersions = len(branchNodeElements1D[branchNodeIdx])            
        for versionIdx in range(1,numberOfVersions+1):
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,A0[node][versionIdx-1])
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,2,E[node][versionIdx-1])
            materialsField1DNS.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,3,H[node][versionIdx-1])
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
#  3D
# -----------------------------------------------
# Create independent field for Navier Stokes - will hold fitted values on the NSE side
independentField3D = CMISS.Field()
independentField3D.CreateStart(independentField3DUserNumber,region3D)
independentField3D.LabelSet("Fitted data")
independentField3D.TypeSet(CMISS.FieldTypes.GENERAL)
independentField3D.MeshDecompositionSet(decomposition3D)
independentField3D.GeometricFieldSet(geometricField3D)
independentField3D.DependentTypeSet(CMISS.FieldDependentTypes.INDEPENDENT)
independentField3D.NumberOfVariablesSet(1)
# PCV values field
independentField3D.VariableTypesSet([CMISS.FieldVariableTypes.U])
independentField3D.DimensionSet(CMISS.FieldVariableTypes.U,CMISS.FieldDimensionTypes.VECTOR)
independentField3D.NumberOfComponentsSet(CMISS.FieldVariableTypes.U,numberOfDimensions)
independentField3D.VariableLabelSet(CMISS.FieldVariableTypes.U,"FittedData")
for dimension in range(numberOfDimensions):
    dimensionId = dimension + 1
    independentField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,dimensionId,1)
independentField3D.CreateFinish()
equationsSet3D.IndependentCreateStart(independentField3DUserNumber,independentField3D)
equationsSet3D.IndependentCreateFinish()
# Initialise data point vector field to 0
independentField3D.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

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
# Set the normal wave direction for branching nodes
for branch in range(len(branchNodeNumbers1D)):
    branchNode = branchNodeNumbers1D[branch]
    nodeDomain = decomposition1D.NodeDomainGet(branchNode,meshComponentNumber)    
    if (nodeDomain == computationalNodeNumber):
        # Incoming(parent)
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                    1,1,branchNode,1,1.0)
        for daughterVersion in range(2,len(branchNodeElements1D[branch])+1):
            # Outgoing(branches)
            independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                        daughterVersion,1,branchNode,2,-1.0)
# Set the normal wave direction for terminal
if (RCRBoundaries or nonReflecting):
    for terminalIdx in range (1,numberOfTerminalNodes1D+1):
        nodeIdx = terminalNodeNumbers1D[terminalIdx-1]
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            # Incoming (parent) - outgoing component to come from 0D
            versionIdx = 1
            independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeIdx,1,1.0)

for inletNode in couplingNodes1D:
    nodeDomain = decomposition1D.NodeDomainGet(inletNode,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        # Outgoing - incoming component to come from 0D
        versionIdx = 1
        independentField1D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                    versionIdx,1,inletNode,2,-1.0)
# Finish the parameter update
independentField1D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
independentField1D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

#DOC-START cellml define field maps
#================================================================================================================================
#  RCR CellML Model Maps
#================================================================================================================================


if (RCRBoundaries):

    if (ProgressDiagnostics):
        print " == >> CellML Models << == "
    #----------------------------------------------------------------------------------------------------------------------------
    # Description
    #----------------------------------------------------------------------------------------------------------------------------
    # A CellML OD model is used to provide the impedance from the downstream vascular bed beyond the termination
    # point of the 1D model. This is iteratively coupled with the the 1D solver. In the case of a simple resistance
    # model, P=RQ, which is analogous to Ohm's law: V=IR. A variable map copies the guess for the FlowRate, Q at 
    # the boundary from the OpenCMISS Dependent Field to the CellML equation, which then returns presssure, P.
    # The initial guess value for Q is taken from the previous time step or is 0 for t=0. In OpenCMISS this P value is 
    # then used to compute a new Area value based on the P-A relationship and the Riemann variable W_2, which gives a
    # new value for Q until the values for Q and P converge within tolerance of the previous value.
    #----------------------------------------------------------------------------------------------------------------------------

    # Create CellML models with specified RCR values
    terminalPressure = 0.0
    modelDirectory = inputDir1D+'CellMLModels/terminalArteryRCR/'
    Utilities1D.WriteCellMLRCRModels(terminalArteryNames1D,RCRParameters,terminalPressure,modelDirectory)    

    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,region1D)
    # Number of CellML models
    CellMLModelIndex = [0]*(numberOfTerminalNodes1D+1)

    # Windkessel Model
    localmodels = 0
    for terminalIdx in range (1,numberOfTerminalNodes1D+1):
        nodeIdx = terminalNodeNumbers1D[terminalIdx-1]
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
        modelFile = modelDirectory + terminalArteryNames1D[terminalIdx-1] + '.cellml'
        print('reading model: ' + modelFile)
#        if (nodeDomain == computationalNodeNumber):
        CellMLModelIndex[terminalIdx] = CellML.ModelImport(modelFile)
        # known (to OpenCMISS) variables
        CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Circuit/Qin")
        # to get from the CellML side 
        CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Circuit/Pout")

    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    for terminalIdx in range (1,numberOfTerminalNodes1D+1):
        nodeIdx = terminalNodeNumbers1D[terminalIdx-1]
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
#        if (nodeDomain == computationalNodeNumber):
        # Now we can set up the field variable component <--> CellML model variable mappings.
        # Map the OpenCMISS boundary flow rate values --> CellML
        # Q is component 1 of the DependentField
        CellML.CreateFieldToCellMLMap(dependentField1DNS,CMISS.FieldVariableTypes.U,1,
                                      CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[terminalIdx],
                                      "Circuit/Qin",CMISS.FieldParameterSetTypes.VALUES)
        # Map the returned pressure values from CellML --> CMISS
        # pCellML is component 1 of the Dependent field U1 variable
        CellML.CreateCellMLToFieldMap(CellMLModelIndex[terminalIdx],"Circuit/Pout",CMISS.FieldParameterSetTypes.VALUES,
                                      dependentField1DNS,CMISS.FieldVariableTypes.U1,pCellMLComponent,CMISS.FieldParameterSetTypes.VALUES)

    # Finish the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateFinish()

    CellMLModelsField = CMISS.Field()
    CellML.ModelsFieldCreateStart(CellMLModelsFieldUserNumber,CellMLModelsField)
    CellML.ModelsFieldCreateFinish()
    
    # Set the models field at boundary nodes
    for terminalIdx in range (1,numberOfTerminalNodes1D+1):
        nodeIdx = terminalNodeNumbers1D[terminalIdx-1]
        nodeDomain = decomposition1D.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            versionIdx = 1
            CellMLModelsField.ParameterSetUpdateNode(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                     versionIdx,1,nodeIdx,1,CellMLModelIndex[terminalIdx])

    CellMLStateField = CMISS.Field()
    CellML.StateFieldCreateStart(CellMLStateFieldUserNumber,CellMLStateField)
    CellML.StateFieldCreateFinish()

    CellMLParametersField = CMISS.Field()
    CellML.ParametersFieldCreateStart(CellMLParametersFieldUserNumber,CellMLParametersField)
    CellML.ParametersFieldCreateFinish()

    CellMLIntermediateField = CMISS.Field()
    CellML.IntermediateFieldCreateStart(CellMLIntermediateFieldUserNumber,CellMLIntermediateField)
    CellML.IntermediateFieldCreateFinish()

    # Finish the parameter update
    dependentField1DNS.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U1,CMISS.FieldParameterSetTypes.VALUES)
    dependentField1DNS.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U1,CMISS.FieldParameterSetTypes.VALUES)
# DOC-END cellml define field maps


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
#   L1                 L2                         L3                                 L4                          L5
#                                                                             
#                                                                               | 1) 0D Simple subloop              | 1) 0D CellML Solver
#                                                                               |
#  Time Loop --- | 1) Iterative 3D-1D Coupling --- | 1) 1D-0D Iterative coupling   | 2) 1D NS/C coupling subloop ---   | CharacteristicSolver (solver 1)n
#                     (while loop, subloop 1)      | convergence (while loop)      | (while loop, subloop 1)           | 1DNavierStokesSolver (solver 2)
#                                               |
#                                               | 2) 3D Navier-Stokes subloop --- | 3DNavierStokesSolver (solver 1)
#                                               |    (simple, subloop 2)

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
iterative3D1DLoop.AbsoluteToleranceSet(flowTolerance3D1D)
iterative3D1DLoop.AbsoluteTolerance2Set(stressTolerance3D1D)
iterative3D1DLoop.RelativeToleranceSet(relativeTolerance3D1D)

# Set tolerances for iterative convergence loops
if (RCRBoundaries):
    iterative1DLoop = CMISS.ControlLoop()
    problem.ControlLoopGet([1,1,2,CMISS.ControlLoopIdentifiers.NODE],iterative1DLoop)
    iterative1DLoop.AbsoluteToleranceSet(couplingTolerance1D)
    iterative1D0DLoop = CMISS.ControlLoop()
    problem.ControlLoopGet([1,1,CMISS.ControlLoopIdentifiers.NODE],iterative1D0DLoop)
    iterative1D0DLoop.AbsoluteToleranceSet(couplingTolerance1D0D)
else:
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
#  0D
# -----------------------------------------------
# 1st Solver, Simple 0D subloop - CellML
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    problem.SolverGet([1,1,1,CMISS.ControlLoopIdentifiers.NODE],1,CellMLSolver)
    CellMLSolver.OutputTypeSet(CMISS.SolverOutputTypes.NONE)

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
nonlinearSolver1DC = CMISS.Solver()
problem.SolverGet([1,1,2,CMISS.ControlLoopIdentifiers.NODE],1,nonlinearSolver1DC)
nonlinearSolver1DC.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS)
nonlinearSolver1DC.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
nonlinearSolver1DC.newtonAbsoluteTolerance = 1.0E-8
nonlinearSolver1DC.newtonSolutionTolerance = 1.0E-8
nonlinearSolver1DC.newtonRelativeTolerance = 1.0E-8
# Get the nonlinear linear solver
linearSolver1DC = CMISS.Solver()
nonlinearSolver1DC.NewtonLinearSolverGet(linearSolver1DC)
linearSolver1DC.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
# Set the solver settings
linearSolver1DC.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
linearSolver1DC.LinearIterativeMaximumIterationsSet(100000)
linearSolver1DC.LinearIterativeDivergenceToleranceSet(1.0E+10)
linearSolver1DC.LinearIterativeRelativeToleranceSet(1.0E-8)
linearSolver1DC.LinearIterativeAbsoluteToleranceSet(1.0E-8)
linearSolver1DC.LinearIterativeGMRESRestartSet(3000)

# Navier-Stokes
dynamicSolver1DNS = CMISS.Solver()
problem.SolverGet([1,1,2,CMISS.ControlLoopIdentifiers.NODE],2,dynamicSolver1DNS)
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
dynamicSolver3D.dynamicTheta = [1.0]
nonlinearSolver3D = CMISS.Solver()
dynamicSolver3D.DynamicNonlinearSolverGet(nonlinearSolver3D)
nonlinearSolver3D.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
nonlinearSolver3D.outputType = CMISS.SolverOutputTypes.NONE
nonlinearSolver3D.newtonAbsoluteTolerance = 1.0E-10
nonlinearSolver3D.newtonRelativeTolerance = 1.0E-8
nonlinearSolver3D.newtonSolutionTolerance = 1.0E-10
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
#  0D
# -----------------------------------------------
# CellML Solver
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    CellMLEquations = CMISS.CellMLEquations()
    problem.CellMLEquationsCreateStart()
    problem.SolverGet([1,1,1,CMISS.ControlLoopIdentifiers.NODE],1,CellMLSolver)
    CellMLSolver.CellMLEquationsGet(CellMLEquations)
    # Add in the equations set
    CellMLEquations.CellMLAdd(CellML)    
    problem.CellMLEquationsCreateFinish()

# -----------------------------------------------
#  1D
# -----------------------------------------------

solver1DC = CMISS.Solver()
solver1DNS = CMISS.Solver()
solverEquations1DC = CMISS.SolverEquations()
solverEquations1DNS = CMISS.SolverEquations()
# Characteristic
if (RCRBoundaries):
    problem.SolverGet([1,1,2,CMISS.ControlLoopIdentifiers.NODE],1,solver1DC)
else:
    problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],1,solver1DC)
solver1DC.SolverEquationsGet(solverEquations1DC)
solverEquations1DC.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
equationsSet1DC = solverEquations1DC.EquationsSetAdd(equationsSet1DC)
# Navier-Stokes
if (RCRBoundaries):
    problem.SolverGet([1,1,2,CMISS.ControlLoopIdentifiers.NODE],2,solver1DNS)
else:
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

# O u t l e t
#--------------
# Outlet boundary nodes: Coupling stress (traction)
value=0.0
for branch in range(2):
    for nodeNumber in outletNodes3D[branch]:
        nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DPressure)
        if (nodeDomain == computationalNodeNumber):
            boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                         1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                         nodeNumber,4,CMISS.BoundaryConditionsTypes.COUPLING_STRESS,value)
    # Outlet boundary elements
    print('outlet branch: ' + str(branch))
    for elementNumber in outletElements3D[branch]:
        elementDomain=decomposition3D.ElementDomainGet(elementNumber)
        if (elementDomain == computationalNodeNumber):
            boundaryID = 3.0 + float(branch)
            print('   element ' + str(elementNumber))
            # Boundary ID: used to identify common faces for flowrate calculation
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,8,boundaryID)
            # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
            equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                            elementNumber,9,CMISS.BoundaryConditionsTypes.COUPLING_STRESS)
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
# inlet boundary nodes p = f(t) - will be updated in pre-solve
value = 0.0
boundaryType = CMISS.BoundaryConditionsTypes.FIXED_INLET#FITTED
# if fitData:
#     boundaryType = CMISS.BoundaryConditionsTypes.FIXED_FITTED
#     print('setting inlet boundary to fitted velocity')
# else:
#     boundaryType = CMISS.BoundaryConditionsTypes.FIXED_INLET
#     print('setting inlet boundary to fixed velocity')

for nodeNumber in inletNodes3D:
    nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,meshComponent3DVelocity)
    if (nodeDomain == computationalNodeNumber):
        for component in range(1,4):
                boundaryConditions3D.SetNode(dependentField3D,CMISS.FieldVariableTypes.U,
                                             1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                             nodeNumber,component,boundaryType,value)
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

solverEquations3D.BoundaryConditionsCreateFinish()

# -----------------------------------------------
#  1D
# -----------------------------------------------
# Characteristic
boundaryConditions1DC = CMISS.BoundaryConditions()
solverEquations1DC.BoundaryConditionsCreateStart(boundaryConditions1DC)
# Inlet: coupling
for nodeIdx in couplingNodes1D:
    nodeNumber = nodeIdx
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      1,1,nodeIdx,1,CMISS.BoundaryConditionsTypes.COUPLING_FLOW,Q0)
# outlets: terminals
versionIdx = 1
for terminalIdx in range (1,numberOfTerminalNodes1D+1):
    nodeNumber = terminalNodeNumbers1D[terminalIdx-1]
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A0[nodeNumber-1][0])
        elif (RCRBoundaries):
            boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_CELLML,A[nodeNumber-1,0])
        else:
            boundaryConditions1DC.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber-1,0])


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
# Inlet: coupling
for nodeIdx in couplingNodes1D:
    nodeNumber = nodeIdx
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                      1,1,nodeIdx,1,CMISS.BoundaryConditionsTypes.COUPLING_FLOW,Q0)
# outlets: terminals
versionIdx = 1
for terminalIdx in range (1,numberOfTerminalNodes1D+1):
    nodeNumber = terminalNodeNumbers1D[terminalIdx-1]
    nodeDomain = decomposition1D.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A0[nodeNumber-1][0])
        elif (RCRBoundaries):
            boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_CELLML,A[nodeNumber-1,0])
        else:
            boundaryConditions1DNS.SetNode(dependentField1DNS,CMISS.FieldVariableTypes.U,
                                          versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber-1,0])

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

if fitData:
    numberOfNodes = numberOfNodes3D
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
    addZeroLayer = False
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
    dataPoints.CreateStart(region3D,numberOfPcvPoints)
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
    nodeLocations = numpy.zeros((numberOfNodes3D,numberOfDimensions))
    nodeData = numpy.zeros((numberOfNodes3D,numberOfDimensions))
    # Get node locations from the mesh topology
    for node in xrange(numberOfNodes3D):
        nodeId = node + 1
        nodeNumber = nodes3D.UserNumberGet(nodeId)
        nodeNumberPython = nodeNumber - 1
        nodeDomain=decomposition3D.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            nodeList.append(nodeNumberPython)
            for component in xrange(numberOfDimensions):
                componentId = component + 1
                value = geometricField3D.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,componentId)
                nodeLocations[nodeNumberPython,component] = value

    # Calculate weights based on data point/node distance
    print("Calculating geometric-based weighting parameters")
    dataList = [[] for i in range(numberOfNodes3D+1)]
    sumWeights = numpy.zeros((numberOfNodes3D+1))
    weight = numpy.zeros((numberOfNodes3D+1,((vicinityFactor*2)**3 + numberOfWallNodes3D*3)))
    fit.CalculateWeights(p,vicinityFactor,dataPointResolution,pcvGeometry,
                         nodeLocations,nodeList,wallNodes3D,dataList,weight,sumWeights)

    # Apply weights to interpolate nodal velocity
    for timestep in xrange(startFit,numberOfTimesteps):
        # Calculate node-based velocities
        velocityNodes = numpy.zeros((numberOfNodes3D,3))
        fit.VectorFit(velocityDataPoints[timestep],velocityNodes,nodeList,wallNodes3D,dataList,weight,sumWeights)
        if timestep == startFit:
            # Update Field
            if initialiseVelocity:
                print('Initialising velocity field to PCV solution')
                for nodeNumberPython in nodeList:
                    nodeNumberCmiss = nodeNumberPython + 1
                    for component in xrange(numberOfDimensions):
                        componentId = component + 1
                        value = velocityNodes[nodeNumberPython,component]
                        dependentField3D.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
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
                    outputData = numpy.zeros((numberOfNodes3D,3))
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
            fields.CreateRegion(region3D)
            print('mesh name: ' + meshName)
            fields.NodesExport( outputDirectory +'/fittingResults/' + meshName + "_t"+ str(timestep),"FORTRAN")
            fields.Finalise()
            timeStop = time.time()            
            print("Finished CMGUI data export, time: " + str(timeStop-timeStart))

    endIdwTime = time.time()
    idwTime = endIdwTime - startIdwTime 
    print("time for idw solve: " + str(idwTime))


#=========================================================
# S O L V E
#=========================================================

# Solve the coupled problem
print("solving problem...")
# change to new directory and solve problem (note will return to original directory on exit)
with ChangeDirectory(outputDirectory):
    problem.Solve()
#========================================================

print("Finished. Time to solve (seconds): " + str(time.time()-startRunTime))    

CMISS.Finalise()
