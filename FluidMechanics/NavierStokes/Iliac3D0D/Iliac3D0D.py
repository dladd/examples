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
numberOfPeriods = 4.0
timePeriod      = 1020.0
timeIncrement   = 0.2
startTime       = 0.0
stopTime  = numberOfPeriods*timePeriod + timeIncrement*0.01 
outputFrequency = 1
dynamicSolverNavierStokesTheta = [1.0]
flowTolerance3D0D = 1.0E-5
stressTolerance3D0D = 1.0E-5
relativeTolerance3D0D = 0.001

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

# make a new output directory if necessary
outputDirectory = "./output/" #ReymondIliac_Dt" + str(round(timeIncrement,5)) + meshName + "/"
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   

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

# Apply scale factors        
density = density*Rhos
viscosity  = viscosity*Mus

#==========================================================
# Setup
#==========================================================

if computationalNodeNumber == 0:
    print(""" Setting up the problem and solving


        Coupled flow from a 3D iliac bifurcation to 0D 

                                    
        -------------------------------| 
        >                              ||/\//\/\\
        ->                   ----------|          
        --> u(r,t)          |           
        ->                   ----------|          
        >                              ||/\/\/\/\/
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
#for component in range(1,3):
#    equationsSetField3D.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,component,meshComponent3DVelocity)        
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
            nodeDomain = decomposition3D.NodeDomainGet(nodeNumber,1)
            if (nodeDomain == computationalNodeNumber):
                analyticField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1,
                                                       nodeNumber,parameterNumber,parameter)
if not analyticInflow:
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

#DOC-START cellml define field maps
#================================================================================================================================
#  RCR CellML Model Maps
#================================================================================================================================
RCRBoundaries = True
if (RCRBoundaries):

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
    modelDirectory = './input/CellMLModels/'

    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,region3D)

    CellMLModel = [0,0]
    # Windkessel Models 
    modelFile = modelDirectory + 'LeftIliac.cellml'
    print('reading model: ' + modelFile)
    CellMLModel[0] = CellML.ModelImport(modelFile)
    # known (to OpenCMISS) variables
    CellML.VariableSetAsKnown(CellMLModel[0],"Circuit/Qin")
    # to get from the CellML side 
    CellML.VariableSetAsWanted(CellMLModel[0],"Circuit/Pout")
    modelFile = modelDirectory + 'RightIliac.cellml'
    print('reading model: ' + modelFile)
    CellMLModel[1] = CellML.ModelImport(modelFile)
    # known (to OpenCMISS) variables
    CellML.VariableSetAsKnown(CellMLModel[1],"Circuit/Qin")
    # to get from the CellML side 
    CellML.VariableSetAsWanted(CellMLModel[1],"Circuit/Pout")
    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    equationsSetField3D.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_ITERATION_VALUES)
    for outletIdx in range(2):
        for nodeIdx in outletNodes3D[outletIdx]:
            nodeDomain = decomposition3D.NodeDomainGet(nodeIdx,1)            
            if nodeDomain == computationalNodeNumber:
                #equationsSetFieldComponent = equationsSetField3D.ComponentMeshComponentGet(CMISS.FieldVariableTypes.U,1,1)
                #dependentFieldComponent = dependentField3D.ComponentMeshComponentGet(CMISS.FieldVariableTypes.U,4,1)


                CellML.CreateFieldToCellMLMap(equationsSetField3D,CMISS.FieldVariableTypes.U,1,
                                              CMISS.FieldParameterSetTypes.VALUES,CellMLModel[outletIdx],
                                              "Circuit/Qin",CMISS.FieldParameterSetTypes.VALUES)
                # Map the returned pressure values from CellML --> CMISS
                # pCellML is component 1 of the Dependent field U variable
                CellML.CreateCellMLToFieldMap(CellMLModel[outletIdx],"Circuit/Pout",CMISS.FieldParameterSetTypes.VALUES,
                                              dependentField3D,CMISS.FieldVariableTypes.U,4,CMISS.FieldParameterSetTypes.PRESSURE_VALUES)

                # Map the returned pressure values from CellML --> CMISS
                # pCellML is component 1 of the Dependent field U variable
                CellML.CreateCellMLToFieldMap(CellMLModel[outletIdx],"Circuit/Qin",CMISS.FieldParameterSetTypes.VALUES,
                                              equationsSetField3D,CMISS.FieldVariableTypes.U,1,CMISS.FieldParameterSetTypes.PREVIOUS_ITERATION_VALUES)
                CellML.CreateCellMLToFieldMap(CellMLModel[outletIdx],"Circuit/Pout",CMISS.FieldParameterSetTypes.VALUES,
                                              equationsSetField3D,CMISS.FieldVariableTypes.U,2,CMISS.FieldParameterSetTypes.PREVIOUS_ITERATION_VALUES)


    # Finish the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateFinish()

    CellMLModelsField = CMISS.Field()
    CellML.ModelsFieldCreateStart(CellMLModelsFieldUserNumber,CellMLModelsField)
    CellML.ModelsFieldCreateFinish()

    # ModelIndex
    for outletIdx in range(2):
        for nodeIdx in outletNodes3D[outletIdx]:
            nodeDomain = decomposition3D.NodeDomainGet(nodeIdx,1)            
            if nodeDomain == computationalNodeNumber:
                versionIdx = 1
                CellMLModelsField.ParameterSetUpdateNode(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                         versionIdx,1,nodeIdx,1,CellMLModel[outletIdx])

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
    equationsSetField3D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
    equationsSetField3D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
    dependentField3D.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PRESSURE_VALUES)
    dependentField3D.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PRESSURE_VALUES)
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

# ========================================================
#  Problem
# ========================================================
# Create coupled (multiscale) problem
problem = CMISS.Problem()
problem.CreateStart(problemUserNumber)
if RCRBoundaries:
    problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                             CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                             CMISS.ProblemSubTypes.COUPLED3D0D_NAVIER_STOKES)
else:
    problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                             CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                             CMISS.ProblemSubTypes.TRANSIENT_RBS_NAVIER_STOKES)
problem.CreateFinish()

# ========================================================
#  Control Loops
# ========================================================
#   L1                 L2                         L3                                 L4                          L5
#                                                                             

#  Time Loop --- | 1) Iterative 3D-0D Coupling --- | 1) 0D Simple subloop          | 1) 0D CellML Solver
#                     (while loop, subloop 1)      
#                                                  |
   #                                               | 2) 3D Navier-Stokes subloop --- | 3DNavierStokesSolver (solver 1)
#                                                  |    (simple, subloop 2)

problem.ControlLoopCreateStart()

# Create time loop
timeLoop = CMISS.ControlLoop()
problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],timeLoop)
timeLoop.LabelSet('Time Loop')
timeLoop.TimesSet(startTime,stopTime,timeIncrement)
timeLoop.TimeOutputSet(outputFrequency)

if (RCRBoundaries):
    # Create iterative 3D-1D coupling loop
    iterative3D0DLoop = CMISS.ControlLoop()
    problem.ControlLoopGet([1,CMISS.ControlLoopIdentifiers.NODE],iterative3D0DLoop)
    iterative3D0DLoop.AbsoluteToleranceSet(flowTolerance3D0D)
    iterative3D0DLoop.AbsoluteTolerance2Set(stressTolerance3D0D)
    #iterative3D0DLoop.RelativeToleranceSet(relativeTolerance3D1D)

    # Create simple 3D NS loop
    simple3DLoop = CMISS.ControlLoop()
    problem.ControlLoopGet([1,2,CMISS.ControlLoopIdentifiers.NODE],simple3DLoop)

problem.ControlLoopCreateFinish()

# ========================================================
#  Solvers
# ========================================================
problem.SolversCreateStart()

if (RCRBoundaries):
    # -----------------------------------------------
    #  0D
    # -----------------------------------------------
    # 1st Solver, Simple 0D subloop - CellML
    CellMLSolver = CMISS.Solver()
    problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],1,CellMLSolver)
    CellMLSolver.OutputTypeSet(CMISS.SolverOutputTypes.NONE)
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
    nonlinearSolver3D.newtonRelativeTolerance = 1.0E-6
    nonlinearSolver3D.newtonSolutionTolerance = 1.0E-8
    nonlinearSolver3D.newtonMaximumFunctionEvaluations = 10000
    linearSolver3D = CMISS.Solver()
    nonlinearSolver3D.NewtonLinearSolverGet(linearSolver3D)
    linearSolver3D.outputType = CMISS.SolverOutputTypes.NONE
    linearSolver3D.linearType = CMISS.LinearSolverTypes.DIRECT
    linearSolver3D.libraryType = CMISS.SolverLibraries.MUMPS
else:
    # -----------------------------------------------
    #  3D
    # -----------------------------------------------
    # Create problem solver
    dynamicSolver3D = CMISS.Solver()
    problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver3D)
    dynamicSolver3D.outputType = CMISS.SolverOutputTypes.NONE
    dynamicSolver3D.dynamicTheta = [1.0]
    nonlinearSolver3D = CMISS.Solver()
    dynamicSolver3D.DynamicNonlinearSolverGet(nonlinearSolver3D)
    nonlinearSolver3D.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
    nonlinearSolver3D.outputType = CMISS.SolverOutputTypes.NONE
    nonlinearSolver3D.newtonAbsoluteTolerance = 1.0E-8
    nonlinearSolver3D.newtonRelativeTolerance = 1.0E-6
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

if (RCRBoundaries):
    # -----------------------------------------------
    #  0D
    # -----------------------------------------------
    # CellML Solver
    CellMLSolver = CMISS.Solver()
    CellMLEquations = CMISS.CellMLEquations()
    problem.CellMLEquationsCreateStart()
    problem.SolverGet([1,1,CMISS.ControlLoopIdentifiers.NODE],1,CellMLSolver)
    CellMLSolver.CellMLEquationsGet(CellMLEquations)
    # Add in the equations set
    CellMLEquations.CellMLAdd(CellML)    
    problem.CellMLEquationsCreateFinish()
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
else:
    # -----------------------------------------------
    #  3D
    # -----------------------------------------------
    # Create solver equations and add equations set to solver equations
    solver3D = CMISS.Solver()
    solverEquations3D = CMISS.SolverEquations()
    problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,solver3D)
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
                                         nodeNumber,4,CMISS.BoundaryConditionsTypes.FREE,value)
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
            if RCRBoundaries:
                # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
                equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                elementNumber,9,CMISS.BoundaryConditionsTypes.FIXED_CELLML)
            else:
                # Boundary Type: workaround since we don't have access to BC object during FE evaluation routines
                equationsSetField3D.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                elementNumber,9,CMISS.BoundaryConditionsTypes.FREE)

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
