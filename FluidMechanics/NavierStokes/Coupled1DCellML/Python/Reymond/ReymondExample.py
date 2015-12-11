#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an example program to solve for flow using 1D transient Navier-Stokes 
#>  over an arterial tree with coupled 0D lumped models (RCR) defined in CellML. The geometry and
#>  boundary conditions are based on published data from Reymond et al. 2011: 'Validation of a patient-specific one-dimensional model of the systemic arterial tree'
#>  Results are compared against the data presented in this paper.
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
#> The Original Code is OpenCMISS
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
#> of those above. If you wish to allow use of your version of this file only
#> under the terms of either the GPL or the LGPL, and not to allow others to
#> use your version of this file under the terms of the MPL, indicate your
#> decision by deleting the provisions above and replace them with the notice
#> and other provisions required by the GPL or the LGPL. If you do not delete
#> the provisions above, a recipient may use your version of this file under
#> the terms of any one of the MPL, the GPL or the LGPL.
#>
#> OpenCMISS/examples/FluidMechanics/NavierStokes/Coupled1DCellML/Python/Reymond/ReymondExample.py
#>

#================================================================================================================================
#  Initialise OpenCMISS and any other needed libraries
#================================================================================================================================
import numpy as np
import math,csv,time,sys,os,glob,shutil
import contextlib
import FluidExamples1DUtilities as Utilities1D

sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))
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

#================================================================================================================================
#  Set up field and system values 
#================================================================================================================================

(CoordinateSystemUserNumber,
 BasisUserNumber,
 RegionUserNumber,
 MeshUserNumber,
 DecompositionUserNumber,
 GeometricFieldUserNumber,
 DependentFieldUserNumber,
 MaterialsFieldUserNumber,  
 IndependentFieldUserNumber,
 EquationsSetUserNumberCharacteristic,
 EquationsSetUserNumberNavierStokes,
 EquationsSetFieldUserNumberCharacteristic,
 EquationsSetFieldUserNumberNavierStokes,
 ProblemUserNumber,
 CellMLUserNumber,
 CellMLModelsFieldUserNumber,
 CellMLStateFieldUserNumber,
 CellMLIntermediateFieldUserNumber,
 CellMLParametersFieldUserNumber,
 AnalyticFieldUserNumber) = range(1,21)

# Solver user numbers
SolverDAEUserNumber            = 1
SolverCharacteristicUserNumber = 2
SolverNavierStokesUserNumber   = 3

# Other system constants
numberOfDimensions     = 1  #(One-dimensional)
numberOfComponents     = 2  #(Flow & Area)

# Get the computational nodes info
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber    = CMISS.ComputationalNodeNumberGet()

#================================================================================================================================
#  Problem Control Panel
#================================================================================================================================

# Set the flags
RCRBoundaries            = True   # Set to use coupled 0D Windkessel models (from CellML) at model outlet boundaries
nonReflecting            = False    # Set to use non-reflecting outlet boundaries
CheckTimestepStability   = False   # Set to do a basic check of the stability of the hyperbolic problem based on the timestep size
initialiseFromFile       = False   # Set to initialise values
ProgressDiagnostics      = True   # Set to diagnostics
simpleTube               = False   # Set to solve over a simple 1d tube 
simpleBif                = False   # Set to solve over a simple 1d tube 
reymondRefined           = False   # Set to solve over a simple 1d tube 

if(nonReflecting and RCRBoundaries):
    sys.exit('Please set either RCR or non-reflecting boundaries- not both.')

#================================================================================================================================
#  Mesh Reading
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> Reading geometry from files... << == "

# Read nodes
inputNodeNumbers = []
branchNodeNumbers = []
coupledNodeNumbers = []
nodeCoordinates = []
branchNodeElements = []
terminalArteryNames = []
RCRParameters = []
if simpleTube:
    if simpleBif:
        filename='input/simpleBif/simpleBifNodes.csv'
    else:
        filename='input/simpleTube/simpleNode.csv'
else:
    filename='input/Reymond2009_Nodes.csv'
if reymondRefined:
    filename='input/reymondRefined/Node.csv'
    Utilities1D.CsvNodeReader(filename,inputNodeNumbers,branchNodeNumbers,coupledNodeNumbers,
                           nodeCoordinates,branchNodeElements,terminalArteryNames,RCRParameters)
else:
    Utilities1D.CsvNodeReader2(filename,inputNodeNumbers,branchNodeNumbers,coupledNodeNumbers,
                           nodeCoordinates,branchNodeElements,terminalArteryNames,RCRParameters)
numberOfInputNodes     = len(inputNodeNumbers)
numberOfNodes          = len(nodeCoordinates)
numberOfBranches       = len(branchNodeNumbers)
numberOfTerminalNodes  = len(coupledNodeNumbers)

# Read elements
elementNodes = []
elementArteryNames = []
elementNodes.append([0,0,0])
if simpleTube:
    if simpleBif:
        filename='input/simpleBif/simpleBifElements.csv'
    else:
        filename='input/simpleTube/simpleElement.csv'
else:
    filename='input/Reymond2009_Elements.csv'
Utilities1D.CsvElementReader2(filename,elementNodes,elementArteryNames)
numberOfElements = len(elementNodes)-1
        
if (ProgressDiagnostics):
    print " Number of nodes: " + str(numberOfNodes)
    print " Number of elements: " + str(numberOfElements)
    print " Input at nodes: " + str(inputNodeNumbers)
    print " Branches at nodes: " + str(branchNodeNumbers)
    print " Terminal at nodes: " + str(coupledNodeNumbers)
    print " == >> Finished reading geometry... << == "

#================================================================================================================================
#  Initial Data & Default Values
#================================================================================================================================

# Set the material parameters
Rho  = 1050.0     # Density     (kg/m3)
Mu   = 0.004      # Viscosity   (Pa.s)
G0   = 0.0        # Gravitational acceleration (m/s2)
#Pext = 0.0 #12932.2697 # External pressure (Pa)
Alpha = 4.0/3.0    # Flow profile type: 4/3 parabolic, 1 flat

# Material parameter scaling factors
Ls = 1000.0              # Length   (m -> mm)
Ts = 1000.0              # Time     (s -> ms)
Ms = 1000.0              # Mass     (kg -> g)
Qs    = (Ls**3.0)/Ts     # Flow             (m3/s)  
As    = Ls**2.0          # Area             (m2)
Hs    = Ls               # vessel thickness (m)
Es    = Ms/(Ls*Ts**2.0)  # Elasticity Pa    (kg/(ms2) --> g/(mm.ms^2)
Rhos  = Ms/(Ls**3.0)     # Density          (kg/m3)
Mus   = Ms/(Ls*Ts)       # Viscosity        (kg/(ms))
Ps    = Ms/(Ls*Ts**2.0)  # Pressure         (kg/(ms2))
Gs    = Ls/(Ts**2.0)     # Acceleration    (m/s2)

# Initialise the node-based parameters
A0   = []
H    = []
E    = []
# Read the MATERIAL csv file
if simpleTube:
    if simpleBif:
        filename='input/simpleBif/simpleBifMaterials.csv'
    else:
        filename = 'input/simpleTube/simpleMaterial.csv'
else:
    filename = 'input/Reymond2009_Materials.csv'
if reymondRefined:
    filename='input/reymondRefined/Materials.csv'
    print('Reading materials from: '+filename)
    Utilities1D.CsvMaterialReader(filename,A0,E,H)
else:
    print('Reading materials from: '+filename)
    Utilities1D.CsvMaterialReader2(filename,A0,E,H)
for i in range(len(A0[:])):
    for j in range(len(A0[i][:])):
        A0[i][j] = A0[i][j]*As
        E[i][j] = E[i][j]*Es#*0.5 #DEBUG: reduce Young's modulus
        H[i][j] = H[i][j]*Hs#*0.5 #DEBUG: reduce 
        # #DEBUG: set middle nodes to distal values
        # if i % 2 == 0:
        #     A0[i][j] = A0[i][j]*As
        #     E[i][j] = E[i][j]*Es
        #     H[i][j] = H[i][j]*Hs            
        # else:
        #     A0[i][j] = A0[i+1][j]*As
        #     E[i][j] = E[i+1][j]*Es
        #     H[i][j] = H[i+1][j]*Hs

# Initial conditions
# Zero reference state: Q=0, A=A0 state
# Reymond2009 initial conditions: Q = 1 ml/s, A = 100 mmHg
Q  = np.zeros((numberOfNodes,4))
A  = np.zeros((numberOfNodes,4))
dQ = np.zeros((numberOfNodes,4))
dA = np.zeros((numberOfNodes,4))
for i in range(len(A0[:])):
    for j in range(len(A0[i][:])):
            A[i,j] = A0[i][j]

pInit = 0.0 #0.0133322
#pExternal = 0.0133322 # 100 mmHg
pExternal = 0.0 #70.0*0.000133322 # 80 mmHg
#pExternal = 0.0

#A = A0
if initialiseFromFile:
    #filename = './output/Reymond2009ExpInputCellML/MainTime_39500.part0.exnode'
    filename = './output/Reymond2009ExpInputCellML_30/MainTime_78260.part0.exnode'
    Utilities1D.ExnodeInitReader(filename,Q,A,dQ,dA)
elif abs(pInit) > 0.0001:
    # non-zero pressure
    for i in range(len(A0[:])):
        for j in range(len(A0[i][:])):
            A[i][j]  = ((pInit-pExternal)*(3.0*A0[i][j])/(4.0*math.pi*H[i][j]*E[i][j]) + math.sqrt(A0[i][j]))**2.0

# Apply scale factors        
Rho = Rho*Rhos
Mu = Mu*Mus
print('density: '+str(Rho))
print('viscosity: '+str(Mu))
#!DEBUG
#Mu  = Rho/(8.0*math.pi) # kappa = 1
G0  = G0*Gs

# Set the output parameters
# (NONE/PROGRESS/TIMING/SOLVER/MATRIX)
dynamicSolverNavierStokesOutputType    = CMISS.SolverOutputTypes.NONE
nonlinearSolverNavierStokesOutputType  = CMISS.SolverOutputTypes.NONE
nonlinearSolverCharacteristicsOutputType = CMISS.SolverOutputTypes.NONE
linearSolverCharacteristicOutputType    = CMISS.SolverOutputTypes.NONE
linearSolverNavierStokesOutputType     = CMISS.SolverOutputTypes.NONE
# (NONE/TIMING/SOLVER/MATRIX)
cmissSolverOutputType = CMISS.SolverOutputTypes.NONE
dynamicSolverNavierStokesOutputFrequency = 10

# Set the time parameters
numberOfPeriods = 3.0
timePeriod      = 790.
timeIncrement   = 0.2
startTime       = 0.0
stopTime  = numberOfPeriods*timePeriod + timeIncrement*0.01 
dynamicSolverNavierStokesTheta = [1.0]

# Set the solver parameters
relativeToleranceNonlinearNavierStokes   = 1.0E-07  # default: 1.0E-05
absoluteToleranceNonlinearNavierStokes   = 1.0E-10  # default: 1.0E-10
solutionToleranceNonlinearNavierStokes   = 1.0E-07  # default: 1.0E-05
relativeToleranceLinearNavierStokes      = 1.0E-07  # default: 1.0E-05
absoluteToleranceLinearNavierStokes      = 1.0E-10  # default: 1.0E-10
relativeToleranceNonlinearCharacteristic = 1.0E-07  # default: 1.0E-05
absoluteToleranceNonlinearCharacteristic = 1.0E-10  # default: 1.0E-10
solutionToleranceNonlinearCharacteristic = 1.0E-07  # default: 1.0E-05
relativeToleranceLinearCharacteristic    = 1.0E-07  # default: 1.0E-05
absoluteToleranceLinearCharacteristic    = 1.0E-10  # default: 1.0E-10

DIVERGENCE_TOLERANCE = 1.0E+10  # default: 1.0E+05
MAXIMUM_ITERATIONS   = 100000   # default: 100000
RESTART_VALUE        = 3000     # default: 30

# N-S/C coupling tolerance
couplingTolerance1D = 1.0E+15
# 1D-0D coupling tolerance
couplingTolerance1D0D = 0.001

# Navier-Stokes solver
if(RCRBoundaries):
    EquationsSetSubtype = CMISS.EquationsSetSubtypes.COUPLED1D0D_NAVIER_STOKES
    # Characteristic solver
    EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.CHARACTERISTIC
    ProblemSubtype = CMISS.ProblemSubTypes.COUPLED1D0D_NAVIER_STOKES
else:
    EquationsSetSubtype = CMISS.EquationsSetSubtypes.TRANSIENT1D_NAVIER_STOKES
    # Characteristic solver
    EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.CHARACTERISTIC
    ProblemSubtype = CMISS.ProblemSubTypes.TRANSIENT1D_NAVIER_STOKES

#================================================================================================================================
#  Coordinate System
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> COORDINATE SYSTEM << == "

# Start the creation of RC coordinate system
CoordinateSystem = CMISS.CoordinateSystem()
CoordinateSystem.CreateStart(CoordinateSystemUserNumber)
CoordinateSystem.DimensionSet(3)
CoordinateSystem.CreateFinish()

#================================================================================================================================
#  Region
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> REGION << == "

# Start the creation of  region
Region = CMISS.Region()
Region.CreateStart(RegionUserNumber,CMISS.WorldRegion)
Region.label = "ArterialSystem"
Region.coordinateSystem = CoordinateSystem
Region.CreateFinish()

#================================================================================================================================
#  Bases
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> BASIS << == "

# Start the creation of  bases
basisXiGauss = 3
Basis = CMISS.Basis()
Basis.CreateStart(BasisUserNumber)
Basis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
Basis.numberOfXi = numberOfDimensions
Basis.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]
Basis.quadratureNumberOfGaussXi = [basisXiGauss]
Basis.CreateFinish()

#================================================================================================================================
#  Nodes
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> NODES << == "

# Start the creation of mesh nodes
Nodes = CMISS.Nodes()
Nodes.CreateStart(Region,numberOfNodes)
Nodes.CreateFinish()

#================================================================================================================================
#  Mesh
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MESH << == "

# Start the creation of  mesh
Mesh = CMISS.Mesh()
Mesh.CreateStart(MeshUserNumber,Region,numberOfDimensions)
Mesh.NumberOfElementsSet(numberOfElements)
meshNumberOfComponents = 1
# Specify the mesh components
Mesh.NumberOfComponentsSet(meshNumberOfComponents)
# Specify the mesh components
MeshElements = CMISS.MeshElements()
meshComponentNumber = 1

# Specify the  mesh component
MeshElements.CreateStart(Mesh,meshComponentNumber,Basis)
for elemIdx in range(1,numberOfElements+1):
    MeshElements.NodesSet(elemIdx,elementNodes[elemIdx])

for nodeIdx in range(len(branchNodeNumbers)):
    versionIdx = 1
    for element in branchNodeElements[nodeIdx]:
        if versionIdx == 1:
            MeshElements.LocalElementNodeVersionSet(element,versionIdx,1,3)            
        else:
            MeshElements.LocalElementNodeVersionSet(element,versionIdx,1,1)            
        versionIdx+=1

MeshElements.CreateFinish()

# Finish the creation of the mesh
Mesh.CreateFinish()

#================================================================================================================================
#  Decomposition
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MESH DECOMPOSITION << == "

# Start the creation of  mesh decomposition
Decomposition = CMISS.Decomposition()
Decomposition.CreateStart(DecompositionUserNumber,Mesh)
Decomposition.TypeSet(CMISS.DecompositionTypes.CALCULATED)
Decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
Decomposition.CreateFinish()

#================================================================================================================================
#  Geometric Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> GEOMETRIC FIELD << == "

# Start the creation of  geometric field
GeometricField = CMISS.Field()
GeometricField.CreateStart(GeometricFieldUserNumber,Region)
GeometricField.NumberOfVariablesSet(1)
GeometricField.VariableLabelSet(CMISS.FieldVariableTypes.U,'Coordinates')
GeometricField.TypeSet = CMISS.FieldTypes.GEOMETRIC
GeometricField.meshDecomposition = Decomposition
GeometricField.ScalingTypeSet = CMISS.FieldScalingTypes.NONE
# Set the mesh component to be used by the geometric field components
for componentNumber in range(1,CoordinateSystem.dimension+1):
    GeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,componentNumber,
                                             meshComponentNumber)
GeometricField.CreateFinish()

# Set the geometric field values
for node in range(len(nodeCoordinates)):
    nodeNumber = node+1
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeNumber in branchNodeNumbers:
            branchNodeIdx = branchNodeNumbers.index(nodeNumber)
            numberOfVersions = len(branchNodeElements[branchNodeIdx])
        for component in range(3):
            componentNumber = component+1
            for versionIdx in range(1,numberOfVersions+1):
                GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                        versionIdx,1,nodeNumber,componentNumber,
                                                        nodeCoordinates[node][component])
# Finish the parameter update
GeometricField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
GeometricField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)     

# # Export Geometry
# Fields = CMISS.Fields()
# Fields.CreateRegion(Region)
# Fields.NodesExport("Geometry","FORTRAN")
# Fields.ElementsExport("Geometry","FORTRAN")
# Fields.Finalise()

#================================================================================================================================
#  Equations Sets
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> EQUATIONS SET << == "

# Create the equations set for CHARACTERISTIC
EquationsSetCharacteristic = CMISS.EquationsSet()
EquationsSetFieldCharacteristic = CMISS.Field()
# Set the equations set to be a static nonlinear problem
EquationsSetCharacteristic.CreateStart(EquationsSetUserNumberCharacteristic,Region,GeometricField,
                                       CMISS.EquationsSetClasses.FLUID_MECHANICS,CMISS.EquationsSetTypes.CHARACTERISTIC_EQUATION,
                                       EquationsSetCharacteristicSubtype,EquationsSetFieldUserNumberCharacteristic,EquationsSetFieldCharacteristic)
EquationsSetCharacteristic.CreateFinish()

# Create the equations set for NAVIER-STOKES
EquationsSetNavierStokes = CMISS.EquationsSet()
EquationsSetFieldNavierStokes = CMISS.Field()
# Set the equations set to be a dynamic nonlinear problem
EquationsSetNavierStokes.CreateStart(EquationsSetUserNumberNavierStokes,Region,GeometricField,
    CMISS.EquationsSetClasses.FLUID_MECHANICS,CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
     EquationsSetSubtype,EquationsSetFieldUserNumberNavierStokes,EquationsSetFieldNavierStokes)
EquationsSetNavierStokes.CreateFinish()

#================================================================================================================================
#  Dependent Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> DEPENDENT FIELD << == "

# CHARACTERISTIC
# Create the equations set dependent field variables
DependentFieldNavierStokes = CMISS.Field()
EquationsSetCharacteristic.DependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'General')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Derivatives')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.V,'Characteristics')
if (RCRBoundaries):
    DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U1,'CellML Q and P')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U2,'Pressure')
# Set the mesh component to be used by the field components.
# Flow & Area
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumber)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumber)
# Derivatives
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,meshComponentNumber)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,2,meshComponentNumber)
# Riemann
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,1,meshComponentNumber)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,2,meshComponentNumber)
# qCellML & pCellml
if (RCRBoundaries):
    DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,1,meshComponentNumber)
    DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,2,meshComponentNumber)
# Pressure
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,1,meshComponentNumber)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,2,meshComponentNumber)

EquationsSetCharacteristic.DependentCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.DependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
EquationsSetNavierStokes.DependentCreateFinish()

DependentFieldNavierStokes.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES)

# Initialise the dependent field variables
for nodeIdx in range (1,numberOfNodes+1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeIdx in branchNodeNumbers:
            branchNodeIdx = branchNodeNumbers.index(nodeIdx)
            numberOfVersions = len(branchNodeElements[branchNodeIdx])    
        for versionIdx in range(1,numberOfVersions+1):
            # U variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,1,Q[nodeIdx-1,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,2,A[nodeIdx-1,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                                versionIdx,1,nodeIdx,1,Q[nodeIdx-1,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                                versionIdx,1,nodeIdx,2,A[nodeIdx-1,versionIdx-1])
            # delUdelN variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,1,dQ[nodeIdx-1,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,2,dA[nodeIdx-1,versionIdx-1])

# revert default version to 1
versionIdx = 1
             
# Finish the parameter update
DependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
DependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)   

#================================================================================================================================
#  Materials Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MATERIALS FIELD << == "

# CHARACTERISTIC
# Create the equations set materials field variables 
MaterialsFieldNavierStokes = CMISS.Field()
EquationsSetCharacteristic.MaterialsCreateStart(MaterialsFieldUserNumber,MaterialsFieldNavierStokes)
MaterialsFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'MaterialsConstants')
MaterialsFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.V,'MaterialsVariables')
# Set the mesh component to be used by the field components.
for componentNumber in range(1,4):
    MaterialsFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,componentNumber,meshComponentNumber)
EquationsSetCharacteristic.MaterialsCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.MaterialsCreateStart(MaterialsFieldUserNumber,MaterialsFieldNavierStokes)
EquationsSetNavierStokes.MaterialsCreateFinish()

# Set the materials field constants
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,1,Mu)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,2,Rho)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,3,Alpha)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,4,pExternal)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,5,Ls)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,6,Ts)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,7,Ms)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
                                                       CMISS.FieldParameterSetTypes.VALUES,8,G0)

# Initialise the materials field variables (A0,E,H)
for node in range(numberOfNodes):
    nodeIdx = node+1
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        numberOfVersions = 1
        if nodeIdx in branchNodeNumbers:
            branchNodeIdx = branchNodeNumbers.index(nodeIdx)
            numberOfVersions = len(branchNodeElements[branchNodeIdx])            
        for versionIdx in range(1,numberOfVersions+1):
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,1,A0[node][versionIdx-1])
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,2,E[node][versionIdx-1])
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,1,nodeIdx,3,H[node][versionIdx-1])

# Finish the parameter update
MaterialsFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
MaterialsFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)

#================================================================================================================================
# Independent Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> INDEPENDENT FIELD << == "

# CHARACTERISTIC
# Create the equations set independent field variables  
IndependentFieldNavierStokes = CMISS.Field()
EquationsSetCharacteristic.IndependentCreateStart(IndependentFieldUserNumber,IndependentFieldNavierStokes)
IndependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Normal Wave Direction')
# Set the mesh component to be used by the field components.
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumber)
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumber)
EquationsSetCharacteristic.IndependentCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.IndependentCreateStart(IndependentFieldUserNumber,IndependentFieldNavierStokes)
EquationsSetNavierStokes.IndependentCreateFinish()

# Set the normal wave direction for branching nodes
for branch in range(len(branchNodeNumbers)):
    branchNode = branchNodeNumbers[branch]
    nodeDomain = Decomposition.NodeDomainGet(branchNode,meshComponentNumber)    
    if (nodeDomain == computationalNodeNumber):
        # Incoming(parent)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                              1,1,branchNode,1,1.0)
        for daughterVersion in range(2,len(branchNodeElements[branch])+1):
            # Outgoing(branches)
            IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                                  daughterVersion,1,branchNode,2,-1.0)

# Set the normal wave direction for terminal
if (RCRBoundaries or nonReflecting):
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumbers[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            # Incoming (parent) - outgoing component to come from 0D
            versionIdx = 1
            IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                  versionIdx,1,nodeIdx,1,1.0)

# Finish the parameter update
IndependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
IndependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

#================================================================================================================================
# Analytic Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> ANALYTIC FIELD << == "

AnalyticFieldNavierStokes = CMISS.Field()
EquationsSetNavierStokes.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SPLINT_FROM_FILE,AnalyticFieldUserNumber,
                                             AnalyticFieldNavierStokes)
AnalyticFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Input Flow')
EquationsSetNavierStokes.AnalyticCreateFinish()

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
    terminalPressure = 0.0#0.00133322 #pExternal #0.0 #0.00133322 # 0.0133 = 100 mmHg
    modelDirectory = './input/CellMLModels/terminalArteryRCR/'
    Utilities1D.WriteCellMLRCRModels(terminalArteryNames,RCRParameters,terminalPressure,modelDirectory)    

    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,Region)
    # Number of CellML models
    CellMLModelIndex = [0]*(numberOfTerminalNodes+1)

    # Windkessel Model
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumbers[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
        modelFile = modelDirectory + terminalArteryNames[terminalIdx-1] + '.cellml'
        print('reading model: ' + modelFile)
        if (nodeDomain == computationalNodeNumber):
            CellMLModelIndex[terminalIdx] = CellML.ModelImport(modelFile)
            # known (to OpenCMISS) variables
            CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Circuit/Qin")
            # to get from the CellML side 
            CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Circuit/Pout")
    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumbers[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
        if (nodeDomain == computationalNodeNumber):
            # Now we can set up the field variable component <--> CellML model variable mappings.
            # Map the OpenCMISS boundary flow rate values --> CellML
            # Q is component 1 of the DependentField
            CellML.CreateFieldToCellMLMap(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,1,
                                          CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[terminalIdx],
                                          "Circuit/Qin",CMISS.FieldParameterSetTypes.VALUES)
            # Map the returned pressure values from CellML --> CMISS
            # pCellML is component 1 of the Dependent field U1 variable
            CellML.CreateCellMLToFieldMap(CellMLModelIndex[terminalIdx],"Circuit/Pout",CMISS.FieldParameterSetTypes.VALUES,
                                          DependentFieldNavierStokes,CMISS.FieldVariableTypes.U1,pCellMLComponent,CMISS.FieldParameterSetTypes.VALUES)

    # Finish the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateFinish()

    CellMLModelsField = CMISS.Field()
    CellML.ModelsFieldCreateStart(CellMLModelsFieldUserNumber,CellMLModelsField)
    CellML.ModelsFieldCreateFinish()
    
    # Set the models field at boundary nodes
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumbers[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumber)
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
    DependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U1,CMISS.FieldParameterSetTypes.VALUES)
    DependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U1,CMISS.FieldParameterSetTypes.VALUES)
# DOC-END cellml define field maps

#================================================================================================================================
#  Equations
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> EQUATIONS << == "

# 2nd Equations Set - CHARACTERISTIC
EquationsCharacteristic = CMISS.Equations()
EquationsSetCharacteristic.EquationsCreateStart(EquationsCharacteristic)
EquationsCharacteristic.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
EquationsCharacteristic.outputType = CMISS.EquationsOutputTypes.NONE
EquationsSetCharacteristic.EquationsCreateFinish()

#------------------

# 3rd Equations Set - NAVIER-STOKES
EquationsNavierStokes = CMISS.Equations()
EquationsSetNavierStokes.EquationsCreateStart(EquationsNavierStokes)
EquationsNavierStokes.sparsityType = CMISS.EquationsSparsityTypes.FULL
EquationsNavierStokes.lumpingType = CMISS.EquationsLumpingTypes.UNLUMPED
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
EquationsNavierStokes.outputType = CMISS.EquationsOutputTypes.NONE
EquationsSetNavierStokes.EquationsCreateFinish()

#================================================================================================================================
#  Problems
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> PROBLEM << == "

# Start the creation of a problem.
Problem = CMISS.Problem()
Problem.CreateStart(ProblemUserNumber)
Problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                         CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,ProblemSubtype)    
Problem.CreateFinish()

#================================================================================================================================
#  Control Loops
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> PROBLEM CONTROL LOOP << == "
    
'''
   Solver Control Loops

                   L1                                 L2                        L3

1D0D
------

                                                      | 1) 0D Simple subloop   | 1) 0D/CellML DAE Solver
                                                      |                              
    Time Loop, L0  | 1) 1D-0D Iterative Coupling, L1  | 2) 1D NS/C coupling:   | 1) Characteristic Nonlinear Solver
                   |    Convergence Loop (while loop) |    (while loop)        | 2) 1DNavierStokes Transient Solver
                   |
                   | 2) (optional) Simple subloop     | 


'''

# Order of solvers within their respective subloops
SolverCharacteristicUserNumber = 1
SolverNavierStokesUserNumber   = 2
SolverCellmlUserNumber         = 1
if (RCRBoundaries):
   Iterative1d0dControlLoopNumber   = 1
   Simple0DControlLoopNumber        = 1
   Iterative1dControlLoopNumber     = 2
else:
   Iterative1dControlLoopNumber     = 1

# Start the creation of the problem control loop
TimeLoop = CMISS.ControlLoop()
Problem.ControlLoopCreateStart()
Problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],TimeLoop)
TimeLoop.LabelSet('Time Loop')
TimeLoop.TimesSet(startTime,stopTime,timeIncrement)
TimeLoop.TimeOutputSet(dynamicSolverNavierStokesOutputFrequency)

# Set tolerances for iterative convergence loops
if (RCRBoundaries):
    Iterative1DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
                            CMISS.ControlLoopIdentifiers.NODE],Iterative1DCouplingLoop)
    Iterative1DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D)
    Iterative1D0DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1d0dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                           Iterative1D0DCouplingLoop)
    Iterative1D0DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D0D)
else:
    Iterative1DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                           Iterative1DCouplingLoop)
    Iterative1DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D)

Problem.ControlLoopCreateFinish()

#================================================================================================================================
#  Solvers
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> SOLVERS << == "

# Start the creation of the problem solvers    
DynamicSolverNavierStokes     = CMISS.Solver()
NonlinearSolverNavierStokes   = CMISS.Solver()
LinearSolverNavierStokes      = CMISS.Solver()
NonlinearSolverCharacteristic = CMISS.Solver()
LinearSolverCharacteristic    = CMISS.Solver()

Problem.SolversCreateStart()

# 1st Solver, Simple 0D subloop - CellML
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverDAEUserNumber,CellMLSolver)
    CellMLSolver.OutputTypeSet(cmissSolverOutputType)

# 1st Solver, Iterative 1D subloop - CHARACTERISTIC
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
# Set the nonlinear Jacobian type
NonlinearSolverCharacteristic.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
NonlinearSolverCharacteristic.OutputTypeSet(nonlinearSolverCharacteristicsOutputType)
# Set the solver settings
NonlinearSolverCharacteristic.NewtonAbsoluteToleranceSet(absoluteToleranceNonlinearCharacteristic)
NonlinearSolverCharacteristic.NewtonSolutionToleranceSet(solutionToleranceNonlinearCharacteristic)
NonlinearSolverCharacteristic.NewtonRelativeToleranceSet(relativeToleranceNonlinearCharacteristic)
# Get the nonlinear linear solver
NonlinearSolverCharacteristic.NewtonLinearSolverGet(LinearSolverCharacteristic)
LinearSolverCharacteristic.OutputTypeSet(linearSolverCharacteristicOutputType)
# Set the solver settings
LinearSolverCharacteristic.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
LinearSolverCharacteristic.LinearIterativeMaximumIterationsSet(MAXIMUM_ITERATIONS)
LinearSolverCharacteristic.LinearIterativeDivergenceToleranceSet(DIVERGENCE_TOLERANCE)
LinearSolverCharacteristic.LinearIterativeRelativeToleranceSet(relativeToleranceLinearCharacteristic)
LinearSolverCharacteristic.LinearIterativeAbsoluteToleranceSet(absoluteToleranceLinearCharacteristic)
LinearSolverCharacteristic.LinearIterativeGMRESRestartSet(RESTART_VALUE)

#------------------

# 2nd Solver, Iterative 1D subloop - NAVIER-STOKES
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.OutputTypeSet(dynamicSolverNavierStokesOutputType)
DynamicSolverNavierStokes.DynamicThetaSet(dynamicSolverNavierStokesTheta)
# Get the dynamic nonlinear solver
DynamicSolverNavierStokes.DynamicNonlinearSolverGet(NonlinearSolverNavierStokes)
# Set the nonlinear Jacobian type
NonlinearSolverNavierStokes.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
NonlinearSolverNavierStokes.OutputTypeSet(nonlinearSolverNavierStokesOutputType)

# Set the solver settings
NonlinearSolverNavierStokes.NewtonAbsoluteToleranceSet(absoluteToleranceNonlinearNavierStokes)
NonlinearSolverNavierStokes.NewtonSolutionToleranceSet(solutionToleranceNonlinearNavierStokes)
NonlinearSolverNavierStokes.NewtonRelativeToleranceSet(relativeToleranceNonlinearNavierStokes)
# Get the dynamic nonlinear linear solver
NonlinearSolverNavierStokes.NewtonLinearSolverGet(LinearSolverNavierStokes)
LinearSolverNavierStokes.OutputTypeSet(linearSolverNavierStokesOutputType)
# Set the solver settings
LinearSolverNavierStokes.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
LinearSolverNavierStokes.LinearIterativeMaximumIterationsSet(MAXIMUM_ITERATIONS)
LinearSolverNavierStokes.LinearIterativeDivergenceToleranceSet(DIVERGENCE_TOLERANCE)
LinearSolverNavierStokes.LinearIterativeRelativeToleranceSet(relativeToleranceLinearNavierStokes)
LinearSolverNavierStokes.LinearIterativeAbsoluteToleranceSet(absoluteToleranceLinearNavierStokes)
LinearSolverNavierStokes.LinearIterativeGMRESRestartSet(RESTART_VALUE)
    
# Finish the creation of the problem solver
Problem.SolversCreateFinish()

#================================================================================================================================
#  Solver Equations
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> SOLVER EQUATIONS << == "

# Start the creation of the problem solver equations
NonlinearSolverCharacteristic = CMISS.Solver()
SolverEquationsCharacteristic = CMISS.SolverEquations()
DynamicSolverNavierStokes     = CMISS.Solver()
SolverEquationsNavierStokes   = CMISS.SolverEquations()

Problem.SolverEquationsCreateStart()

# CellML Solver
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    CellMLEquations = CMISS.CellMLEquations()
    Problem.CellMLEquationsCreateStart()
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverDAEUserNumber,CellMLSolver)
    CellMLSolver.CellMLEquationsGet(CellMLEquations)
    # Add in the equations set
    CellMLEquations.CellMLAdd(CellML)    
    Problem.CellMLEquationsCreateFinish()

#------------------

# CHARACTERISTIC solver
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
NonlinearSolverCharacteristic.SolverEquationsGet(SolverEquationsCharacteristic)
SolverEquationsCharacteristic.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetCharacteristic = SolverEquationsCharacteristic.EquationsSetAdd(EquationsSetCharacteristic)

#  NAVIER-STOKES solver
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.SolverEquationsGet(SolverEquationsNavierStokes)
SolverEquationsNavierStokes.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetNavierStokes = SolverEquationsNavierStokes.EquationsSetAdd(EquationsSetNavierStokes)

# Finish the creation of the problem solver equations
Problem.SolverEquationsCreateFinish()
    
#================================================================================================================================
#  Boundary Conditions
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> BOUNDARY CONDITIONS << == "

# CHARACTERISTIC
BoundaryConditionsCharacteristic = CMISS.BoundaryConditions()
SolverEquationsCharacteristic.BoundaryConditionsCreateStart(BoundaryConditionsCharacteristic)

# Area-outlet
versionIdx = 1
for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumbers[terminalIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A0[nodeNumber-1][0])
        elif (RCRBoundaries):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_CELLML,A[nodeNumber-1,0])
        else:
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber-1,0])

SolverEquationsCharacteristic.BoundaryConditionsCreateFinish()

#------------------

# NAVIER-STOKES
BoundaryConditionsNavierStokes = CMISS.BoundaryConditions()
SolverEquationsNavierStokes.BoundaryConditionsCreateStart(BoundaryConditionsNavierStokes)

# Inlet (Flow)
versionIdx = 1
for inputIdx in range (1,numberOfInputNodes+1):
    nodeNumber = inputNodeNumbers[inputIdx-1]
    print(nodeNumber)
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                               versionIdx,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED_FITTED,Q[nodeNumber-1,0])
# Area-outlet
versionIdx = 1
for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumbers[terminalIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumber)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_NONREFLECTING,A[nodeNumber-1,0])
        elif (RCRBoundaries):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_CELLML,A[nodeNumber-1,0])
        else:
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber-1,0])

# Finish the creation of boundary conditions
SolverEquationsNavierStokes.BoundaryConditionsCreateFinish()

if (CheckTimestepStability):
    QMax = 430.0
    maxTimestep = Utilities1D.GetMaxStableTimestep(elementNodes,QMax,nodeCoordinates,H,E,A0,Rho)
    if (timeIncrement > maxTimestep):
        sys.exit('Timestep size '+str(timeIncrement)+' above maximum allowable size of '+str(maxTimestep)+'. Please reduce step size and re-run')

#================================================================================================================================
#  Run Solvers
#================================================================================================================================

if RCRBoundaries:    
#    outputDirectory = "./output/Reymond2009ExpInputCellML_pExt70/"
    outputDirectory = "./output/Reymond2009ExpInputCellML_pExt0/"
elif nonReflecting:
#    outputDirectory = "./output/" #"./output/Reymond2009ExpInputNonreflecting/"
    if simpleTube:
        if nonReflecting:
            outputDirectory = "./output/simple_nonRef/"
        else:
            outputDirectory = "./output/simple/"
    elif reymondRefined:
        outputDirectory = "./output/Reymond2009ExpInputNonreflecting_pExt70_refine_t2/"
    else:
        outputDirectory = "./output/Reymond2009ExpInputNonreflecting_pExt70_new_t05/"
else:
    if simpleTube:
        outputDirectory = "./output/simple_fixed/"
    else:
        outputDirectory = "./output/"

# Create a results directory if needed
try:
    os.makedirs(outputDirectory)
except OSError, e:
    if e.errno != 17:
        raise   

# Solve the problem
print "Solving problem..."
start = time.time()
print("Results outputted to directory: " + outputDirectory)
with ChangeDirectory(outputDirectory):
    Problem.Solve()

end = time.time()
elapsed = end - start
print "Total Number of Elements = %d " %numberOfElements
print "Calculation Time = %3.4f" %elapsed
print "Problem solved!"

#================================================================================================================================
#  Finish Program
#================================================================================================================================
