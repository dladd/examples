#> \file
#> \author Soroush Safaei
#> \brief This is an example program to solve 1D Transient Navier-Stokes 
#>  over the arterial tree with coupled 0D lumped models (RCR) defined in CellML.
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
#> Contributor(s): David Ladd, Alys Clark
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
#> OpenCMISS/example/FluidMechanics/NavierStokes/Coupled1DCellML/Python/1DTransientExample.py
#<

#================================================================================================================================
#  Start Program
#================================================================================================================================

# Set program variables
EquationsSetFieldUserNumberNavierStokes   = 1337
EquationsSetFieldUserNumberCharacteristic = 1338
EquationsSetFieldUserNumberAdvection      = 1339

CoordinateSystemUserNumber = 1
BasisUserNumberSpace       = 1
BasisUserNumberConc        = 2
DomainUserNumber           = 1
RegionUserNumber           = 2
MeshUserNumber             = 3
DecompositionUserNumber    = 4
GeometricFieldUserNumber   = 5
DependentFieldUserNumber   = 6
DependentFieldUserNumber2  = 7
MaterialsFieldUserNumber   = 8
MaterialsFieldUserNumber2  = 9
IndependentFieldUserNumber = 10
EquationsSetUserNumberNavierStokes   = 11
EquationsSetUserNumberCharacteristic = 12
EquationsSetUserNumberAdvection      = 13
ProblemUserNumber                    = 14
CellMLUserNumber                     = 15
CellMLModelsFieldUserNumber          = 16
CellMLStateFieldUserNumber           = 17
CellMLIntermediateFieldUserNumber    = 18
CellMLParametersFieldUserNumber      = 19
MaterialsFieldUserNumberCellML       = 20
AnalyticFieldUserNumber              = 21

SolverDAEUserNumber            = 1
SolverCharacteristicUserNumber = 2
SolverNavierStokesUserNumber   = 3
SolverAdvectionUserNumber      = 4

# Materials constants
MaterialsFieldUserNumberMu  = 1
MaterialsFieldUserNumberRho = 2
MaterialsFieldUserNumberAlpha=3
MaterialsFieldUserNumberPext= 4
MaterialsFieldUserNumberLs  = 5
MaterialsFieldUserNumberTs  = 6
MaterialsFieldUserNumberMs  = 7
# Materials variables
MaterialsFieldUserNumberA0  = 1
MaterialsFieldUserNumberE   = 2
MaterialsFieldUserNumberH   = 3

#================================================================================================================================
#  Initialise OpenCMISS
#================================================================================================================================

# Import the libraries (OpenCMISS,python,numpy,scipy)
import numpy,math
from scipy.sparse import linalg
from scipy.linalg import inv
from scipy.linalg import eig
from opencmiss import CMISS
import csv,time
import sys,os
import re
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

# Get the computational nodes info
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber    = CMISS.ComputationalNodeNumberGet()

#================================================================================================================================
#  Problem Control Panel
#================================================================================================================================

numberOfDimensions    = 1  #(One-dimensional)
numberOfComponents    = 2  #(Flow & Area)
numberOfBifurcations  = 0
numberOfTrifurcations  = 0
numberOfTerminalNodes = 0
numberOfInputNodes    = 0
bifurcationNodeNumber     = []
bifurcationElementNumber  = []
bifurcationNodeNumber.append('null')
bifurcationElementNumber.append('null')
trifurcationNodeNumber     = []
trifurcationElementNumber  = []
trifurcationNodeNumber.append('null')
trifurcationElementNumber.append('null')

# Set the user number
derivIdx   = 1
versionIdx = 1

# Set to use coupled 0D Windkessel models (from CellML) at model outlet boundaries
RCRBoundaries = True 
# Set to use non-reflecting outlet boundaries
nonReflecting = False
# Set to solve a coupled advection problem
coupledAdvection = False
# Set to use spline interpolation of tabulated data from file as inlet flow boundary condition
splineInterpolatedFlowrate = True
# Set to do a basic check of the stability of the hyperbolic problem based on the timestep size
checkTimestepStability     = True
# Set to do a postprocessing check on the amplification matrix
analysisFlag = False
# Set to initialise values
initialiseFromFile = True

#================================================================================================================================
#  Mesh Reading
#================================================================================================================================
# Read the node file
with open('Input/Node.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        # Read the number of nodes
        if int(row[0]) == 1:
            numberOfNodesSpace = int(row[5])
            totalNumberOfNodes = numberOfNodesSpace*3
            xValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)
            yValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)
            zValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)            
        # Initialise the coordinates
        xValues[int(row[0])][0] = float(row[1])
        yValues[int(row[0])][0] = float(row[2])
        zValues[int(row[0])][0] = float(row[3])
        # Read the bifurcation nodes
        if row[4] == 'bif':
            numberOfBifurcations+=1
            bifurcationNodeNumber.append(int(row[0]))
            xValues[int(row[0])][1] = float(row[1])
            yValues[int(row[0])][1] = float(row[2])
            zValues[int(row[0])][1] = float(row[3])
            xValues[int(row[0])][2] = float(row[1])
            yValues[int(row[0])][2] = float(row[2])
            zValues[int(row[0])][2] = float(row[3])
        # Read the trifurcation nodes
        elif row[5] == 'trif':
            numberOfTrifurcations+=1
            trifurcationNodeNumber.append(int(row[0]))
            xValues[int(row[0])][1] = float(row[1])
            yValues[int(row[0])][1] = float(row[2])
            zValues[int(row[0])][1] = float(row[3])
            xValues[int(row[0])][2] = float(row[1])
            yValues[int(row[0])][2] = float(row[2])
            zValues[int(row[0])][2] = float(row[3])
            xValues[int(row[0])][3] = float(row[1])
            yValues[int(row[0])][3] = float(row[2])
            zValues[int(row[0])][3] = float(row[3])

# Read the element file
with open('Input/Element.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    i = 0
    j = 0
    for row in reader:
        # Read the number of elements
        if int(row[0]) == 1:
            totalNumberOfElements = int(row[len(row)-1])
            elementNodes = (totalNumberOfElements+1)*[3*[0]]
            bifurcationElements = (numberOfBifurcations+1)*[3*[0]]
            trifurcationElements = (numberOfTrifurcations+1)*[4*[0]]
        # Read the element nodes
        elementNodes[int(row[0])] = [int(row[1]),int(row[2]),int(row[3])]
        # Read the bifurcation elements
        if row[4]:
            i+=1
            bifurcationElements[i] = [int(row[4]),int(row[5]),int(row[6])]
        # Read the trifurcation elements
        elif row[7]:
            j+=1
            trifurcationElements[j] = [int(row[7]),int(row[8]),int(row[9]),int(row[10])]



#================================================================================================================================
#  Initial Data & Default Values
#================================================================================================================================

# Set the material parameters
Rho = 1050.0                      # Rho         (kg/m3)
Mu  = 0.004                       # Mu          (Pa.s)
pressureExternal = 0.0#10665.7895     # External pressure (Pa, or 80mmHg)
A0  = numpy.zeros((numberOfNodesSpace+1,4))  # Area        (m2)
H   = numpy.zeros((numberOfNodesSpace+1,4))  # Thickness   (m)
E   = numpy.zeros((numberOfNodesSpace+1,4))  # Elasticity  (Pa)
dt  = [0]*(numberOfNodesSpace+1)  # TimeStep    (s)
eig = [0]*(numberOfNodesSpace+1)  # Eigenvalues

# Material parameter scaling factors
Ls = 1000.0                       # Length   (m -> mm)
Ts = 1000.0                       # Time     (s -> ms)
Ms = 1000.0                       # Mass     (kg -> g)

Alpha  = 1.0                     # Flow profile     (non-dimensional)
Qs = (Ls**3.0)/Ts                # Flow             (m3/s)      -->  mm3/ms
As = Ls**2.0                     # Area             (m2)        -->  mm2
Hs = Ls                          # vessel thickness (m)         -->  mm
Es = Ms/(Ls*Ts**2.0)             # Elasticity       (kg/(m.s2)  -->  g/(mm.ms^2)
Rhos = Ms/(Ls**3.0)              # Density          (kg/m3)     -->  g/mm3
Mus = Ms/(Ls*Ts)                 # Viscosity        (kg/(m.s))  -->  g/(mm.ms)
Ps = Ms/(Ls*Ts**2.0)             # Pressure         (kg/(m.s2)) -->  g/(mm.ms2)

# Read the MATERIAL file
inputNodeNumber  = [0]
coupledNodeNumber  = [0]
with open('Input/Material.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        A0[int(row[0])][0] = float(row[1])
        E [int(row[0])][0] = float(row[2])
        H [int(row[0])][0] = float(row[3])
        if row[4]:
            coupledNodeNumber.append(int(row[4]))
            numberOfTerminalNodes = numberOfTerminalNodes+1
        if row[5]:
            inputNodeNumber.append(int(row[5]))
            numberOfInputNodes = numberOfInputNodes+1

Rho = Rho*Rhos
Mu = Mu*Mus
pressureExternal = pressureExternal*Ps
A0 = A0*As
E = E*Es
H = H*Hs

Q = numpy.zeros([numberOfNodesSpace+1,4])
A = numpy.zeros([numberOfNodesSpace+1,4])
dQ = numpy.zeros([numberOfNodesSpace+1,4])
dA = numpy.zeros([numberOfNodesSpace+1,4])

# Set A0 for branch nodes
for bifIdx in range(1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx]
    for versionIdx in range(1,3):
        A0[nodeIdx][versionIdx] = A0[elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
        E[nodeIdx][versionIdx] = E[elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
        H[nodeIdx][versionIdx] = H[elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
for trifIdx in range(1,numberOfTrifurcations+1):
    nodeIdx = trifurcationNodeNumber[trifIdx]
    for versionIdx in range(1,4):
        A0[nodeIdx][versionIdx] = A0[elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]
        E[nodeIdx][versionIdx] = E[elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]
        H[nodeIdx][versionIdx] = H[elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]

# Start with Q=0, A=A0 state
A = A0

# Or initialise from init file
if initialiseFromFile:
    init = numpy.zeros([numberOfNodesSpace+1,4,4])
    init = numpy.load('./Input/init.npy')
    Q[1:numberOfNodesSpace+1,:] = init[:,0,:]
    A[1:numberOfNodesSpace+1,:] = init[:,1,:]
    dQ[1:numberOfNodesSpace+1,:] = init[:,2,:]
    dA[1:numberOfNodesSpace+1,:] = init[:,3,:]

# Set the output parameters
# (NONE/PROGRESS/TIMING/SOLVER/MATRIX)
DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE   = CMISS.SolverOutputTypes.NONE
NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
NONLINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
LINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE    = CMISS.SolverOutputTypes.NONE
# (NONE/TIMING/SOLVER/MATRIX)
CMISS_SOLVER_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY = 10

# Set the time parameters
DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME     = 0.0
DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME      = 4400.0001 #2900.0001 #995.0001 #1000.00001
DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT = 0.1
DYNAMIC_SOLVER_NAVIER_STOKES_THETA = [1.0]
DYNAMIC_SOLVER_ADVECTION_THETA     = [0.5]

# Set the solver parameters
relativeToleranceNonlinearNavierStokes = 1.0E-05   # default: 1.0E-05
absoluteToleranceNonlinearNavierStokes = 1.0E-10   # default: 1.0E-10
solutionToleranceNonlinearNavierStokes = 1.0E-05  # default: 1.0E-05
relativeToleranceLinearNavierStokes = 1.0E-05   # default: 1.0E-05
absoluteToleranceLinearNavierStokes = 1.0E-10  # default: 1.0E-10

relativeToleranceNonlinearCharacteristic = 1.0E-05 # default: 1.0E-05
absoluteToleranceNonlinearCharacteristic = 1.0E-10  # default: 1.0E-10
solutionToleranceNonlinearCharacteristic = 1.0E-05  # default: 1.0E-05
relativeToleranceLinearCharacteristic = 1.0E-05  # default: 1.0E-05
absoluteToleranceLinearCharacteristic = 1.0E-10  # default: 1.0E-10

DIVERGENCE_TOLERANCE = 1.0E+10  # default: 1.0E+05
MAXIMUM_ITERATIONS   = 100000   # default: 100000
RESTART_VALUE        = 3000     # default: 30

# N-S/C coupling tolerance
couplingTolerance1D = 1.0e6
# 1D-0D coupling tolerance
couplingTolerance1D0D = 0.001

# Check the CellML flag
if (RCRBoundaries):
    if (coupledAdvection):
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.Coupled1D0DAdv_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        # Advection solver
        EquationsSetAdvectionSubtype = CMISS.EquationsSetSubtypes.ADVECTION
        ProblemSubtype = CMISS.ProblemSubTypes.Coupled1D0DAdv_NAVIER_STOKES
    else:
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        ProblemSubtype = CMISS.ProblemSubTypes.Coupled1D0D_NAVIER_STOKES
else:
    if (coupledAdvection):
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.OnedTransientAdv_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        # Advection solver
        EquationsSetAdvectionSubtype = CMISS.EquationsSetSubtypes.ADVECTION
        ProblemSubtype = CMISS.ProblemSubTypes.OnedTransientAdv_NAVIER_STOKES
    else:
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.OnedTransient_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        ProblemSubtype = CMISS.ProblemSubTypes.OnedTransient_NAVIER_STOKES

#================================================================================================================================
#  Coordinate System
#================================================================================================================================

# Start the creation of a new RC coordinate system
CoordinateSystem = CMISS.CoordinateSystem()
CoordinateSystem.CreateStart(CoordinateSystemUserNumber)
CoordinateSystem.Dimension = 3
CoordinateSystem.CreateFinish()

#================================================================================================================================
#  Region
#================================================================================================================================

# Start the creation of a new region
Region = CMISS.Region()
Region.CreateStart(RegionUserNumber,CMISS.WorldRegion)
Region.label = "OpenCMISS"
Region.coordinateSystem = CoordinateSystem
Region.CreateFinish()

#================================================================================================================================
#  Bases
#================================================================================================================================
basisXiGaussSpace = 3
# Start the creation of SPACE bases
BasisSpace = CMISS.Basis()
BasisSpace.CreateStart(BasisUserNumberSpace)
BasisSpace.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
BasisSpace.numberOfXi = numberOfDimensions
BasisSpace.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]
BasisSpace.quadratureNumberOfGaussXi = [basisXiGaussSpace]
BasisSpace.CreateFinish()

#------------------

basisXiGaussConc  = 2
# Start the creation of CONCENTRATION bases
BasisConc = CMISS.Basis()
BasisConc.CreateStart(BasisUserNumberConc)
BasisConc.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
BasisConc.numberOfXi = numberOfDimensions
BasisConc.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]
BasisConc.quadratureNumberOfGaussXi = [basisXiGaussConc]
BasisConc.CreateFinish()

#================================================================================================================================
#  Mesh
#================================================================================================================================

meshNumberOfComponents = 2
# Start the creation of mesh nodes
Nodes = CMISS.Nodes()
Nodes.CreateStart(Region,totalNumberOfNodes)
Nodes.CreateFinish()
# Start the creation of mesh
Mesh = CMISS.Mesh()
Mesh.CreateStart(MeshUserNumber,Region,numberOfDimensions)
Mesh.NumberOfElementsSet(totalNumberOfElements)
Mesh.NumberOfComponentsSet(meshNumberOfComponents)
# Specify the mesh components
MeshElementsSpace = CMISS.MeshElements()
MeshElementsConc  = CMISS.MeshElements()
meshComponentNumberSpace = 1
meshComponentNumberConc  = 2

#------------------

# Specify the SPACE mesh component
print('Bifurcations at nodes: ' + str(bifurcationNodeNumber))
print('Trifurcations at nodes: ' + str(trifurcationNodeNumber))
MeshElementsSpace.CreateStart(Mesh,meshComponentNumberSpace,BasisSpace)
for elemIdx in range(1,totalNumberOfElements+1):
    MeshElementsSpace.NodesSet(elemIdx,elementNodes[elemIdx])
for bifIdx in range(1,numberOfBifurcations+1):
    MeshElementsSpace.LocalElementNodeVersionSet(int(bifurcationElements[bifIdx][0]),1,1,3)
    MeshElementsSpace.LocalElementNodeVersionSet(int(bifurcationElements[bifIdx][1]),2,1,1) 
    MeshElementsSpace.LocalElementNodeVersionSet(int(bifurcationElements[bifIdx][2]),3,1,1) 
for trifIdx in range(1,numberOfTrifurcations+1):
    MeshElementsSpace.LocalElementNodeVersionSet(int(trifurcationElements[trifIdx][0]),1,1,3)
    MeshElementsSpace.LocalElementNodeVersionSet(int(trifurcationElements[trifIdx][1]),2,1,1) 
    MeshElementsSpace.LocalElementNodeVersionSet(int(trifurcationElements[trifIdx][2]),3,1,1) 
    MeshElementsSpace.LocalElementNodeVersionSet(int(trifurcationElements[trifIdx][3]),4,1,1) 
MeshElementsSpace.CreateFinish()                        

#------------------

# Specify the CONCENTRATION mesh component
MeshElementsConc.CreateStart(Mesh,meshComponentNumberConc,BasisSpace)
for elemIdx in range(1,totalNumberOfElements+1):
    MeshElementsConc.NodesSet(elemIdx,elementNodes[elemIdx])
MeshElementsConc.CreateFinish()  

# Finish the creation of the mesh
Mesh.CreateFinish()

#================================================================================================================================
#  Decomposition
#================================================================================================================================

# Start the creation of a new decomposition
Decomposition = CMISS.Decomposition()
Decomposition.CreateStart(DecompositionUserNumber,Mesh)
Decomposition.TypeSet(CMISS.DecompositionTypes.CALCULATED)
Decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
Decomposition.CreateFinish()

#================================================================================================================================
#  Geometric Field
#================================================================================================================================

# Start the creation of a new geometric field
GeometricField = CMISS.Field()
GeometricField.CreateStart(GeometricFieldUserNumber,Region)
GeometricField.NumberOfVariablesSet(1)
GeometricField.VariableLabelSet(CMISS.FieldVariableTypes.U,'Coordinates')
GeometricField.TypeSet = CMISS.FieldTypes.GEOMETRIC
GeometricField.meshDecomposition = Decomposition
GeometricField.ScalingTypeSet = CMISS.FieldScalingTypes.NONE
# Set the mesh component to be used by the geometric field components
for componentNumber in range(1,CoordinateSystem.dimension+1):
    GeometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,componentNumber,meshComponentNumberSpace)
GeometricField.CreateFinish()

# Set the geometric field values for version 1
versionIdx = 1
for nodeIdx in range(1,numberOfNodesSpace+1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,derivIdx,nodeIdx,1,xValues[nodeIdx][0])
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,derivIdx,nodeIdx,2,yValues[nodeIdx][0])
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                versionIdx,derivIdx,nodeIdx,3,zValues[nodeIdx][0])

# Set the geometric field for bifurcation
for bifIdx in range (1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        for versionNumber in range(2,4):
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,1,xValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,2,yValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,3,zValues[nodeIdx][versionNumber-1])
# Set the geometric field for trifurcation
for trifIdx in range (1,numberOfTrifurcations+1):
    nodeIdx = trifurcationNodeNumber[trifIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        for versionNumber in range(2,5):
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,1,xValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,2,yValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,3,zValues[nodeIdx][versionNumber-1])

# Finish the parameter update
GeometricField.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
GeometricField.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)     

#================================================================================================================================
#  Equations Sets
#================================================================================================================================

print('creating char field')
# Create the equations set for CHARACTERISTIC
EquationsSetCharacteristic = CMISS.EquationsSet()
EquationsSetFieldCharacteristic = CMISS.Field()
# Set the equations set to be a static nonlinear problem
EquationsSetCharacteristic.CreateStart(EquationsSetUserNumberCharacteristic,Region,GeometricField,
    CMISS.EquationsSetClasses.FLUID_MECHANICS,CMISS.EquationsSetTypes.CHARACTERISTIC_EQUATION,
     EquationsSetCharacteristicSubtype,EquationsSetFieldUserNumberCharacteristic,EquationsSetFieldCharacteristic)
EquationsSetCharacteristic.CreateFinish()

print('creating ns field')
# Create the equations set for NAVIER-STOKES
EquationsSetNavierStokes = CMISS.EquationsSet()
EquationsSetFieldNavierStokes = CMISS.Field()
# Set the equations set to be a dynamic nonlinear problem
EquationsSetNavierStokes.CreateStart(EquationsSetUserNumberNavierStokes,Region,GeometricField,
    CMISS.EquationsSetClasses.FLUID_MECHANICS,CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
     EquationsSetSubtype,EquationsSetFieldUserNumberNavierStokes,EquationsSetFieldNavierStokes)
EquationsSetNavierStokes.CreateFinish()

#------------------
if (coupledAdvection):
    # Create the equations set for ADVECTION
    EquationsSetAdvection = CMISS.EquationsSet()
    EquationsSetFieldAdvection = CMISS.Field()
    # Set the equations set to be a dynamic linear problem
    EquationsSetAdvection.CreateStart(EquationsSetUserNumberAdvection,Region,GeometricField,
        CMISS.EquationsSetClasses.CLASSICAL_FIELD,CMISS.EquationsSetTypes.ADVECTION_EQUATION,
         EquationsSetAdvectionSubtype,EquationsSetFieldUserNumberAdvection,EquationsSetFieldAdvection)
    EquationsSetAdvection.CreateFinish()

#================================================================================================================================
#  Dependent Field
#================================================================================================================================

# Create the equations set dependent field variables
DependentFieldNavierStokes = CMISS.Field()
DependentFieldAdvection    = CMISS.Field()

# CHARACTERISTIC
EquationsSetCharacteristic.DependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'General')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Derivatives')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.V,'Characteristics')
if RCRBoundaries:
    DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U1,'CellML Q and P')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U2,'Pressure')
# Set the mesh component to be used by the field components.
# Flow & Area
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumberSpace)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumberSpace)
# Derivatives
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,meshComponentNumberSpace)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,2,meshComponentNumberSpace)
# Riemann
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,1,meshComponentNumberSpace)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,2,meshComponentNumberSpace)
# qCellML & pCellml
if RCRBoundaries:
    DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,1,meshComponentNumberSpace)
    DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U1,2,meshComponentNumberSpace)
# Pressure
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,1,meshComponentNumberSpace)
DependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U2,2,meshComponentNumberSpace)

EquationsSetCharacteristic.DependentCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.DependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
EquationsSetNavierStokes.DependentCreateFinish()

DependentFieldNavierStokes.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES)

# Initialise the dependent field variables
for nodeIdx in range (1,numberOfNodesSpace+1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        if nodeIdx in trifurcationNodeNumber:
            versions = [1,2,3,4]
        elif nodeIdx in bifurcationNodeNumber:
            versions = [1,2,3]
        else:
            versions = [1]
        for versionIdx in versions:
            # U variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,1,Q[nodeIdx,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,2,A[nodeIdx,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                                versionIdx,derivIdx,nodeIdx,1,Q[nodeIdx,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                                versionIdx,derivIdx,nodeIdx,2,A[nodeIdx,versionIdx-1])
            # delUdelN variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,1,dQ[nodeIdx,versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,2,dA[nodeIdx,versionIdx-1])

# revert default version to 1
versionIdx = 1

# Finish the parameter update
DependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
DependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)   

#------------------
if (coupledAdvection):
    # ADVECTION
    EquationsSetAdvection.DependentCreateStart(DependentFieldUserNumber2,DependentFieldAdvection)
    DependentFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.U,'Concentration')
    DependentFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Deriv')
    # Set the mesh component to be used by the field components.
    DependentFieldAdvection.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumberConc)
    DependentFieldAdvection.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,meshComponentNumberConc)
    EquationsSetAdvection.DependentCreateFinish()

    # Initialise the dependent field variables
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeIdx = inputNodeNumber[inputIdx]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberConc)
        if nodeDomain == computationalNodeNumber:
            DependentFieldAdvection.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,Conc)

    # Finish the parameter update
    DependentFieldAdvection.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
    DependentFieldAdvection.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES) 

#================================================================================================================================
#  Materials Field
#================================================================================================================================

# Create the equations set materials field variables 
MaterialsFieldNavierStokes = CMISS.Field()
MaterialsFieldAdvection    = CMISS.Field()

# CHARACTERISTIC
EquationsSetCharacteristic.MaterialsCreateStart(MaterialsFieldUserNumber,MaterialsFieldNavierStokes)
MaterialsFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'MaterialsConstants')
MaterialsFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.V,'MaterialsVariables')
# Set the mesh component to be used by the field components.
for componentNumber in range(1,4):
    MaterialsFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.V,componentNumber,meshComponentNumberSpace)
EquationsSetCharacteristic.MaterialsCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.MaterialsCreateStart(MaterialsFieldUserNumber,MaterialsFieldNavierStokes)
EquationsSetNavierStokes.MaterialsCreateFinish()

print("Density: " + str(Rho))
print("Viscosity: " + str(Mu))
# Set the materials field constants
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberMu,Mu)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberRho,Rho)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberAlpha,Alpha)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberPext,pressureExternal)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberLs,Ls)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
                                                       MaterialsFieldUserNumberTs,Ts)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                       MaterialsFieldUserNumberMs,Ms)

# Initialise the materials field variables (A0,E,H)
bifIdx = 0
trifIdx = 0
for nodeIdx in range(1,numberOfNodesSpace+1,1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        if nodeIdx in trifurcationNodeNumber:
            versions = [1,2,3,4]
        elif nodeIdx in bifurcationNodeNumber:
            versions = [1,2,3]
        else:
            versions = [1]
        for versionIdx in versions:
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,MaterialsFieldUserNumberA0,A0[nodeIdx][versionIdx-1])
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,MaterialsFieldUserNumberE,E[nodeIdx][versionIdx-1])
            MaterialsFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES,
                                                                versionIdx,derivIdx,nodeIdx,MaterialsFieldUserNumberH,H[nodeIdx][versionIdx-1])

# Finish the parameter update
MaterialsFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)
MaterialsFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.V,CMISS.FieldParameterSetTypes.VALUES)

#------------------
if (coupledAdvection):
    # ADVECTION
    EquationsSetAdvection.MaterialsCreateStart(MaterialsFieldUserNumber2,MaterialsFieldAdvection)
    MaterialsFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.U,'Materials')
    EquationsSetAdvection.MaterialsCreateFinish()
    # Set the materials field constant
    MaterialsFieldAdvection.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
        MaterialsFieldUserNumberD,D)

#================================================================================================================================
# Independent Field
#================================================================================================================================

# Create the equations set independent field variables  
IndependentFieldNavierStokes = CMISS.Field()
IndependentFieldAdvection    = CMISS.Field()

# CHARACTERISTIC
EquationsSetCharacteristic.IndependentCreateStart(IndependentFieldUserNumber,IndependentFieldNavierStokes)
IndependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Normal Wave Direction')
# Set the mesh component to be used by the field components.
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumberSpace)
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumberSpace)
EquationsSetCharacteristic.IndependentCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.IndependentCreateStart(IndependentFieldUserNumber,IndependentFieldNavierStokes)
EquationsSetNavierStokes.IndependentCreateFinish()

# Set the normal wave direction for bifurcation
for bifIdx in range (1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        # Incoming(parent)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         1,derivIdx,nodeIdx,1,1.0)
        # Outgoing(branches)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         2,derivIdx,nodeIdx,2,-1.0)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         3,derivIdx,nodeIdx,2,-1.0)
# Set the normal wave direction for trifurcation
for trifIdx in range (1,numberOfTrifurcations+1):
    nodeIdx = trifurcationNodeNumber[trifIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        # Incoming(parent)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         1,derivIdx,nodeIdx,1,1.0)
        # Outgoing(branches)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         2,derivIdx,nodeIdx,2,-1.0)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         3,derivIdx,nodeIdx,2,-1.0)
        IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
         4,derivIdx,nodeIdx,2,-1.0)


# Set the normal wave direction for terminal
if (RCRBoundaries or nonReflecting):
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if nodeDomain == computationalNodeNumber:
            # Incoming
            IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,1.0)

# Finish the parameter update
IndependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
IndependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

#------------------
if (coupledAdvection):
    # ADVECTION
    EquationsSetAdvection.IndependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
    EquationsSetAdvection.IndependentCreateFinish()

#================================================================================================================================
# Analytic Field
#================================================================================================================================

if (splineInterpolatedFlowrate):
    AnalyticFieldNavierStokes = CMISS.Field()
    # FlowrateReymonds,FlowrateOlufsen,FlowrateSheffield,FlowrateAorta
    #EquationsSetNavierStokes.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.FlowrateOlufsen,AnalyticFieldUserNumber,
    # AnalyticFieldNavierStokes)
    EquationsSetNavierStokes.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.SplintFromFile,AnalyticFieldUserNumber,
                                                 AnalyticFieldNavierStokes)
    AnalyticFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Spline interpolated flow rate')
    #AnalyticFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Analytic inlet flow rate')
    EquationsSetNavierStokes.AnalyticCreateFinish()
    AnalyticFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1000.0)


# DOC-START cellml define field maps
#================================================================================================================================
#  CellML Model Maps
#================================================================================================================================

if (RCRBoundaries):

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

    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,Region)
    # Number of CellML models
    CellMLModelIndex = [0]*(numberOfTerminalNodes+1)

    # Windkessel Model
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx]
        # Veins output
        if (nodeIdx%2 == 0):
            nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
            if nodeDomain == computationalNodeNumber:
                CellMLModelIndex[terminalIdx] = CellML.ModelImport("./Input/CellMLModels/"+str(terminalIdx)+"/ModelHeart.cellml")
                # known (to OpenCMISS) variables
                CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Heart/Qi")
                CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Heart/Po")
                # to get from the CellML side 
                CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Heart/Qo")
                CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Heart/Pi")
        # Arteries outlet
        else:
            print('reading model: ' + "./Input/CellMLModels/"+str(terminalIdx)+"/ModelRCR.cellml")
            nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
            if nodeDomain == computationalNodeNumber:
                CellMLModelIndex[terminalIdx] = CellML.ModelImport("./Input/CellMLModels/"+str(terminalIdx)+"/ModelRCR.cellml")
                # known (to OpenCMISS) variables
                CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Circuit/Qin")
                # to get from the CellML side 
                CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Circuit/Pout")
    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx]
        # Veins output
        if (nodeIdx%2 == 0):
            nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
            if nodeDomain == computationalNodeNumber:
                # Now we can set up the field variable component <--> CellML model variable mappings.
                # Map the OpenCMISS boundary flow rate values --> CellML
                # Q is component 1 of the DependentField
                CellML.CreateFieldToCellMLMap(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,1,
                 CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[terminalIdx],"Heart/Qi",CMISS.FieldParameterSetTypes.VALUES)
                CellML.CreateFieldToCellMLMap(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U2,1,
                 CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[terminalIdx],"Heart/Po",CMISS.FieldParameterSetTypes.VALUES)
                # Map the returned pressure values from CellML --> CMISS
                # pCellML is component 2 of the Dependent field U1 variable
                CellML.CreateCellMLToFieldMap(CellMLModelIndex[terminalIdx],"Heart/Qo",CMISS.FieldParameterSetTypes.VALUES,
                 DependentFieldNavierStokes,CMISS.FieldVariableTypes.U1,qCellMLComponent,CMISS.FieldParameterSetTypes.VALUES)
                CellML.CreateCellMLToFieldMap(CellMLModelIndex[terminalIdx],"Heart/Pi",CMISS.FieldParameterSetTypes.VALUES,
                 DependentFieldNavierStokes,CMISS.FieldVariableTypes.U1,pCellMLComponent,CMISS.FieldParameterSetTypes.VALUES)
        # Arteries output
        else:
            nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
            if nodeDomain == computationalNodeNumber:
                # Now we can set up the field variable component <--> CellML model variable mappings.
                # Map the OpenCMISS boundary flow rate values --> CellML
                # Q is component 1 of the DependentField
                CellML.CreateFieldToCellMLMap(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,1,
                                              CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[terminalIdx],"Circuit/Qin",CMISS.FieldParameterSetTypes.VALUES)
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
        nodeIdx = coupledNodeNumber[terminalIdx]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if nodeDomain == computationalNodeNumber:
            CellMLModelsField.ParameterSetUpdateNode(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,CellMLModelIndex[terminalIdx])

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

# 1st Equations Set - CHARACTERISTIC
EquationsCharacteristic = CMISS.Equations()
EquationsSetCharacteristic.EquationsCreateStart(EquationsCharacteristic)
EquationsCharacteristic.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
EquationsCharacteristic.outputType = CMISS.EquationsOutputTypes.NONE
EquationsSetCharacteristic.EquationsCreateFinish()

#------------------

# 2nd Equations Set - NAVIER-STOKES
EquationsNavierStokes = CMISS.Equations()
EquationsSetNavierStokes.EquationsCreateStart(EquationsNavierStokes)
EquationsNavierStokes.sparsityType = CMISS.EquationsSparsityTypes.FULL
EquationsNavierStokes.lumpingType = CMISS.EquationsLumpingTypes.UNLUMPED
# (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
EquationsNavierStokes.outputType = CMISS.EquationsOutputTypes.NONE
EquationsSetNavierStokes.EquationsCreateFinish()

#------------------
if (coupledAdvection):
    # 3rd Equations Set - ADVECTION
    EquationsAdvection = CMISS.Equations()
    EquationsSetAdvection.EquationsCreateStart(EquationsAdvection)
    EquationsAdvection.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
    # (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
    EquationsAdvection.outputType = CMISS.EquationsOutputTypes.NONE
    EquationsSetAdvection.EquationsCreateFinish()

#================================================================================================================================
#  Problems
#================================================================================================================================

# Start the creation of a problem.
Problem = CMISS.Problem()
Problem.CreateStart(ProblemUserNumber)
Problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,ProblemSubtype)    
Problem.CreateFinish()

#================================================================================================================================
#  Control Loops
#================================================================================================================================
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
                   | 2) (optional) Simple subloop     | 1) Advection Linear Solver


1D
------
              

    Time Loop, L0  | 1) 1D NS/C coupling subloop      | 1) Characteristic Nonlinear Solver
                   |    (while loop)                  | 2) 1DNavierStokes Transient Solver
                   |
                   | 2) (optional) Simple subloop     | 1) Advection Linear Solver


'''

# Order of solvers within their respective subloops
SolverCharacteristicUserNumber = 1
SolverNavierStokesUserNumber   = 2
SolverAdvectionUserNumber      = 1
SolverCellmlUserNumber         = 1
if (RCRBoundaries):
   Iterative1d0dControlLoopNumber = 1
   SimpleAdvectionControlLoopNumber = 2
   Simple0DControlLoopNumber = 1
   Iterative1dControlLoopNumber = 2
else:
   Iterative1dControlLoopNumber = 1
   SimpleAdvectionControlLoopNumber = 2

# Start the creation of the problem control loop
TimeLoop = CMISS.ControlLoop()
Problem.ControlLoopCreateStart()
Problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],TimeLoop)
TimeLoop.LabelSet('Time Loop')
TimeLoop.TimesSet(DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME,DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME,  
                  DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT)
TimeLoop.TimeOutputSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY)

# Set tolerances for iterative convergence loops
if(RCRBoundaries):
    Iterative1DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],Iterative1DCouplingLoop)
    Iterative1DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D)
    Iterative1D0DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1d0dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],Iterative1D0DCouplingLoop)
    Iterative1D0DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D0D)
else:
    Iterative1DCouplingLoop = CMISS.ControlLoop()
    Problem.ControlLoopGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],Iterative1DCouplingLoop)
    Iterative1DCouplingLoop.AbsoluteToleranceSet(couplingTolerance1D)

Problem.ControlLoopCreateFinish()

#================================================================================================================================
#  Solvers
#================================================================================================================================

# Start the creation of the problem solvers
DynamicSolverNavierStokes     = CMISS.Solver()
NonlinearSolverNavierStokes   = CMISS.Solver()
LinearSolverNavierStokes      = CMISS.Solver()
NonlinearSolverCharacteristic = CMISS.Solver()
LinearSolverCharacteristic    = CMISS.Solver()
DynamicSolverAdvection        = CMISS.Solver()
LinearSolverAdvection         = CMISS.Solver()

Problem.SolversCreateStart()

#------------------

# 1st Solver, Simple 0D subloop - CellML
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverDAEUserNumber,CellMLSolver)
    CellMLSolver.OutputTypeSet(CMISS_SOLVER_OUTPUT_TYPE)

#------------------

# 1st Solver, Iterative 1D subloop - CHARACTERISTIC
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)

# Set the nonlinear Jacobian type
NonlinearSolverCharacteristic.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
NonlinearSolverCharacteristic.OutputTypeSet(NONLINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE)
# Set the solver settings
NonlinearSolverCharacteristic.NewtonAbsoluteToleranceSet(absoluteToleranceNonlinearCharacteristic)
NonlinearSolverCharacteristic.NewtonSolutionToleranceSet(solutionToleranceNonlinearCharacteristic)
NonlinearSolverCharacteristic.NewtonRelativeToleranceSet(relativeToleranceNonlinearCharacteristic)
# Get the nonlinear linear solver
NonlinearSolverCharacteristic.NewtonLinearSolverGet(LinearSolverCharacteristic)
LinearSolverCharacteristic.OutputTypeSet(LINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE)
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
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.OutputTypeSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
DynamicSolverNavierStokes.DynamicThetaSet(DYNAMIC_SOLVER_NAVIER_STOKES_THETA)
# Get the dynamic nonlinear solver
DynamicSolverNavierStokes.DynamicNonlinearSolverGet(NonlinearSolverNavierStokes)
# Set the nonlinear Jacobian type
NonlinearSolverNavierStokes.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
NonlinearSolverNavierStokes.OutputTypeSet(NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
#NonlinearSolverNavierStokes.NewtonConvergenceTestTypeSet(CMISS.NewtonConvergenceTypes.PETSC_DEFAULT) #PETSC_DEFAULT/ENERGY_NORM/DIFFERENTIATED_RATIO

# Set the solver settings
NonlinearSolverNavierStokes.NewtonAbsoluteToleranceSet(absoluteToleranceNonlinearNavierStokes)
NonlinearSolverNavierStokes.NewtonSolutionToleranceSet(solutionToleranceNonlinearNavierStokes)
NonlinearSolverNavierStokes.NewtonRelativeToleranceSet(relativeToleranceNonlinearNavierStokes)
# Get the dynamic nonlinear linear solver
NonlinearSolverNavierStokes.NewtonLinearSolverGet(LinearSolverNavierStokes)
LinearSolverNavierStokes.OutputTypeSet(LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
# Set the solver settings
LinearSolverNavierStokes.LinearTypeSet(CMISS.LinearSolverTypes.ITERATIVE)
LinearSolverNavierStokes.LinearIterativeMaximumIterationsSet(MAXIMUM_ITERATIONS)
LinearSolverNavierStokes.LinearIterativeDivergenceToleranceSet(DIVERGENCE_TOLERANCE)
LinearSolverNavierStokes.LinearIterativeRelativeToleranceSet(relativeToleranceLinearNavierStokes)
LinearSolverNavierStokes.LinearIterativeAbsoluteToleranceSet(absoluteToleranceLinearNavierStokes)
LinearSolverNavierStokes.LinearIterativeGMRESRestartSet(RESTART_VALUE)
    
#------------------
if (coupledAdvection):
    # 1st Solver, Simple advection subloop - ADVECTION
    Problem.SolverGet([SimpleAdvectionControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverAdvectionUserNumber,DynamicSolverAdvection)
    DynamicSolverAdvection.OutputTypeSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
    DynamicSolverAdvection.DynamicThetaSet(DYNAMIC_SOLVER_ADVECTION_THETA)
    # Get the dynamic linear solver
    DynamicSolverAdvection.DynamicLinearSolverGet(LinearSolverAdvection)

# Finish the creation of the problem solver
Problem.SolversCreateFinish()

#================================================================================================================================
#  Solver Equations
#================================================================================================================================

# Start the creation of the problem solver equations
NonlinearSolverCharacteristic = CMISS.Solver()
DynamicSolverNavierStokes     = CMISS.Solver()
DynamicSolverAdvection        = CMISS.Solver()
SolverEquationsCharacteristic = CMISS.SolverEquations()
SolverEquationsNavierStokes   = CMISS.SolverEquations()
SolverEquationsAdvection      = CMISS.SolverEquations()

Problem.SolverEquationsCreateStart()

#------------------

# CellML Solver
if (RCRBoundaries):
    CellMLSolver = CMISS.Solver()
    CellMLEquations = CMISS.CellMLEquations()
    Problem.CellMLEquationsCreateStart()
    Problem.SolverGet([Iterative1d0dControlLoopNumber,
                       Simple0DControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],
                      SolverDAEUserNumber,CellMLSolver)
    CellMLSolver.CellMLEquationsGet(CellMLEquations)
    # Add in the equations set
    CellMLEquations.CellMLAdd(CellML)    
    Problem.CellMLEquationsCreateFinish()

#------------------

# CHARACTERISTIC solver
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,
                       Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
NonlinearSolverCharacteristic.SolverEquationsGet(SolverEquationsCharacteristic)
SolverEquationsCharacteristic.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetCharacteristic = SolverEquationsCharacteristic.EquationsSetAdd(EquationsSetCharacteristic)

#------------------

#  NAVIER-STOKES solver
if (RCRBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,
                       Iterative1dControlLoopNumber,
                       CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.SolverEquationsGet(SolverEquationsNavierStokes)
SolverEquationsNavierStokes.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetNavierStokes = SolverEquationsNavierStokes.EquationsSetAdd(EquationsSetNavierStokes)

#------------------
if (coupledAdvection):
    # ADVECTION Solver
    Problem.SolverGet([SimpleAdvectionControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
                      SolverAdvectionUserNumber,DynamicSolverAdvection)
    DynamicSolverAdvection.SolverEquationsGet(SolverEquationsAdvection)
    SolverEquationsAdvection.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
    # Add in the equations set
    EquationsSetAdvection = SolverEquationsAdvection.EquationsSetAdd(EquationsSetAdvection)

# Finish the creation of the problem solver equations
Problem.SolverEquationsCreateFinish()
    
#================================================================================================================================
#  Boundary Conditions
#================================================================================================================================

# CHARACTERISTIC
BoundaryConditionsCharacteristic = CMISS.BoundaryConditions()
SolverEquationsCharacteristic.BoundaryConditionsCreateStart(BoundaryConditionsCharacteristic)

# Outlets (Area)
for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumber[terminalIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        if (nonReflecting):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedNonreflecting,A[nodeNumber,0])
        elif (RCRBoundaries):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedCellml,A[nodeNumber,0])
        else:
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                     versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber,0])


SolverEquationsCharacteristic.BoundaryConditionsCreateFinish()

#------------------

# NAVIER-STOKES
BoundaryConditionsNavierStokes = CMISS.BoundaryConditions()
SolverEquationsNavierStokes.BoundaryConditionsCreateStart(BoundaryConditionsNavierStokes)

# Inlet (Flow)
for inputIdx in range (1,numberOfInputNodes+1):
    nodeNumber = inputNodeNumber[inputIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                               versionIdx,derivIdx,nodeNumber,1,CMISS.BoundaryConditionsTypes.FixedFitted,Q[inputIdx][0])

for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumber[terminalIdx]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if nodeDomain == computationalNodeNumber:
        if (nonReflecting):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedNonreflecting,A[nodeNumber,0])
        elif (RCRBoundaries):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedCellml,A[nodeNumber,0])
        else:
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
                                                   versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber,0])

# Finish the creation of boundary conditions
SolverEquationsNavierStokes.BoundaryConditionsCreateFinish()

#------------------
if (coupledAdvection):
    # ADVECTION
    BoundaryConditionsAdvection = CMISS.BoundaryConditions()
    SolverEquationsAdvection.BoundaryConditionsCreateStart(BoundaryConditionsAdvection)
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeNumber = inputNodeNumber[inputIdx]
        nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberConc)
        if nodeDomain == computationalNodeNumber:
            BoundaryConditionsAdvection.SetNode(DependentFieldAdvection,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,0.0)
    SolverEquationsAdvection.BoundaryConditionsCreateFinish()
  
#================================================================================================================================
#  Element Length
#================================================================================================================================

if (checkTimestepStability):
    QMax = 430.0
    # Check the element length
    elementNumber = [0]*(totalNumberOfElements+1)
    elementLength = [0]*(totalNumberOfElements+1)
    for i in range(1,totalNumberOfElements+1):
        Node1 = elementNodes[i][0]
        Node2 = elementNodes[i][1]
        Node3 = elementNodes[i][2]
        Length1 = (((xValues[Node1][0]-xValues[Node2][0])**2)
                  +((yValues[Node1][0]-yValues[Node2][0])**2)
                  +((zValues[Node1][0]-zValues[Node2][0])**2))**0.5
        Length2 = (((xValues[Node2][0]-xValues[Node3][0])**2)
                  +((yValues[Node2][0]-yValues[Node3][0])**2)
                  +((zValues[Node2][0]-zValues[Node3][0])**2))**0.5
        elementNumber[i] = i
        elementLength[i] = Length1 + Length2
        elementLength[0] = elementLength[i]
        print "Element %1.0f" %elementNumber[i], 
        print "Length: %1.1f" %elementLength[i],
        print "Length1: %1.1f" %Length1,
        print "Length2: %1.1f" %Length2
    maxElementLength = max(elementLength)
    minElementLength = min(elementLength)
    print("Max Element Length: %1.3f" % maxElementLength)
    print("Min Element Length: %1.3f" % minElementLength)
               
    # Check the timestep
    for i in range(1,numberOfNodesSpace+1):
        beta = (3.0*math.sqrt(math.pi)*H[i,0]*E[i,0])/(4.0*A0[i,0])
        eig[i] = QMax/A0[i,0] + (A0[i,0]**0.25)*(math.sqrt(beta/(2.0*Rho)))
#        eig[i] = (430.0/(A0[i][0]))+(A0[i][0]**(0.25))*((2.0*(math.pi**(0.5))
#                   *E[i][0]*H[i][0]/(3.0*A0[i][0]*Rho))**(0.5))
        dt[i] = ((3.0**(0.5))/3.0)*minElementLength/eig[i]
        dt[0] = dt[i]
    minTimeStep = min(dt)
    print("Max allowable timestep:      %3.5f" % minTimeStep )
    
#================================================================================================================================
#  Run Solvers
#================================================================================================================================
# Solve the problem
print "Solving problem..."
start = time.time()

Problem.Solve()

end = time.time()
elapsed = end - start
print "Total Number of Elements = %d " %totalNumberOfElements
print "Calculation Time = %3.4f" %elapsed
print "Problem solved!"
print "#"

#================================================================================================================================
#  Data Analysis
#================================================================================================================================
        
if (analysisFlag):
    # Get the stiffness matrix using the dynamic type
    stiffnessMatrix = CMISS.DistributedMatrix()
    EquationsNavierStokes.DynamicMatrixGetByType(CMISS.EquationsSetDynamicMatrixTypes.STIFFNESS,stiffnessMatrix)
    stiffness = stiffnessMatrix.DataGet()
    #print('K Matrix:')
    #print(stiffness)

    # Get the damping matrix using the dynamic type
    dampingMatrix = CMISS.DistributedMatrix()
    EquationsNavierStokes.DynamicMatrixGetByType(CMISS.EquationsSetDynamicMatrixTypes.DAMPING,dampingMatrix)
    damping = dampingMatrix.DataGet()
    #print('C Matrix:')
    #print(damping)

    # Get the jacobian matrix using the dynamic type
    solverJacobian = CMISS.DistributedMatrix()
    SolverEquationsNavierStokes.JacobianMatrixGet(solverJacobian)
    Jacobian = solverJacobian.DataGet()
    #print("solverJacobian:")
    #print(Jacobian)

    dampingMatrix   = dampingMatrix.ToSciPy()
    stiffnessMatrix = stiffnessMatrix.ToSciPy()
    solverJacobian  = solverJacobian.ToSciPy()

    theta = DYNAMIC_SOLVER_NAVIER_STOKES_THETA[0]
    dt    = DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT
    dofNumber = solverJacobian.shape
    A_Matrix  = dampingMatrix + dt*theta*stiffnessMatrix
    Identity  = numpy.matrix(numpy.identity(dofNumber[0]))

    solverA_Matrix           = numpy.zeros(shape=(dofNumber[0],dofNumber[0]))
    solverJacobianMatrix     = numpy.zeros(shape=(dofNumber[0],dofNumber[0]))
    solverStiffnessMatrix    = numpy.zeros(shape=(dofNumber[0],dofNumber[0]))
    inv_solverJacobianMatrix = numpy.zeros(shape=(dofNumber[0],dofNumber[0]))

    for i in range(0,dofNumber[0]):
        for j in range(0,dofNumber[0]):
            solverJacobianMatrix[i][j]  = solverJacobian[i,j]

    for i in range(0,dofNumber[0]):
        for j in range(0,dofNumber[0]):
            solverStiffnessMatrix[i][j] = stiffnessMatrix[i+1][j+1]
            solverA_Matrix[i][j]        = A_Matrix[i+1][j+1]
            
    import numpy as np
    inv_solverJacobianMatrix = linalg.inv(solverJacobianMatrix)
    JacobianMatrix = (solverJacobianMatrix-solverA_Matrix)/(dt*theta)
    AmplificationMatrix = Identity-dt*np.dot(inv_solverJacobianMatrix,(JacobianMatrix+solverStiffnessMatrix))

    eigenvalues, eigenvectors = linalg.eig(AmplificationMatrix)
    maxEig = max(abs(e) for e in eigenvalues)
    print("Max Eigenvalue: %f" % maxEig)
    eigenvalues, eigenvectors = linalg.eig(AmplificationMatrix)
    minEig = min(abs(e) for e in eigenvalues)
    print("Min Eigenvalue: %f" % minEig)

    try:
        cond = maxEig / minEig
        print("Amplification condition number: %f" % cond)
    except ZeroDivisionError:
        # If condition number is infinty we effectively have a zero row
        print("Amplification condition number is infinte.")

#================================================================================================================================
#  Finish Program
#================================================================================================================================
