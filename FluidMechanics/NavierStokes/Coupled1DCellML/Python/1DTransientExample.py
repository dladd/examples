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
#> Contributor(s): David Ladd
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
#> OpenCMISS/examples/FluidMechanics/NavierStokes/Coupled1DCellML/Python/1DTransientExample.py
#>

#================================================================================================================================
#  Start Program
#================================================================================================================================

# Set program variables
EquationsSetFieldUserNumberStree          = 1336
EquationsSetFieldUserNumberCharacteristic = 1337
EquationsSetFieldUserNumberNavierStokes   = 1338
EquationsSetFieldUserNumberAdvection      = 1339

CoordinateSystemUserNumber = 1
BasisUserNumberSpace       = 2
BasisUserNumberTime        = 3
RegionUserNumber           = 4
RegionUserNumber2          = 5
MeshUserNumber             = 6
MeshUserNumber2            = 7
DecompositionUserNumber    = 8
DecompositionUserNumber2   = 9
GeometricFieldUserNumber   = 10
GeometricFieldUserNumber2  = 11
DependentFieldUserNumber   = 12
DependentFieldUserNumber2  = 13
DependentFieldUserNumber3  = 14
MaterialsFieldUserNumber   = 15
MaterialsFieldUserNumber2  = 16
IndependentFieldUserNumber = 17
EquationsSetUserNumberStree          = 18
EquationsSetUserNumberCharacteristic = 19
EquationsSetUserNumberNavierStokes   = 20
EquationsSetUserNumberAdvection      = 21
ProblemUserNumber                    = 22
CellMLUserNumber                     = 23
CellMLModelsFieldUserNumber          = 24
CellMLStateFieldUserNumber           = 25
CellMLIntermediateFieldUserNumber    = 26
CellMLParametersFieldUserNumber      = 27
MaterialsFieldUserNumberCellML       = 28
AnalyticFieldUserNumber              = 29
# Solver user numbers
SolverDAEUserNumber            = 1
SolverStreeUserNumber          = 1
SolverCharacteristicUserNumber = 2
SolverNavierStokesUserNumber   = 3
SolverAdvectionUserNumber      = 4
# Materials constants
MaterialsFieldUserNumberMu    = 1
MaterialsFieldUserNumberRho   = 2
MaterialsFieldUserNumberAlpha = 3
MaterialsFieldUserNumberPext  = 4
MaterialsFieldUserNumberLs    = 5
MaterialsFieldUserNumberTs    = 6
MaterialsFieldUserNumberMs    = 7
MaterialsFieldUserNumberG0    = 8
MaterialsFieldUserNumberD     = 1
# Materials variables
MaterialsFieldUserNumberA0    = 1
MaterialsFieldUserNumberE     = 2
MaterialsFieldUserNumberH     = 3

#================================================================================================================================
#  Initialise OpenCMISS
#================================================================================================================================

# Import the libraries (OpenCMISS,python,numpy,scipy)
import numpy,math,cmath,csv,time,sys,os,pdb
from scipy.fftpack import fft,ifft
from scipy.sparse  import linalg
from scipy.linalg  import inv,eig
from scipy.special import jn
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))
from opencmiss import CMISS

# Diagnostics
#CMISS.DiagnosticsSetOn(CMISS.DiagnosticTypes.ALL,[1,2,3,4,5],"Diagnostics",[""])
#CMISS.ErrorHandlingModeSet(CMISS.ErrorHandlingModes.TRAP_ERROR)
#CMISS.OutputSetOn("Testing")

# Get the computational nodes info
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber    = CMISS.ComputationalNodeNumberGet()

#================================================================================================================================
#  Problem Control Panel
#================================================================================================================================

numberOfDimensions     = 1  #(One-dimensional)
numberOfComponents     = 2  #(Flow & Area)
numberOfInputNodes     = 0
numberOfBifurcations   = 0
numberOfTrifurcations  = 0
numberOfTerminalNodes  = 0
inputNodeNumber        = []
bifurcationNodeNumber  = []
trifurcationNodeNumber = []
coupledNodeNumber      = []

# Set the user number
derivIdx   = 1
versionIdx = 1

# Set the flags
Heart               = False   # Set to use coupled 0D Windkessel models (from CellML) at model inlet boundaries
RCRBoundaries       = False   # Set to use coupled 0D Windkessel models (from CellML) at model outlet boundaries
nonReflecting       = False   # Set to use non-reflecting outlet boundaries
streeBoundaries     = False   # Set to use structured tree outlet boundaries
coupledAdvection    = False   # Set to solve a coupled advection problem
timestepStability   = False   # Set to do a basic check of the stability of the hyperbolic problem based on the timestep size
initialiseFromFile  = False   # Set to initialise values
ProgressDiagnostics = False   # Set to diagnostics

#================================================================================================================================
#  Mesh Reading
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> Reading geometry from files... << == "
    
# Read the node file
with open('Input/Node.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rownum = 0
    for row in reader:
        if (rownum == 0):
            # Read the header row
            header = row
        else:
            # Read the number of nodes
            if (rownum == 1):
                numberOfNodesSpace = int(row[5])
                totalNumberOfNodes = numberOfNodesSpace*3
                xValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)
                yValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)
                zValues = numpy.zeros((numberOfNodesSpace+1,4),dtype = numpy.float)
            # Initialise the coordinates
            xValues[rownum][0] = float(row[1])
            yValues[rownum][0] = float(row[2])
            zValues[rownum][0] = float(row[3])
            # Read the input nodes
            if (row[4] == 'input'):
                inputNodeNumber.append(rownum)
                numberOfInputNodes = numberOfInputNodes+1
            # Read the bifurcation nodes
            elif (row[4] == 'bifurcation'):
                numberOfBifurcations+=1
                bifurcationNodeNumber.append(rownum)
                xValues[rownum][1] = float(row[1])
                yValues[rownum][1] = float(row[2])
                zValues[rownum][1] = float(row[3])
                xValues[rownum][2] = float(row[1])
                yValues[rownum][2] = float(row[2])
                zValues[rownum][2] = float(row[3])
            # Read the trifurcation nodes
            elif (row[4] == 'trifurcation'):
                numberOfTrifurcations+=1
                trifurcationNodeNumber.append(rownum)
                xValues[rownum][1] = float(row[1])
                yValues[rownum][1] = float(row[2])
                zValues[rownum][1] = float(row[3])
                xValues[rownum][2] = float(row[1])
                yValues[rownum][2] = float(row[2])
                zValues[rownum][2] = float(row[3])
                xValues[rownum][3] = float(row[1])
                yValues[rownum][3] = float(row[2])
                zValues[rownum][3] = float(row[3])
            # Read the terminal nodes
            elif (row[4] == 'terminal'):
                coupledNodeNumber.append(rownum)
                numberOfTerminalNodes = numberOfTerminalNodes+1
        # Next line
        rownum+=1      

#------------------

# Read the element file
with open('Input/Element.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rownum = 0
    i = 0
    k = 0
    for row in reader:
        if (rownum == 0):
            # Read the header row
            header = row
        else:
            # Read the number of elements
            if (rownum == 1):
                totalNumberOfElements = int(row[11])
                elementNodes          = (totalNumberOfElements+1)*[3*[0]]
                bifurcationElements   = (numberOfBifurcations+1)*[3*[0]]
                trifurcationElements  = (numberOfTrifurcations+1)*[4*[0]]
            # Read the element nodes
            elementNodes[rownum] = [int(row[1]),int(row[2]),int(row[3])]
            # Read the bifurcation elements
            if (row[4]):
                i+=1
                bifurcationElements[i] = [int(row[4]),int(row[5]),int(row[6])]
            # Read the trifurcation elements
            elif (row[7]):
                k+=1
                trifurcationElements[k] = [int(row[7]),int(row[8]),int(row[9]),int(row[10])]
        # Next line
        rownum+=1
        
if (ProgressDiagnostics):
    print " Input at nodes: " + str(inputNodeNumber)
    print " Bifurcations at nodes: " + str(bifurcationNodeNumber)
    print " Trifurcations at nodes: " + str(trifurcationNodeNumber)
    print " Terminal at nodes: " + str(coupledNodeNumber)
    print " == >> Finished reading geometry... << == "

#================================================================================================================================
#  Initial Data & Default Values
#================================================================================================================================

# Set the material parameters
Rho  = 1050.0                                 # Rho         (kg/m3)
Mu   = 0.004                                  # Mu          (Pa.s)
D    = 100.0                                  # Diffusivity (m2/s)
G0   = 9.81                                   # Gravitational acceleration (m/s2)
Pext = 0.0                                    # External pressure (Pa)
Pv   = 2667.0                                 # Venous pressure = 20.0 mmHg (Pa)
dt   = [0]*(numberOfNodesSpace+1)             # TimeStep    (s)
eig  = [0]*(numberOfNodesSpace+1)             # Eigenvalues
A0   = numpy.zeros((numberOfNodesSpace+1,4))  # Area        (m2)
H    = numpy.zeros((numberOfNodesSpace+1,4))  # Thickness   (m)
E    = numpy.zeros((numberOfNodesSpace+1,4))  # Elasticity  (Pa)

# Material parameter scaling factors
Ls = 1000.0              # Length   (m -> mm)
Ts = 1000.0              # Time     (s -> ms)
Ms = 1000.0              # Mass     (kg -> g)

Alpha = 1.3              # Flow profile
Qs    = (Ls**3.0)/Ts     # Flow             (m3/s)  
As    = Ls**2.0          # Area             (m2)
Hs    = Ls               # vessel thickness (m)
Es    = Ms/(Ls*Ts**2.0)  # Elasticity Pa    (kg/(ms2) --> g/(mm.ms^2)
Rhos  = Ms/(Ls**3.0)     # Density          (kg/m3)
Mus   = Ms/(Ls*Ts)       # Viscosity        (kg/(ms))
Ps    = Ms/(Ls*Ts**2.0)  # Pressure         (kg/(ms2))
Ds    = (Ls**2.0)/Ts     # Diffusivity      (m2/s)
Zs    = Ps/Qs            # Impedance        (pa/(m3/s))
Gs    = Ls/(Ts**2.0)     # Acceleration    (m/s2)

# Read the MATERIAL file
with open('Input/Material.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rownum = 0
    for row in reader:
        if (rownum == 0):
            # Read the header row
            header = row
        else:
            A0[rownum][0] = float(row[1])
            E [rownum][0] = float(row[2])
            H [rownum][0] = float(row[3])
        # Next line
        rownum+=1
        
Rho = Rho*Rhos
Mu  = Mu*Mus
P   = Pext*Ps
A0  = A0*As
E   = E*Es
H   = H*Hs
D   = D*Ds
G0  = G0*Gs

Q  = numpy.zeros((numberOfNodesSpace+1,4))
A  = numpy.zeros((numberOfNodesSpace+1,4))
dQ = numpy.zeros((numberOfNodesSpace+1,4))
dA = numpy.zeros((numberOfNodesSpace+1,4))

# Set A0 for branch nodes
for bifIdx in range(1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx-1]
    for versionIdx in range(1,3):
        A0[nodeIdx][versionIdx] = A0[elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
        E [nodeIdx][versionIdx] = E [elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
        H [nodeIdx][versionIdx] = H [elementNodes[bifurcationElements[bifIdx][versionIdx]][1]][0]
for trifIdx in range(1,numberOfTrifurcations+1):
    nodeIdx = trifurcationNodeNumber[trifIdx-1]
    for versionIdx in range(1,4):
        A0[nodeIdx][versionIdx] = A0[elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]
        E [nodeIdx][versionIdx] = E [elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]
        H [nodeIdx][versionIdx] = H [elementNodes[trifurcationElements[trifIdx][versionIdx]][1]][0]

# Start with Q=0, A=A0 state
A = A0

# Or initialise from init file
if (initialiseFromFile):
    init = numpy.zeros([numberOfNodesSpace+1,4,4])
    init = numpy.load('./Input/init.npy')
    Q[1:numberOfNodesSpace+1,:] = init[:,0,:]
    A[1:numberOfNodesSpace+1,:] = init[:,1,:]
    dQ[1:numberOfNodesSpace+1,:] = init[:,2,:]
    dA[1:numberOfNodesSpace+1,:] = init[:,3,:]

# Set the output parameters
# (NONE/PROGRESS/TIMING/SOLVER/MATRIX)
DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE    = CMISS.SolverOutputTypes.NONE
NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE  = CMISS.SolverOutputTypes.NONE
NONLINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
LINEAR_SOLVER_CHARACTERISTIC_OUTPUT_TYPE    = CMISS.SolverOutputTypes.NONE
LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE     = CMISS.SolverOutputTypes.NONE
# (NONE/TIMING/SOLVER/MATRIX)
CMISS_SOLVER_OUTPUT_TYPE = CMISS.SolverOutputTypes.NONE
DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY = 10

# Set the time parameters
numberOfPeriods = 1.0
timePeriod      = 800
timeIncrement   = 0.1
startTime       = 0.0
stopTime  = numberOfPeriods*timePeriod
dynamicSolverNavierStokesTheta = [1.0]
dynamicSolverAdvectionTheta    = [0.5]

# Set the solver parameters
relativeToleranceNonlinearNavierStokes   = 1.0E-05  # default: 1.0E-05
absoluteToleranceNonlinearNavierStokes   = 1.0E-08  # default: 1.0E-10
solutionToleranceNonlinearNavierStokes   = 1.0E-05  # default: 1.0E-05
relativeToleranceLinearNavierStokes      = 1.0E-05  # default: 1.0E-05
absoluteToleranceLinearNavierStokes      = 1.0E-08  # default: 1.0E-10
relativeToleranceNonlinearCharacteristic = 1.0E-05  # default: 1.0E-05
absoluteToleranceNonlinearCharacteristic = 1.0E-08  # default: 1.0E-10
solutionToleranceNonlinearCharacteristic = 1.0E-05  # default: 1.0E-05
relativeToleranceLinearCharacteristic    = 1.0E-05  # default: 1.0E-05
absoluteToleranceLinearCharacteristic    = 1.0E-08  # default: 1.0E-10

DIVERGENCE_TOLERANCE = 1.0E+10  # default: 1.0E+05
MAXIMUM_ITERATIONS   = 100000   # default: 100000
RESTART_VALUE        = 3000     # default: 30

# N-S/C coupling tolerance
couplingTolerance1D = 1.0E+6
# 1D-0D coupling tolerance
couplingTolerance1D0D = 0.001

# Check the CellML flag
if (RCRBoundaries or Heart):
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
elif (streeBoundaries):
    if (coupledAdvection):
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.Coupled1D0DAdv_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        # Stree solver
        EquationsSetStreeSubtype = CMISS.EquationsSetSubtypes.Stree1D0DAdv
        # Advection solver
        EquationsSetAdvectionSubtype = CMISS.EquationsSetSubtypes.ADVECTION
        ProblemSubtype = CMISS.ProblemSubTypes.Stree1D0DAdv_NAVIER_STOKES
    else:
        # Navier-Stokes solver
        EquationsSetSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_NAVIER_STOKES
        # Characteristic solver
        EquationsSetCharacteristicSubtype = CMISS.EquationsSetSubtypes.Coupled1D0D_CHARACTERISTIC
        # Stree solver
        EquationsSetStreeSubtype = CMISS.EquationsSetSubtypes.Stree1D0D
        ProblemSubtype = CMISS.ProblemSubTypes.Stree1D0D_NAVIER_STOKES
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

# Start the creation of SPACE region
Region = CMISS.Region()
Region.CreateStart(RegionUserNumber,CMISS.WorldRegion)
Region.label = "ArterialSystem"
Region.coordinateSystem = CoordinateSystem
Region.CreateFinish()

if (streeBoundaries):
    # Start the creation of TIME region
    RegionStree = CMISS.Region()
    RegionStree.CreateStart(RegionUserNumber2,CMISS.WorldRegion)
    RegionStree.label = "StructuredTree"
    RegionStree.coordinateSystem = CoordinateSystem
    RegionStree.CreateFinish()

#================================================================================================================================
#  Bases
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> BASIS << == "

# Start the creation of SPACE bases
basisXiGaussSpace = 3
BasisSpace = CMISS.Basis()
BasisSpace.CreateStart(BasisUserNumberSpace)
BasisSpace.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
BasisSpace.numberOfXi = numberOfDimensions
BasisSpace.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]
BasisSpace.quadratureNumberOfGaussXi = [basisXiGaussSpace]
BasisSpace.CreateFinish()

if (streeBoundaries):
    # Start the creation of TIME bases
    basisXiGaussSpace = 3
    BasisTime = CMISS.Basis()
    BasisTime.CreateStart(BasisUserNumberTime)
    BasisTime.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
    BasisTime.numberOfXi = numberOfDimensions
    BasisTime.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]
    BasisTime.quadratureNumberOfGaussXi = [basisXiGaussSpace]
    BasisTime.CreateFinish()

#================================================================================================================================
#  Nodes
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> NODES << == "

# Start the creation of mesh nodes
Nodes = CMISS.Nodes()
Nodes.CreateStart(Region,totalNumberOfNodes)
Nodes.CreateFinish()

if (streeBoundaries):
    NodesStree = CMISS.Nodes()
    NodesStree.CreateStart(RegionStree,timePeriod+1)
    NodesStree.CreateFinish()

#================================================================================================================================
#  Mesh
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MESH << == "

# Start the creation of SPACE mesh
Mesh = CMISS.Mesh()
Mesh.CreateStart(MeshUserNumber,Region,numberOfDimensions)
Mesh.NumberOfElementsSet(totalNumberOfElements)
if (coupledAdvection):
    meshNumberOfComponents = 2
    Mesh.NumberOfComponentsSet(meshNumberOfComponents)
    # Specify the mesh components
    MeshElementsSpace = CMISS.MeshElements()
    MeshElementsConc  = CMISS.MeshElements()
    meshComponentNumberSpace = 1
    meshComponentNumberConc  = 2
else:
    meshNumberOfComponents = 1
    # Specify the mesh components
    Mesh.NumberOfComponentsSet(meshNumberOfComponents)
    # Specify the mesh components
    MeshElementsSpace = CMISS.MeshElements()
    meshComponentNumberSpace = 1

#------------------

# Specify the SPACE mesh component
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
if (coupledAdvection):
    MeshElementsConc.CreateStart(Mesh,meshComponentNumberConc,BasisSpace)
    for elemIdx in range(1,totalNumberOfElements+1):
        MeshElementsConc.NodesSet(elemIdx,elementNodes[elemIdx])
    MeshElementsConc.CreateFinish()

# Finish the creation of the mesh
Mesh.CreateFinish()

#------------------

if (streeBoundaries):
    # Start the creation of TIME mesh
    MeshTime = CMISS.Mesh()
    MeshTime.CreateStart(MeshUserNumber2,RegionStree,numberOfDimensions)
    MeshTime.NumberOfElementsSet(timePeriod)
    MeshTime.NumberOfComponentsSet(1)
    # Specify the mesh components
    MeshElementsTime = CMISS.MeshElements()
    meshComponentNumberTime = 1
    MeshElementsTime.CreateStart(MeshTime,meshComponentNumberTime,BasisTime)
    for elemIdx in range(1,timePeriod+1):
        MeshElementsTime.NodesSet(elemIdx,[elemIdx,elemIdx+1])
    MeshElementsTime.CreateFinish()                        
    MeshTime.CreateFinish()
    
#================================================================================================================================
#  Decomposition
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MESH DECOMPOSITION << == "

# Start the creation of SPACE mesh decomposition
Decomposition = CMISS.Decomposition()
Decomposition.CreateStart(DecompositionUserNumber,Mesh)
Decomposition.TypeSet(CMISS.DecompositionTypes.CALCULATED)
Decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
Decomposition.CreateFinish()

#------------------

if (streeBoundaries):
    # Start the creation of TIME mesh decomposition
    DecompositionTime = CMISS.Decomposition()
    DecompositionTime.CreateStart(DecompositionUserNumber2,MeshTime)
    DecompositionTime.TypeSet(CMISS.DecompositionTypes.CALCULATED)
    DecompositionTime.NumberOfDomainsSet(numberOfComputationalNodes)
    DecompositionTime.CreateFinish()

#================================================================================================================================
#  Geometric Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> GEOMETRIC FIELD << == "

# Start the creation of SPACE geometric field
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
     meshComponentNumberSpace)
GeometricField.CreateFinish()

# Set the geometric field values for version 1
versionIdx = 1
for nodeIdx in range(1,numberOfNodesSpace+1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
         versionIdx,derivIdx,nodeIdx,1,xValues[nodeIdx][0])
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
         versionIdx,derivIdx,nodeIdx,2,yValues[nodeIdx][0])
        GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
         versionIdx,derivIdx,nodeIdx,3,zValues[nodeIdx][0])
# Set the geometric field for bifurcation
for bifIdx in range (1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        for versionNumber in range(2,4):
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,1,xValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,2,yValues[nodeIdx][versionNumber-1])
            GeometricField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionNumber,derivIdx,nodeIdx,3,zValues[nodeIdx][versionNumber-1])
# Set the geometric field for trifurcation
for trifIdx in range (1,numberOfTrifurcations+1):
    nodeIdx = trifurcationNodeNumber[trifIdx-1]
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

#------------------

if (streeBoundaries):
    # Start the creation of TIME geometric field
    GeometricFieldTime = CMISS.Field()
    GeometricFieldTime.CreateStart(GeometricFieldUserNumber2,RegionStree)
    GeometricFieldTime.NumberOfVariablesSet(1)
    GeometricFieldTime.VariableLabelSet(CMISS.FieldVariableTypes.U,'Time')
    GeometricFieldTime.TypeSet = CMISS.FieldTypes.GEOMETRIC
    GeometricFieldTime.meshDecomposition = DecompositionTime
    GeometricFieldTime.ScalingTypeSet = CMISS.FieldScalingTypes.NONE
    # Set the mesh component to be used by the geometric field components
    for componentNumber in range(1,CoordinateSystem.dimension+1):
        GeometricFieldTime.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,
         componentNumber,meshComponentNumberTime)
    GeometricFieldTime.CreateFinish()
    
#================================================================================================================================
#  Equations Sets
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> EQUATIONS SET << == "

# Create the equations set for STREE
if (streeBoundaries):
    EquationsSetStree = CMISS.EquationsSet()
    EquationsSetFieldStree = CMISS.Field()
    # Set the equations set to be a dynamic linear problem
    EquationsSetStree.CreateStart(EquationsSetUserNumberStree,RegionStree,GeometricFieldTime,
        CMISS.EquationsSetClasses.FLUID_MECHANICS,CMISS.EquationsSetTypes.STREE_EQUATION,
         EquationsSetStreeSubtype,EquationsSetFieldUserNumberStree,EquationsSetFieldStree)
    EquationsSetStree.CreateFinish()

#------------------

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

#------------------

# Create the equations set for ADVECTION
if (coupledAdvection):
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

if (ProgressDiagnostics):
    print " == >> DEPENDENT FIELD << == "

# STREE
if (streeBoundaries):
    # Create the equations set dependent field variables
    DependentFieldStree = CMISS.Field()
    EquationsSetStree.DependentCreateStart(DependentFieldUserNumber3,DependentFieldStree)
    DependentFieldStree.VariableLabelSet(CMISS.FieldVariableTypes.U,'Stree 1st Variable')
    DependentFieldStree.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Stree 2nd Variable')
    EquationsSetStree.DependentCreateFinish()

#------------------

# CHARACTERISTIC
# Create the equations set dependent field variables
DependentFieldNavierStokes = CMISS.Field()
EquationsSetCharacteristic.DependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'General')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Derivatives')
DependentFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.V,'Characteristics')
if (RCRBoundaries or Heart):
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
if (RCRBoundaries or Heart):
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
    if (nodeDomain == computationalNodeNumber):
        if (nodeIdx in trifurcationNodeNumber):
            versions = [1,2,3,4]
        elif (nodeIdx in bifurcationNodeNumber):
            versions = [1,2,3]
        else:
            versions = [1]
        for versionIdx in versions:
            # U variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,Q[nodeIdx][versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,2,A[nodeIdx][versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
             versionIdx,derivIdx,nodeIdx,1,Q[nodeIdx][versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
             versionIdx,derivIdx,nodeIdx,2,A[nodeIdx][versionIdx-1])
            # delUdelN variables
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,dQ[nodeIdx][versionIdx-1])
            DependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,2,dA[nodeIdx][versionIdx-1])

# revert default version to 1
versionIdx = 1
             
# Finish the parameter update
DependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
DependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)   

#------------------

# ADVECTION
if (coupledAdvection):
    # Create the equations set dependent field variables
    DependentFieldAdvection = CMISS.Field()
    EquationsSetAdvection.DependentCreateStart(DependentFieldUserNumber2,DependentFieldAdvection)
    DependentFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.U,'Concentration')
    DependentFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.DELUDELN,'Deriv')
    # Set the mesh component to be used by the field components.
    DependentFieldAdvection.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumberConc)
    DependentFieldAdvection.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,meshComponentNumberConc)
    EquationsSetAdvection.DependentCreateFinish()

    # Initialise the dependent field variables
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeIdx = inputNodeNumber[inputIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberConc)
        if (nodeDomain == computationalNodeNumber):
            DependentFieldAdvection.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,0.0)

    # Finish the parameter update
    DependentFieldAdvection.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
    DependentFieldAdvection.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES) 

#================================================================================================================================
#  Materials Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> MATERIALS FIELD << == "

# STREE
if (streeBoundaries):
    # Create the equations set materials field variables 
    MaterialsFieldStree = CMISS.Field()
    EquationsSetStree.MaterialsCreateStart(MaterialsFieldUserNumber2,MaterialsFieldStree)
    MaterialsFieldStree.VariableLabelSet(CMISS.FieldVariableTypes.U,'Stree Impedance')
    MaterialsFieldStree.VariableLabelSet(CMISS.FieldVariableTypes.V,'Stree Flow')
    EquationsSetStree.MaterialsCreateFinish()

#------------------

# CHARACTERISTIC
# Create the equations set materials field variables 
MaterialsFieldNavierStokes = CMISS.Field()
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

# Set the materials field constants
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberMu,Mu)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberRho,Rho)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberAlpha,Alpha)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberPext,Pext)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberLs,Ls)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberTs,Ts)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberMs,Ms)
MaterialsFieldNavierStokes.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,
 CMISS.FieldParameterSetTypes.VALUES,MaterialsFieldUserNumberG0,G0)

# Initialise the materials field variables (A0,E,H)
bifIdx = 0
trifIdx = 0
for nodeIdx in range(1,numberOfNodesSpace+1,1):
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        if (nodeIdx in trifurcationNodeNumber):
            versions = [1,2,3,4]
        elif (nodeIdx in bifurcationNodeNumber):
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

# ADVECTION
if (coupledAdvection):
    # Create the equations set materials field variables 
    MaterialsFieldAdvection = CMISS.Field()
    EquationsSetAdvection.MaterialsCreateStart(MaterialsFieldUserNumber2,MaterialsFieldAdvection)
    MaterialsFieldAdvection.VariableLabelSet(CMISS.FieldVariableTypes.U,'Diffusivity')
    EquationsSetAdvection.MaterialsCreateFinish()
    # Set the materials field constant
    MaterialsFieldAdvection.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,  
     MaterialsFieldUserNumberD,D)

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
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,meshComponentNumberSpace)
IndependentFieldNavierStokes.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,meshComponentNumberSpace)
EquationsSetCharacteristic.IndependentCreateFinish()

#------------------

# NAVIER-STOKES
EquationsSetNavierStokes.IndependentCreateStart(IndependentFieldUserNumber,IndependentFieldNavierStokes)
EquationsSetNavierStokes.IndependentCreateFinish()

# Set the normal wave direction for bifurcation
for bifIdx in range (1,numberOfBifurcations+1):
    nodeIdx = bifurcationNodeNumber[bifIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
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
    nodeIdx = trifurcationNodeNumber[trifIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
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
if (RCRBoundaries or nonReflecting or streeBoundaries or Heart):
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if (nodeDomain == computationalNodeNumber):
            # Incoming
            IndependentFieldNavierStokes.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,1.0)

# Finish the parameter update
IndependentFieldNavierStokes.ParameterSetUpdateStart(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)
IndependentFieldNavierStokes.ParameterSetUpdateFinish(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES)

#------------------

# ADVECTION
if (coupledAdvection):
    # Create the equations set independent field variables  
    IndependentFieldAdvection = CMISS.Field()
    EquationsSetAdvection.IndependentCreateStart(DependentFieldUserNumber,DependentFieldNavierStokes)
    EquationsSetAdvection.IndependentCreateFinish()

#================================================================================================================================
# Analytic Field
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> ANALYTIC FIELD << == "

AnalyticFieldNavierStokes = CMISS.Field()
EquationsSetNavierStokes.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.FlowrateAorta,AnalyticFieldUserNumber,
 AnalyticFieldNavierStokes) # SplintFromFile,FlowrateAorta,FlowrateOlufsen
AnalyticFieldNavierStokes.VariableLabelSet(CMISS.FieldVariableTypes.U,'Input Flow')
EquationsSetNavierStokes.AnalyticCreateFinish()

#DOC-START cellml define field maps
#================================================================================================================================
#  RCR CellML Model Maps
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

    if (ProgressDiagnostics):
        print " == >> RCR CELLML MODEL << == "
        
    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,Region)
    # Number of CellML models
    CellMLModelIndex = [0]*(numberOfTerminalNodes+1)

    # Windkessel Model
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        print('reading model: ' + "./Input/CellMLModels/outlet"+str(terminalIdx)+"/ModelRCR.cellml")
        if (nodeDomain == computationalNodeNumber):
            CellMLModelIndex[terminalIdx] = CellML.ModelImport("./Input/CellMLModels/outlet"+str(terminalIdx)+"/ModelRCR.cellml")
            # known (to OpenCMISS) variables
            CellML.VariableSetAsKnown(CellMLModelIndex[terminalIdx],"Circuit/Qin")
            # to get from the CellML side 
            CellML.VariableSetAsWanted(CellMLModelIndex[terminalIdx],"Circuit/Pout")
    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        nodeIdx = coupledNodeNumber[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if (nodeDomain == computationalNodeNumber):
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
        nodeIdx = coupledNodeNumber[terminalIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if (nodeDomain == computationalNodeNumber):
            print("Terminal node: " + str(nodeIdx))
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
#  Heart CellML Model Maps
#================================================================================================================================

if (Heart):

    #----------------------------------------------------------------------------------------------------------------------------
    # Description
    #----------------------------------------------------------------------------------------------------------------------------
    # A CellML OD model is used to provide Heart model for the 1D model. A variable map copies the guess for the Pressure, P at 
    # the inlet node from the OpenCMISS Dependent Field to the CellML equation, which then returns flow, Q. The initial guess
    # value for P is taken from the previous time step or is 0 for t=0. In OpenCMISS this Q value is then imposed to the inlet.
    #----------------------------------------------------------------------------------------------------------------------------

    if (ProgressDiagnostics):
        print " == >> HEART CELLML MODEL << == "
    
    qCellMLComponent = 1
    pCellMLComponent = 2

    # Create the CellML environment
    CellML = CMISS.CellML()
    CellML.CreateStart(CellMLUserNumber,Region)
    # Number of CellML models
    CellMLModelIndex = [0]*(numberOfInputNodes+1)

    # Heart Model
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeIdx = inputNodeNumber[inputIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        print('reading model: ' + "./Input/CellMLModels/inlet/"+str(inputIdx)+"/Heart.cellml")
        if (nodeDomain == computationalNodeNumber):
            CellMLModelIndex[inputIdx] = CellML.ModelImport("./Input/CellMLModels/inlet/"+str(inputIdx)+"/Heart.cellml")
            # known (to OpenCMISS) variables
            CellML.VariableSetAsKnown(CellMLModelIndex[inputIdx],"Heart/P_art")
            # to get from the CellML side 
            CellML.VariableSetAsWanted(CellMLModelIndex[inputIdx],"Heart/Q_art")
    CellML.CreateFinish()

    # Start the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateStart()
    
    # ModelIndex
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeIdx = inputNodeNumber[inputIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if (nodeDomain == computationalNodeNumber):
            # Now we can set up the field variable component <--> CellML model variable mappings.
            # Map the OpenCMISS boundary flow rate values --> CellML
            # P is component 1 of the Dependent field U2 variable
            CellML.CreateFieldToCellMLMap(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U2,1,
             CMISS.FieldParameterSetTypes.VALUES,CellMLModelIndex[inputIdx],"Heart/P_art",CMISS.FieldParameterSetTypes.VALUES)
            # Map the returned pressure values from CellML --> CMISS
            # qCellML is component 1 of the Dependent field U1 variable
            CellML.CreateCellMLToFieldMap(CellMLModelIndex[inputIdx],"Heart/Q_art",CMISS.FieldParameterSetTypes.VALUES,
             DependentFieldNavierStokes,CMISS.FieldVariableTypes.U1,qCellMLComponent,CMISS.FieldParameterSetTypes.VALUES)

    # Finish the creation of CellML <--> OpenCMISS field maps
    CellML.FieldMapsCreateFinish()

    CellMLModelsField = CMISS.Field()
    CellML.ModelsFieldCreateStart(CellMLModelsFieldUserNumber,CellMLModelsField)
    CellML.ModelsFieldCreateFinish()
    
    # Set the models field at inlet boundary nodes
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeIdx = inputNodeNumber[inputIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeIdx,meshComponentNumberSpace)
        if (nodeDomain == computationalNodeNumber):
            print("Input node: " + str(nodeIdx))
            CellMLModelsField.ParameterSetUpdateNode(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
             versionIdx,derivIdx,nodeIdx,1,CellMLModelIndex[inputIdx])

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
    
#================================================================================================================================
#  Equations
#================================================================================================================================

if (ProgressDiagnostics):
    print " == >> EQUATIONS << == "

# 1th Equations Set - STREE
if (streeBoundaries):
    EquationsStree = CMISS.Equations()
    EquationsSetStree.EquationsCreateStart(EquationsStree)
    EquationsStree.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
    # (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
    EquationsStree.outputType = CMISS.EquationsOutputTypes.NONE
    EquationsSetStree.EquationsCreateFinish()

#------------------

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

#------------------

# 4th Equations Set - ADVECTION
if (coupledAdvection):
    EquationsAdvection = CMISS.Equations()
    EquationsSetAdvection.EquationsCreateStart(EquationsAdvection)
    EquationsAdvection.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
    # (NONE/TIMING/MATRIX/ELEMENT_MATRIX/NODAL_MATRIX)
    EquationsAdvection.outputType = CMISS.EquationsOutputTypes.NONE
    EquationsSetAdvection.EquationsCreateFinish()

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

1D-Stree
------


                                                      | 1) 0D Simple subloop   | 1) 0D/Structured tree Solver
                                                      |                              
    Time Loop, L0  | 1) 1D-0D Iterative Coupling, L1  | 2) 1D NS/C coupling:   | 1) Characteristic Nonlinear Solver
                   |    Convergence Loop (while loop) |    (while loop)        | 2) 1DNavierStokes Transient Solver
                   |
                   | 2) (optional) Simple subloop     | 1) Advection Linear Solver


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
if (RCRBoundaries or streeBoundaries or Heart):
   Iterative1d0dControlLoopNumber   = 1
   SimpleAdvectionControlLoopNumber = 2
   Simple0DControlLoopNumber        = 1
   Iterative1dControlLoopNumber     = 2
else:
   Iterative1dControlLoopNumber     = 1
   SimpleAdvectionControlLoopNumber = 2

# Start the creation of the problem control loop
TimeLoop = CMISS.ControlLoop()
Problem.ControlLoopCreateStart()
Problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],TimeLoop)
TimeLoop.LabelSet('Time Loop')
TimeLoop.TimesSet(startTime,stopTime,timeIncrement)
TimeLoop.TimeOutputSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY)

# Set tolerances for iterative convergence loops
if (RCRBoundaries or streeBoundaries or Heart):
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
if (streeBoundaries):
    LinearSolverStree             = CMISS.Solver()
if (coupledAdvection):
    DynamicSolverAdvection        = CMISS.Solver()
    LinearSolverAdvection         = CMISS.Solver()

Problem.SolversCreateStart()

#------------------

# 1st Solver, Simple 0D subloop - STREE
if (streeBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverStreeUserNumber,LinearSolverStree)
    # Set the nonlinear Jacobian type
    LinearSolverStree.OutputTypeSet(CMISS_SOLVER_OUTPUT_TYPE)

#------------------

# 1st Solver, Simple 0D subloop - CellML
if (RCRBoundaries or Heart):
    CellMLSolver = CMISS.Solver()
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverDAEUserNumber,CellMLSolver)
    CellMLSolver.OutputTypeSet(CMISS_SOLVER_OUTPUT_TYPE)

#------------------

# 1st Solver, Iterative 1D subloop - CHARACTERISTIC
if (RCRBoundaries or streeBoundaries or Heart):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
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
if (RCRBoundaries or streeBoundaries or Heart):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
     SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.OutputTypeSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
DynamicSolverNavierStokes.DynamicThetaSet(dynamicSolverNavierStokesTheta)
# Get the dynamic nonlinear solver
DynamicSolverNavierStokes.DynamicNonlinearSolverGet(NonlinearSolverNavierStokes)
# Set the nonlinear Jacobian type
NonlinearSolverNavierStokes.NewtonJacobianCalculationTypeSet(CMISS.JacobianCalculationTypes.EQUATIONS) #(.FD/EQUATIONS)
NonlinearSolverNavierStokes.OutputTypeSet(NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)

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

# 1st Solver, Simple advection subloop - ADVECTION
if (coupledAdvection):
    Problem.SolverGet([SimpleAdvectionControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
     SolverAdvectionUserNumber,DynamicSolverAdvection)
    DynamicSolverAdvection.OutputTypeSet(DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE)
    DynamicSolverAdvection.DynamicThetaSet(dynamicSolverAdvectionTheta)
    # Get the dynamic linear solver
    DynamicSolverAdvection.DynamicLinearSolverGet(LinearSolverAdvection)

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
if (streeBoundaries):
    LinearSolverStree             = CMISS.Solver()
    SolverEquationsStree          = CMISS.SolverEquations()
if (coupledAdvection):
    DynamicSolverAdvection        = CMISS.Solver()
    SolverEquationsAdvection      = CMISS.SolverEquations()

Problem.SolverEquationsCreateStart()

#------------------

# STREE Solver
if (streeBoundaries):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Simple0DControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverStreeUserNumber,LinearSolverStree)
    LinearSolverStree.SolverEquationsGet(SolverEquationsStree)
    SolverEquationsStree.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
    # Add in the equations set
    EquationsSetStree = SolverEquationsStree.EquationsSetAdd(EquationsSetStree)

#------------------

# CellML Solver
if (RCRBoundaries or Heart):
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
if (RCRBoundaries or streeBoundaries or Heart):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
     SolverCharacteristicUserNumber,NonlinearSolverCharacteristic)
NonlinearSolverCharacteristic.SolverEquationsGet(SolverEquationsCharacteristic)
SolverEquationsCharacteristic.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetCharacteristic = SolverEquationsCharacteristic.EquationsSetAdd(EquationsSetCharacteristic)

#------------------

#  NAVIER-STOKES solver
if (RCRBoundaries or streeBoundaries or Heart):
    Problem.SolverGet([Iterative1d0dControlLoopNumber,Iterative1dControlLoopNumber,
     CMISS.ControlLoopIdentifiers.NODE],SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
else:
    Problem.SolverGet([Iterative1dControlLoopNumber,CMISS.ControlLoopIdentifiers.NODE],
     SolverNavierStokesUserNumber,DynamicSolverNavierStokes)
DynamicSolverNavierStokes.SolverEquationsGet(SolverEquationsNavierStokes)
SolverEquationsNavierStokes.sparsityType = CMISS.SolverEquationsSparsityTypes.SPARSE
# Add in the equations set
EquationsSetNavierStokes = SolverEquationsNavierStokes.EquationsSetAdd(EquationsSetNavierStokes)

#------------------

# ADVECTION Solver
if (coupledAdvection):
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

if (ProgressDiagnostics):
    print " == >> BOUNDARY CONDITIONS << == "

if (streeBoundaries):
    # STREE
    BoundaryConditionsStree = CMISS.BoundaryConditions()
    SolverEquationsStree.BoundaryConditionsCreateStart(BoundaryConditionsStree)
    SolverEquationsStree.BoundaryConditionsCreateFinish()

#------------------

# CHARACTERISTIC
BoundaryConditionsCharacteristic = CMISS.BoundaryConditions()
SolverEquationsCharacteristic.BoundaryConditionsCreateStart(BoundaryConditionsCharacteristic)

# Area-outlet
for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumber[terminalIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedNonreflecting,A[nodeNumber][0])
        elif (RCRBoundaries):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedCellml,A[nodeNumber][0])
        elif (streeBoundaries):
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedStree,A[nodeNumber][0])
        else:
            BoundaryConditionsCharacteristic.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber][0])

SolverEquationsCharacteristic.BoundaryConditionsCreateFinish()

#------------------

# NAVIER-STOKES
BoundaryConditionsNavierStokes = CMISS.BoundaryConditions()
SolverEquationsNavierStokes.BoundaryConditionsCreateStart(BoundaryConditionsNavierStokes)

# Inlet (Flow)
for inputIdx in range (1,numberOfInputNodes+1):
    nodeNumber = inputNodeNumber[inputIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
         versionIdx,derivIdx,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED_INLET,Q[inputIdx][0])
# Area-outlet
for terminalIdx in range (1,numberOfTerminalNodes+1):
    nodeNumber = coupledNodeNumber[terminalIdx-1]
    nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberSpace)
    if (nodeDomain == computationalNodeNumber):
        if (nonReflecting):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedNonreflecting,A[nodeNumber][0])
        elif (RCRBoundaries):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedCellml,A[nodeNumber][0])
        elif (streeBoundaries):
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FixedStree,A[nodeNumber][0])
        else:
            BoundaryConditionsNavierStokes.SetNode(DependentFieldNavierStokes,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,A[nodeNumber][0])

# Finish the creation of boundary conditions
SolverEquationsNavierStokes.BoundaryConditionsCreateFinish()

#------------------

# ADVECTION
if (coupledAdvection):
    BoundaryConditionsAdvection = CMISS.BoundaryConditions()
    SolverEquationsAdvection.BoundaryConditionsCreateStart(BoundaryConditionsAdvection)
    for inputIdx in range (1,numberOfInputNodes+1):
        nodeNumber = inputNodeNumber[inputIdx-1]
        nodeDomain = Decomposition.NodeDomainGet(nodeNumber,meshComponentNumberConc)
        if (nodeDomain == computationalNodeNumber):
            BoundaryConditionsAdvection.SetNode(DependentFieldAdvection,CMISS.FieldVariableTypes.U,
             versionIdx,derivIdx,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,1.0)
    SolverEquationsAdvection.BoundaryConditionsCreateFinish()
  
#================================================================================================================================
#  Element Length
#================================================================================================================================

if (timestepStability):
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
        beta   = (3.0*math.sqrt(math.pi)*H[i,0]*E[i,0])/(4.0*A0[i,0])
        eig[i] = QMax/A0[i,0] + (A0[i,0]**0.25)*(math.sqrt(beta/(2.0*Rho)))
        dt[i]  = ((3.0**(0.5))/3.0)*minElementLength/eig[i]
        dt[0]  = dt[i]
    minTimeStep = min(dt)
    print("Max allowable timestep:      %3.5f" % minTimeStep )

#================================================================================================================================
#  Transmission Line Theory
#================================================================================================================================

'''
terminal 1 ----- brain ----------- right vertebral artery
terminal 2 ----- right hand ------ right radial artery
terminal 3 ----- right hand ------ right ulnar artery
terminal 4 ----- brain ----------- right external carotid artery
terminal 5 ----- brain ----------- right internal carotid artery
terminal 6 ----- brain ----------- left  external carotid artery
terminal 7 ----- brain ----------- left  internal carotid artery
terminal 8 ----- brain ----------- left  vertebral artery
terminal 9 ----- left hand ------- left  ulnar artery
terminal 10 ---- left hand ------- left  radial artery
terminal 11 ---- spleen ---------- splenic artery
terminal 12 ---- liver ----------- hepatic artery
terminal 13 ---- stomach --------- gastric artery
terminal 14 ---- intestines ------ superior mesenteric artery
terminal 15 ---- left kidney ----- left  renal artery
terminal 16 ---- right kidney ---- right renal artery
terminal 17 ---- right leg ------- right internal iliac artery
terminal 18 ---- right leg ------- right deep femoral artery
terminal 19 ---- right foot ------ right anterior femoral artery
terminal 20 ---- right foot ------ right posterior tibial artery
terminal 21 ---- right leg ------- right fibular artery
terminal 22 ---- left leg -------- left  internal iliac artery
terminal 23 ---- left leg -------- left  deep femoral artery
terminal 24 ---- left foot ------- left  anterior tibial artery
terminal 25 ---- left foot ------- left  posterior tibial artery
terminal 26 ---- left leg -------- left  fibular artery
terminal 27 ---- intestines ------ inferior mesenteric artery
'''

if (streeBoundaries):
    if (ProgressDiagnostics):
        print " == >> STREE << == "

    numberOfTerminalNodes = 27
    # Loop through the terminal nodes
    for terminalIdx in range (1,numberOfTerminalNodes+1):
        # Read the organ node file
        with open('Input/stree/'+str(terminalIdx)+'.csv','rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rownum = 0
            for row in reader:
                if (rownum == 0):
                    # Read the header row
                    header = row
                else:
                    # Read number of nodes
                    if (rownum == 1):
                        numberOfSegments = int(row[6])
                        stree = numpy.zeros((numberOfSegments+1,7,timePeriod+1),dtype = numpy.float)
                    stree[rownum][0] = float(row[0])/Ls  # Length of segment
                    stree[rownum][1] = float(row[1])     # Radius of segment
                    stree[rownum][2] = float(row[2])     # Terminal segment
                    stree[rownum][3] = float(row[3])     # Number of parent segment
                    if (row[4]):
                        stree[rownum][4] = float(row[4]) # Number of daughter segments
                        stree[rownum][5] = float(row[5])
                # Next line
                rownum+=1

        # Loop through the segments to calculate each segment impedance
        for idx in range(1,numberOfSegments+1):
            n = numberOfSegments+1-idx                    # Start from last segment
            L = stree[n][0][0]                            # Length of segment
            r = stree[n][1][0]                            # Radius of segment
            term = stree[n][2][0]                         # Terminal segment
            if (term == 0):
                # Calculate daughter segment impedance
                zin1=stree[stree[n][4][0]][6][0]
                zin2=stree[stree[n][5][0]][6][0]
    
            Ng  = 8                                       # Number of generations
            h   = 0.35*r                                  # Thickness
            E   = 0.4E+6                                  # Elasticity
            A0  = math.pi*(r**2.0)                        # Area at rest
            Qin = 6.5E-6                                  # Input flow
            T   = timePeriod/Ts                           # Time period
            zin = [0]*(timePeriod+1)                      # Impedance
            Cp  = (3.0*A0*(A0/math.pi)**0.5)/(2.0*E*h)    # Vessel wall compliance

            # Non-zero frequency condition
            for k in range(0,timePeriod+1):
                if (k == 0):
                    # Zero frequency condition
                    # Terminal load
                    if (term == 0):
                        zL = zin1*zin2/(zin1+zin2)
                    else:
                        zL = (Pv/Qin)*(2.0**Ng)
                    # Transfer function
                    zin[k] = 8.0*(Mu/Mus)*L/((A0**2.0)/math.pi)+zL
                else:
                    # Frequency
                    freq = 2.0*math.pi*k/T                         
                    # Womersley number
                    w = (A0*freq*(Rho/Rhos)/((Mu/Mus)*math.pi))**0.5    
                    w0 = ((1j)**1.5)*w
                    # Bessel function zeroth-order
                    J0 = jn(0,w0)
                    # Bessel function first-order
                    J1 = jn(1,w0)
                    # Bessel function
                    Fj = (2.0*J1)/(w0*J0)
                    # Wave propagation velocity
                    c = cmath.sqrt(A0*(1.0-Fj)/((Rho/Rhos)*Cp))
                    g = c*Cp
                    # Terminal load
                    if (term == 1):
                        zL = 1.0/(c*Cp)
                    else:
                        zL = zin1*zin2/(zin1+zin2)
                    # Transfer function
                    zin[k] = ((1j)*cmath.sin(freq*L/c)/g+zL*cmath.cos(freq*L/c))/(cmath.cos(freq*L/c)+(1j)*g*zL*cmath.sin(freq*L/c))
                #Saving the line's characteristics
                stree[n][6][k] = zin[k]
        # Invrese fourier transform
        zt = ifft(stree[1][6])*Zs
        # Set the impedance
        for k in range(0,timePeriod+1):
            MaterialsFieldStree.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,
             CMISS.FieldParameterSetTypes.VALUES,1,1,k+1,terminalIdx,zt[k].real)
         
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
#  Finish Program
#================================================================================================================================