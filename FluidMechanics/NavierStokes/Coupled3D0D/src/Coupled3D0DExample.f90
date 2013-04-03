!> \file
!> \author David Ladd
!> \brief This is an example program to solve Navier-Stokes over a Carotid bifurcation mesh with CellML
!>  coupled boundary conditions 
!>
!> \section LICENSE
!>
!> Version: MPL 1.1/GPL 2.0/LGPL 2.1
!>
!> The contents of this file are subject to the Mozilla Public License
!> Version 1.1 (the "License"); you may not use this file except in
!> compliance with the License. You may obtain a copy of the License at
!> http://www.mozilla.org/MPL/
!>
!> Software distributed under the License is distributed on an "AS IS"
!> basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
!> License for the specific language governing rights and limitations
!> under the License.
!>
!> The Original Code is OpenCMISS
!>
!> The Initial Developer of the Original Code is University of Auckland,
!> Auckland, New Zealand and University of Oxford, Oxford, United
!> Kingdom. Portions created by the University of Auckland and University
!> of Oxford are Copyright (C) 2007 by the University of Auckland and
!> the University of Oxford. All Rights Reserved.
!>
!> Contributor(s):
!>
!> Alternatively, the contents of this file may be used under the terms of
!> either the GNU General Public License Version 2 or later (the "GPL"), or
!> the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
!> in which case the provisions of the GPL or the LGPL are applicable instead
!> of those above. If you wish to allow use of your version of this file only
!> under the terms of either the GPL or the LGPL, and not to allow others to
!> use your version of this file under the terms of the MPL, indicate your
!> decision by deleting the provisions above and replace them with the notice
!> and other provisions required by the GPL or the LGPL. If you do not delete
!> the provisions above, a recipient may use your version of this file under
!> the terms of any one of the MPL, the GPL or the LGPL.
!>
!> \example FluidMechanics/NavierStokes/CarotidCellML/src/Coupled3D0DExample.f90
!!
!<

!> Main program
PROGRAM Coupled3D0D

  USE OPENCMISS
  USE MPI
  USE FIELDML_API

#ifdef WIN32
  USE IFQWIN
#endif

  IMPLICIT NONE

  !================================================================================================================================
  !Test program parameters
  !================================================================================================================================



  INTEGER(CMISSIntg), PARAMETER :: EquationsSetFieldUserNumber=1337
  TYPE(CMISSFieldType) :: EquationsSetField

  INTEGER(CMISSIntg), PARAMETER :: CoordinateSystemUserNumber=1
  INTEGER(CMISSIntg), PARAMETER :: RegionUserNumber=2
  INTEGER(CMISSIntg), PARAMETER :: MeshUserNumber=3
  INTEGER(CMISSIntg), PARAMETER :: DecompositionUserNumber=4
  INTEGER(CMISSIntg), PARAMETER :: GeometricFieldUserNumber=5
  INTEGER(CMISSIntg), PARAMETER :: DependentFieldUserNumberNavierStokes=6
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberNavierStokes=7
  INTEGER(CMISSIntg), PARAMETER :: EquationsSetUserNumberNavierStokes=8
  INTEGER(CMISSIntg), PARAMETER :: ProblemUserNumber=9
  INTEGER(CMISSIntg), PARAMETER :: BasisUserNumber=10
  INTEGER(CMISSIntg), PARAMETER :: GeneratedMeshUserNumber=11

  INTEGER(CMISSIntg), PARAMETER :: CellMLUserNumber=12
  INTEGER(CMISSIntg), PARAMETER :: CellMLModelsFieldUserNumber=13
  INTEGER(CMISSIntg), PARAMETER :: CellMLStateFieldUserNumber=14
  INTEGER(CMISSIntg), PARAMETER :: CellMLIntermediateFieldUserNumber=15
  INTEGER(CMISSIntg), PARAMETER :: CellMLParametersFieldUserNumber=16
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberCellML=17
  INTEGER(CMISSIntg), PARAMETER :: AnalyticFieldUserNumber=18

  INTEGER(CMISSIntg), PARAMETER :: DomainUserNumber=1
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberNavierStokesMu=1
  INTEGER(CMISSIntg), PARAMETER :: MaterialsFieldUserNumberNavierStokesRho=2

  INTEGER(CMISSIntg), PARAMETER :: basisNumberTriquadratic=1
  INTEGER(CMISSIntg), PARAMETER :: basisNumberTrilinear=2
  INTEGER(CMISSIntg), PARAMETER :: gaussQuadrature(3) = [3,3,3]

  INTEGER(CMISSIntg), PARAMETER :: basisNumberSpace=1
  INTEGER(CMISSIntg), PARAMETER :: basisNumberVelocity=2
  INTEGER(CMISSIntg), PARAMETER :: basisNumberPressure=3

  !Program types
  
  !Program variables

  CHARACTER(KIND=C_CHAR,LEN=*), PARAMETER :: inputFilename = "input/hexCylinder1/hexCylinder1.xml"

  INTEGER(CMISSIntg) :: numberOfDimensions
  INTEGER(CMISSIntg) :: quadratureOrder = 3
  INTEGER(CMISSIntg) :: basisType
  INTEGER(CMISSIntg) :: numberOfDataPoints 

  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_SPACE
  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_VELOCITY
  INTEGER(CMISSIntg) :: BASIS_XI_GAUSS_PRESSURE
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_SPACE
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_VELOCITY
  INTEGER(CMISSIntg) :: BASIS_XI_INTERPOLATION_PRESSURE
  INTEGER(CMISSIntg) :: MESH_NUMBER_OF_COMPONENTS
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_SPACE
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_VELOCITY
  INTEGER(CMISSIntg) :: MESH_COMPONENT_NUMBER_PRESSURE

  INTEGER(CMISSIntg) :: NUMBER_GLOBAL_X_ELEMENTS,NUMBER_GLOBAL_Y_ELEMENTS,NUMBER_GLOBAL_Z_ELEMENTS
  INTEGER(CMISSIntg) :: MPI_IERROR

  INTEGER(CMISSIntg) :: MAXIMUM_ITERATIONS
  INTEGER(CMISSIntg) :: RESTART_VALUE

  INTEGER(CMISSIntg) :: numberOfFixedWallNodes
  INTEGER(CMISSIntg) :: numberOfBoundaryNodesInlet
  INTEGER(CMISSIntg) :: numberOfLinearBoundaryNodesOutlet
  INTEGER(CMISSIntg) :: numberOfQuadraticBoundaryNodesOutlet
  INTEGER(CMISSIntg) :: numberOfBoundaryElementsOutlet
  INTEGER(CMISSIntg) :: numberOfBoundaryElementsInlet

  INTEGER(CMISSIntg) :: EQUATIONS_NAVIER_STOKES_OUTPUT
  INTEGER(CMISSIntg) :: COMPONENT_NUMBER
  INTEGER(CMISSIntg) :: nodeNumber,elementNumber
  INTEGER(CMISSIntg) :: nodeCounter,elementCounter,nodeIdx,dataPointIdx
  INTEGER(CMISSIntg) :: CONDITION
  INTEGER(CMISSIntg) :: argCount
  INTEGER(CMISSIntg) :: numberOfNodesPerRow, rowNodeCount
  INTEGER(CMISSIntg) :: resistanceComponent

  INTEGER(CMISSIntg) :: LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE
  INTEGER(CMISSIntg) :: NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE
  INTEGER(CMISSIntg) :: DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY
  INTEGER(CMISSIntg) :: DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE

  INTEGER(CMISSIntg) :: EquationsSetSubtype
  INTEGER(CMISSIntg) :: ProblemSubtype

  INTEGER(CMISSIntg) :: SolverDAEUserNumber,SolverNavierStokesUserNumber

  INTEGER, ALLOCATABLE, DIMENSION(:):: fixedWallNodes
  INTEGER, ALLOCATABLE, DIMENSION(:):: boundaryNodesInlet
  INTEGER, ALLOCATABLE, DIMENSION(:):: boundaryElementsInlet
  INTEGER, ALLOCATABLE, DIMENSION(:):: quadraticBoundaryNodesOutlet
  INTEGER, ALLOCATABLE, DIMENSION(:):: linearBoundaryNodesOutlet
  INTEGER, ALLOCATABLE, DIMENSION(:):: boundaryElementsOutlet

  REAL(CMISSDP) :: HEIGHT
  REAL(CMISSDP) :: WIDTH
  REAL(CMISSDP) :: LENGTH

  REAL(CMISSDP), DIMENSION(5,3) :: dataPointValues !(number_of_data_points,dimension)

  REAL(CMISSDP) :: INITIAL_FIELD_NAVIER_STOKES(3)
  REAL(CMISSDP) :: BOUNDARY_CONDITIONS_NAVIER_STOKES(3)
  REAL(CMISSDP) :: OutletPressureValue
  REAL(CMISSDP) :: normalValueInlet(3)
  REAL(CMISSDP) :: normalValueOutlet(3)
  REAL(CMISSDP) :: DIVERGENCE_TOLERANCE
  REAL(CMISSDP) :: RELATIVE_TOLERANCE
  REAL(CMISSDP) :: SOLUTION_TOLERANCE
  REAL(CMISSDP) :: ABSOLUTE_TOLERANCE
  REAL(CMISSDP) :: LINESEARCH_ALPHA
  REAL(CMISSDP) :: VALUE
  REAL(CMISSDP) :: MU_PARAM_NAVIER_STOKES
  REAL(CMISSDP) :: RHO_PARAM_NAVIER_STOKES
  REAL(CMISSDP) :: resistanceProximal
  REAL(CMISSDP) :: boundaryID

  REAL(CMISSDP) :: DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME
  REAL(CMISSDP) :: DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME
  REAL(CMISSDP) :: DYNAMIC_SOLVER_NAVIER_STOKES_THETA
  REAL(CMISSDP) :: DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT

  LOGICAL :: LINEAR_SOLVER_NAVIER_STOKES_DIRECTFlag
  LOGICAL :: fixedWallNodesFlag
  LOGICAL :: boundaryNodesInletFlag
  LOGICAL :: quadraticBoundaryNodesOutletFlag
  LOGICAL :: linearBoundaryNodesOutletFlag
  LOGICAL :: SUPGFlag
  LOGICAL :: fixedMeshFlag
  LOGICAL :: ALE_SOLVER_NAVIER_STOKESFlag
  LOGICAL :: DYNAMIC_SOLVER_NAVIER_STOKESFlag
  LOGICAL :: DYNAMIC_SOLVER_NAVIER_STOKES_RESUME_SOLVEFlag
  LOGICAL :: calculateElementFacesFlag
  LOGICAL :: LagrangeBasisFlag
  LOGICAL :: StokesFlag
  LOGICAL :: boundaryElementsOutletFlag
  LOGICAL :: boundaryElementsInletFlag
  LOGICAL :: setOutletPressureOutletFlag
  LOGICAL :: equationsOutputFlag

  LOGICAL :: cellmlFlag
  LOGICAL :: windkesselFlag
  LOGICAL :: initialiseFieldPCV,fixedPressureBoundaryCondition

  CHARACTER *15 BUFFER
  CHARACTER *15 ARG

  !CMISS variables

  !Regions
  TYPE(CMISSRegionType) :: Region
  TYPE(CMISSRegionType) :: WorldRegion
  !Coordinate systems
  TYPE(CMISSCoordinateSystemType) :: CoordinateSystem
  TYPE(CMISSCoordinateSystemType) :: WorldCoordinateSystem
  !Nodes
  TYPE(CMISSNodesType) :: Nodes
  !Meshes
  TYPE(CMISSMeshType) :: Mesh
  TYPE(CMISSGeneratedMeshType) :: GeneratedMesh  
  !Decompositions
  TYPE(CMISSDecompositionType) :: Decomposition
  !Fields
  TYPE(CMISSFieldsType) :: Fields
  !Field types
  TYPE(CMISSFieldType) :: GeometricField
  TYPE(CMISSFieldType) :: DependentFieldNavierStokes
  TYPE(CMISSFieldType) :: MaterialsFieldNavierStokes
  TYPE(CMISSFieldType) :: AnalyticFieldNavierStokes
  ! CellML types
  TYPE(CMISSCellMLType) :: CellML
  TYPE(CMISSFieldType) :: MaterialsFieldCellML
  TYPE(CMISSCellMLEquationsType) :: CellMLEquations
  TYPE(CMISSFieldType) :: CellMLModelsField
  TYPE(CMISSFieldType) :: CellMLStateField
  TYPE(CMISSFieldType) :: CellMLIntermediateField
  TYPE(CMISSFieldType) :: CellMLParametersField
  TYPE(CMISSSolverType) :: CellMLSolver
  !Basis type
  TYPE(CMISSBasisType) :: basisInterpolations(2)
  TYPE(CMISSBasisType) :: basisSpace
  TYPE(CMISSBasisType) :: basisVelocity
  TYPE(CMISSBasisType) :: basisPressure

  !Boundary conditions
  TYPE(CMISSBoundaryConditionsType) :: BoundaryConditionsNavierStokes
  !Equations sets
  TYPE(CMISSEquationsSetType) :: EquationsSetNavierStokes
  !Equations
  TYPE(CMISSEquationsType) :: EquationsNavierStokes
  !Problems
  TYPE(CMISSProblemType) :: Problem
  !Control loops
  TYPE(CMISSControlLoopType) :: ControlLoop
  !Solvers
  TYPE(CMISSSolverType) :: DynamicSolverNavierStokes
  TYPE(CMISSSolverType) :: NonlinearSolverNavierStokes
  TYPE(CMISSSolverType) :: LinearSolverNavierStokes
  !Solver equations
  TYPE(CMISSSolverEquationsType) :: SolverEquationsNavierStokes

  !FieldML parsing variables
  TYPE(CMISSFieldMLIOType) :: fieldmlInfo

  INTEGER(CMISSIntg) :: meshComponentCount
  INTEGER(CMISSIntg) :: coordinateCount

#ifdef WIN32
  !Quickwin type
  LOGICAL :: QUICKWIN_STATUS=.FALSE.
  TYPE(WINDOWCONFIG) :: QUICKWIN_WINDOW_CONFIG
#endif
  
  !Generic CMISS variables

  INTEGER(CMISSIntg) :: NumberOfComputationalNodes,ComputationalNodeNumber,BoundaryNodeDomain,BoundaryElementDomain
  INTEGER(CMISSIntg) :: CellMLIndex
  INTEGER(CMISSIntg) :: CellMLModelIndex
  INTEGER(CMISSIntg) :: EquationsSetIndex
  INTEGER(CMISSIntg) :: Err
  
#ifdef WIN32
  !Initialise QuickWin
  QUICKWIN_WINDOW_CONFIG%TITLE="General Output" !Window title
  QUICKWIN_WINDOW_CONFIG%NUMTEXTROWS=-1 !Max possible number of rows
  QUICKWIN_WINDOW_CONFIG%MODE=QWIN$SCROLLDOWN
  !Set the window parameters
  QUICKWIN_STATUS=SETWINDOWCONFIG(QUICKWIN_WINDOW_CONFIG)
  !If attempt fails set with system estimated values
  IF(.NOT.QUICKWIN_STATUS) QUICKWIN_STATUS=SETWINDOWCONFIG(QUICKWIN_WINDOW_CONFIG)
#endif


  !================================================================================================================================
  !PROBLEM CONTROL PANEL
  !================================================================================================================================

  !Set some defaults
  numberOfDimensions=3
  fixedMeshFlag = .TRUE.
  LagrangeBasisFlag = .TRUE.
  SUPGFlag=.TRUE.
  calculateElementFacesFlag=.TRUE.
  setOutletPressureOutletFlag = .TRUE.
  StokesFlag=.FALSE.
  cellmlFlag = .TRUE.
  windkesselFlag = .TRUE.
  equationsOutputFlag = .FALSE.

  !Default mesh resolution
  NUMBER_GLOBAL_X_ELEMENTS=2
  NUMBER_GLOBAL_Y_ELEMENTS=2
  IF(numberOfDimensions==3) THEN
    NUMBER_GLOBAL_Z_ELEMENTS=6
  ENDIF

  !Set time parameters
  DYNAMIC_SOLVER_NAVIER_STOKESFlag=.TRUE.
  DYNAMIC_SOLVER_NAVIER_STOKES_RESUME_SOLVEFlag=.FALSE.
  DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME=0.0_CMISSDP
  DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME=1.001_CMISSDP 
  DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT=0.01_CMISSDP
  DYNAMIC_SOLVER_NAVIER_STOKES_THETA=1.0_CMISSDP
  !Set result output parameter
  DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY=1
  !Set solver parameters
  LINEAR_SOLVER_NAVIER_STOKES_DIRECTFlag=.TRUE. ! set false to use full matrices
  RELATIVE_TOLERANCE=1.0E-5_CMISSDP !default: 1.0E-05_CMISSDP
  SOLUTION_TOLERANCE=1.0E-5_CMISSDP !default: 1.0E-05_CMISSDP
  ABSOLUTE_TOLERANCE=1.0E-10_CMISSDP !default: 1.0E-10_CMISSDP
  DIVERGENCE_TOLERANCE=1.0E5 !default: 1.0E5
  MAXIMUM_ITERATIONS=100000 !default: 100000
  RESTART_VALUE=30 !default: 30
  LINESEARCH_ALPHA=1.0

  !Set initial values for the velocity field
  INITIAL_FIELD_NAVIER_STOKES(1)=0.0_CMISSDP
  INITIAL_FIELD_NAVIER_STOKES(2)=0.0_CMISSDP
  INITIAL_FIELD_NAVIER_STOKES(3)=0.0_CMISSDP
  !Set normals for Inlet boundary
  normalValueInlet(1)=0.0_CMISSDP
  normalValueInlet(2)=0.0_CMISSDP
  normalValueInlet(3)=-1.0_CMISSDP
  !Set normals for Outlet boundary
  normalValueOutlet(1)=0.0_CMISSDP
  normalValueOutlet(2)=0.0_CMISSDP
  normalValueOutlet(3)=1.0_CMISSDP
  !Set default boundary conditions
  IF(numberOfDimensions==2) THEN
    BOUNDARY_CONDITIONS_NAVIER_STOKES(1)=0.0_CMISSDP
    BOUNDARY_CONDITIONS_NAVIER_STOKES(2)=1.0_CMISSDP
    BOUNDARY_CONDITIONS_NAVIER_STOKES(3)=0.0_CMISSDP
    OutletPressureValue=0.0_CMISSDP
  ELSE
    BOUNDARY_CONDITIONS_NAVIER_STOKES(1)=0.0_CMISSDP
    BOUNDARY_CONDITIONS_NAVIER_STOKES(2)=0.0_CMISSDP
    BOUNDARY_CONDITIONS_NAVIER_STOKES(3)=1.0_CMISSDP
    OutletPressureValue=100.0_CMISSDP
    resistanceProximal=100.0_CMISSDP
  ENDIF
  fixedWallNodesFlag=.TRUE.
  boundaryNodesInletFlag=.TRUE.

  !Set material parameters
  MU_PARAM_NAVIER_STOKES=1.0_CMISSDP
  RHO_PARAM_NAVIER_STOKES=1.0_CMISSDP
  !Set interpolation parameters
  IF(numberOfDimensions==2) THEN
    BASIS_XI_GAUSS_SPACE=2
    BASIS_XI_GAUSS_VELOCITY=2
    BASIS_XI_GAUSS_PRESSURE=2
  ELSE
    BASIS_XI_GAUSS_SPACE=3
    BASIS_XI_GAUSS_VELOCITY=3
    BASIS_XI_GAUSS_PRESSURE=3
  ENDIF

  !Get command line arguments
  DO argCount=1,COMMAND_ARGUMENT_COUNT()
    CALL GET_COMMAND_ARGUMENT(argCount,ARG)
    SELECT CASE(ARG)
    CASE('-lagrange')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) LagrangeBasisFlag
    CASE('-density')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) RHO_PARAM_NAVIER_STOKES
    CASE('-viscosity')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) MU_PARAM_NAVIER_STOKES
    CASE('-directsolver')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) LINEAR_SOLVER_NAVIER_STOKES_DIRECTFlag
    CASE('-dynamic')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKESFlag
    CASE('-ALE')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) ALE_SOLVER_NAVIER_STOKESFlag
    CASE('-outfrequency')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY
    CASE('-SUPG')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) SUPGFlag
    CASE('-resume')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKES_RESUME_SOLVEFlag
    CASE('-stokes')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) StokesFlag
    CASE('-starttime')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME
    CASE('-stoptime')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME
    CASE('-timeincrement')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT
    CASE('-velocity')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) BOUNDARY_CONDITIONS_NAVIER_STOKES(1)
      CALL GET_COMMAND_ARGUMENT(argCount+2,BUFFER)
      READ(BUFFER,*) BOUNDARY_CONDITIONS_NAVIER_STOKES(2)
      CALL GET_COMMAND_ARGUMENT(argCount+3,BUFFER)
      READ(BUFFER,*) BOUNDARY_CONDITIONS_NAVIER_STOKES(3)
    CASE('-setoutlet')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) setOutletPressureOutletFlag
    CASE('-pressurevalue')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) OutletPressureValue
    CASE('-mesh')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) NUMBER_GLOBAL_X_ELEMENTS
      CALL GET_COMMAND_ARGUMENT(argCount+2,BUFFER)
      READ(BUFFER,*) NUMBER_GLOBAL_Y_ELEMENTS
      CALL GET_COMMAND_ARGUMENT(argCount+3,BUFFER)
      READ(BUFFER,*) NUMBER_GLOBAL_Z_ELEMENTS
    CASE('-order')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) quadratureOrder
    CASE('-cellml')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) cellmlFlag
    CASE('-equationsout')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) equationsOutputFlag
    CASE('-fixed')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) fixedMeshFlag
    CASE('-dimension')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) numberOfDimensions 
    CASE('-resistance')
      CALL GET_COMMAND_ARGUMENT(argCount+1,BUFFER)
      READ(BUFFER,*) resistanceProximal
    CASE DEFAULT
      !do nothing
    END SELECT
  ENDDO
  WRITE(*,*)' '
  WRITE(*,*)' ************************************* '
  WRITE(*,*)' '
  WRITE(*,*)'-density........', RHO_PARAM_NAVIER_STOKES
  WRITE(*,*)'-viscosity......', MU_PARAM_NAVIER_STOKES
  WRITE(*,*)'-SUPG.......  ', SUPGFlag
  WRITE(*,*)'-cellml.......  ', cellmlFlag
  WRITE(*,*)'-velocity.......', BOUNDARY_CONDITIONS_NAVIER_STOKES
  WRITE(*,*)'-dynamic........  ', DYNAMIC_SOLVER_NAVIER_STOKESFlag
  WRITE(*,*)'  -ALE............  ', ALE_SOLVER_NAVIER_STOKESFlag
  IF(DYNAMIC_SOLVER_NAVIER_STOKESFlag) THEN
    WRITE(*,*) ' ' 
    WRITE(*,*)'  -resume............  ', DYNAMIC_SOLVER_NAVIER_STOKES_RESUME_SOLVEFlag
    WRITE(*,*)'  -starttime......  ', DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME
    WRITE(*,*)'  -stoptime.......  ', DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME
    WRITE(*,*)'  -timeincrement..  ', DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT
    WRITE(*,*)'  -outputfrequency..  ', DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY
    WRITE(*,*) ' ' 
  ENDIF
  WRITE(*,*)'-directsolver...  ', LINEAR_SOLVER_NAVIER_STOKES_DIRECTFlag
  WRITE(*,*)' '
  WRITE(*,*)' ************************************* '
  WRITE(*,*)' '
  WRITE(*,*) ' ' 

  !Set interpolation order, basis type, and number of dimensions here
  !--------------------------------------------------------------------------------------------------------------------------------
  IF(LagrangeBasisFlag) THEN
    BASIS_XI_INTERPOLATION_SPACE=CMISS_BASIS_QUADRATIC_LAGRANGE_INTERPOLATION
    BASIS_XI_INTERPOLATION_VELOCITY=CMISS_BASIS_QUADRATIC_LAGRANGE_INTERPOLATION
    BASIS_XI_INTERPOLATION_PRESSURE=CMISS_BASIS_LINEAR_LAGRANGE_INTERPOLATION
    basisType=CMISS_BASIS_LAGRANGE_HERMITE_TP_TYPE 
  ELSE
    BASIS_XI_INTERPOLATION_SPACE=CMISS_BASIS_QUADRATIC_SIMPLEX_INTERPOLATION
    BASIS_XI_INTERPOLATION_VELOCITY=CMISS_BASIS_QUADRATIC_SIMPLEX_INTERPOLATION
    BASIS_XI_INTERPOLATION_PRESSURE=CMISS_BASIS_LINEAR_SIMPLEX_INTERPOLATION
    basisType=CMISS_BASIS_SIMPLEX_TYPE
  ENDIF
  MESH_COMPONENT_NUMBER_SPACE = 1
  MESH_COMPONENT_NUMBER_VELOCITY = 1
  MESH_COMPONENT_NUMBER_PRESSURE = 2

  HEIGHT=1.0_CMISSDP
  WIDTH=1.0_CMISSDP
  LENGTH=3.0_CMISSDP

  !--------------------------------------------------------------------------------------------------------------------------------
 
  !Get boundary info
  IF(LagrangeBasisFlag) THEN
    INQUIRE(FILE="./input/hexCylinder1/bc/Wall", EXIST=fixedWallNodesFLAG)
    IF(fixedWallNodesFlag) THEN
      OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Wall",STATUS='unknown')
      READ(1,*) numberOfFixedWallNodes
      ALLOCATE(fixedWallNodes(numberOfFixedWallNodes))
      READ(1,*) fixedWallNodes(1:numberOfFixedWallNodes)
      CLOSE(1)
    ENDIF

    ! Inlet
    INQUIRE(FILE="./input/hexCylinder1/bc/Inlet_nodes", EXIST=boundaryNodesInletFlag)
    IF(boundaryNodesInletFlag) THEN
       OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Inlet_nodes",STATUS='unknown')
      READ(1,*) numberOfBoundaryNodesInlet
      ALLOCATE(boundaryNodesInlet(numberOfBoundaryNodesInlet))
      READ(1,*) boundaryNodesInlet(1:numberOfBoundaryNodesInlet)
      CLOSE(1)
    ENDIF
    INQUIRE(FILE="./input/hexCylinder1/bc/Inlet_elements", EXIST=boundaryElementsInletFlag)
    IF(boundaryElementsInletFlag) THEN
       OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Inlet_elements",STATUS='unknown')
      READ(1,*) numberOfBoundaryElementsInlet
      ALLOCATE(boundaryElementsInlet(numberOfBoundaryElementsInlet))
      READ(1,*) boundaryElementsInlet(1:numberOfBoundaryElementsInlet)
      CLOSE(1)
    ENDIF

    ! outlet
    ! INQUIRE(FILE="./input/hexCylinder1/bc/Outlet_nodes", EXIST=quadraticBoundaryNodesOutletFlag)
    ! IF(quadraticBoundaryNodesOutletFlag) THEN
    !    OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Outlet_nodes",STATUS='unknown')
    !   READ(1,*) numberOfQuadraticBoundaryNodesOutlet
    !   ALLOCATE(quadraticBoundaryNodesOutlet(numberOfQuadraticBoundaryNodesOutlet))
    !   READ(1,*) quadraticBoundaryNodesOutlet(1:numberOfQuadraticBoundaryNodesOutlet)
    !   CLOSE(1)
    ! ENDIF
    INQUIRE(FILE="./input/hexCylinder1/bc/Outlet_nodes", EXIST=linearBoundaryNodesOutletFlag)
    IF(linearBoundaryNodesOutletFlag) THEN
       OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Outlet_nodes",STATUS='unknown')
      READ(1,*) numberOfLinearBoundaryNodesOutlet
      ALLOCATE(linearBoundaryNodesOutlet(numberOfLinearBoundaryNodesOutlet))
      READ(1,*) linearBoundaryNodesOutlet(1:numberOfLinearBoundaryNodesOutlet)
      CLOSE(1)
    ENDIF
    INQUIRE(FILE="./input/hexCylinder1/bc/Outlet_elements", EXIST=boundaryElementsOutletFlag)
    IF(boundaryElementsOutletFlag) THEN
       OPEN(UNIT=1, FILE="./input/hexCylinder1/bc/Outlet_elements",STATUS='unknown')
      READ(1,*) numberOfBoundaryElementsOutlet
      ALLOCATE(boundaryElementsOutlet(numberOfBoundaryElementsOutlet))
      READ(1,*) boundaryElementsOutlet(1:numberOfBoundaryElementsOutlet)
      CLOSE(1)
    ENDIF
  ENDIF

  !Set output parameter
  !(NoOutput/ProgressOutput/TimingOutput/SolverOutput/MatrixOutput)
  DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE=CMISS_SOLVER_Timing_OUTPUT
  LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE=CMISS_SOLVER_NO_OUTPUT
!  NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE=CMISS_SOLVER_Matrix_Output
  NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE=CMISS_SOLVER_Progress_Output
  !(NoOutput/TimingOutput/MatrixOutput/ElementOutput)
  IF(equationsOutputFlag) THEN
    EQUATIONS_NAVIER_STOKES_OUTPUT=CMISS_EQUATIONS_matrix_OUTPUT
  ELSE
    EQUATIONS_NAVIER_STOKES_OUTPUT=CMISS_EQUATIONS_NO_OUTPUT
  ENDIF


  IF(cellmlFlag) THEN
    IF(windkesselFlag) THEN
      ProblemSubtype=CMISS_PROBLEM_Coupled3DDAE_NAVIER_STOKES_SUBTYPE
      SolverDAEUserNumber=1
      SolverNavierStokesUserNumber=2
    ELSE
      ProblemSubtype=CMISS_Problem_Transient_SUPG_Navier_Stokes_CMM_Subtype
      SolverNavierStokesUserNumber=1
    ENDIF
    EquationsSetSubtype=CMISS_Equations_Set_Transient_SUPG_Navier_Stokes_CMM_Subtype
  ELSE
    EquationsSetSubtype=CMISS_EQUATIONS_SET_TRANSIENT_NAVIER_STOKES_SUBTYPE
    ProblemSubtype=CMISS_PROBLEM_TRANSIENT_NAVIER_STOKES_SUBTYPE
    SolverNavierStokesUserNumber=1
  ENDIF

  !================================================================================================================================
  !Intialise OpenCMISS
  !================================================================================================================================

  CALL CMISSInitialise(WorldCoordinateSystem,WorldRegion,Err)
  CALL CMISSErrorHandlingModeSet(CMISS_ERRORS_TRAP_ERROR,Err)

  !Get the computational nodes information
  CALL CMISSComputationalNumberOfNodesGet(NumberOfComputationalNodes,Err)
  CALL CMISSComputationalNodeNumberGet(ComputationalNodeNumber,Err)
       
  !INITIALISE FieldML
  CALL CMISSFieldMLIO_Initialise( fieldmlInfo, err ) 
  CALL CMISSFieldML_InputCreateFromFile( inputFilename, fieldmlInfo, err )

  !================================================================================================================================
  !Coordinate System
  !================================================================================================================================

  !Start the creation of a new RC coordinate system
  CALL CMISSCoordinateSystem_Initialise(CoordinateSystem,Err)
  IF(fixedMeshFlag) THEN
    CALL CMISSFieldML_InputCoordinateSystemCreateStart( fieldmlInfo, "CylinderMesh.coordinates", CoordinateSystem, &
      & CoordinateSystemUserNumber, err )
    CALL CMISSCoordinateSystem_CreateFinish( CoordinateSystem, err )
    CALL CMISSCoordinateSystem_DimensionGet( CoordinateSystem, coordinateCount, err )
  ELSE
    CALL CMISSCoordinateSystem_CreateStart(CoordinateSystemUserNumber,CoordinateSystem,Err)
    IF(numberOfDimensions==2) THEN
      !Set the coordinate system to be 2D
      CALL CMISSCoordinateSystem_DimensionSet(CoordinateSystem,2,Err)
    ELSE
      !Set the coordinate system to be 3D
      CALL CMISSCoordinateSystem_DimensionSet(CoordinateSystem,3,Err)
    ENDIF
    !Finish the creation of the coordinate system
    CALL CMISSCoordinateSystem_CreateFinish(CoordinateSystem,Err)
  ENDIF

  !================================================================================================================================
  !Region
  !================================================================================================================================

  !Start the creation of the region
  CALL CMISSRegion_Initialise(Region,Err)
  CALL CMISSRegion_CreateStart(RegionUserNumber,WorldRegion,Region,Err)
  !Set the regions coordinate system to the RC coordinate system that we have created
  CALL CMISSRegion_CoordinateSystemSet(Region,CoordinateSystem,Err)
  !Finish the creation of the region
  CALL CMISSRegion_CreateFinish(Region,Err)

  !================================================================================================================================
  ! Data Points
  !================================================================================================================================

  ! !Intialise 5 data points
  ! dataPointValues(1,:)=(/0.1_CMISSDP,0.8_CMISSDP,1.0_CMISSDP/)
  ! dataPointValues(2,:)=(/0.5_CMISSDP,0.5_CMISSDP,0.5_CMISSDP/)  
  ! dataPointValues(3,:)=(/0.2_CMISSDP,0.5_CMISSDP,0.5_CMISSDP/)  
  ! dataPointValues(4,:)=(/0.9_CMISSDP,0.6_CMISSDP,0.9_CMISSDP/)
  ! dataPointValues(5,:)=(/0.3_CMISSDP,0.3_CMISSDP,0.3_CMISSDP/)
  ! numberOfDataPoints=SIZE(dataPointValues,1)

  ! !Create Data Points and set the values
  ! CALL CMISSDataPoints_CreateStart(RegionUserNumber,SIZE(dataPointValues,1),Err)
  ! DO dataPointIdx=1,numberOfDataPoints
  !   CALL CMISSDataPoints_ValuesSet(RegionUserNumber,dataPointIdx,dataPointValues(dataPointIdx,:),Err)     
  ! ENDDO
  ! CALL CMISSDataPoints_CreateFinish(RegionUserNumber,Err)  

  !================================================================================================================================
  !Nodes
  !================================================================================================================================

  CALL CMISSFieldML_InputNodesCreateStart( fieldmlInfo, "CylinderMesh.nodes.argument", Region, nodes, err )
  CALL CMISSNodes_CreateFinish( Nodes, err )

  !================================================================================================================================
  !Bases
  !================================================================================================================================
  MESH_NUMBER_OF_COMPONENTS=1

  CALL CMISSFieldML_InputBasisCreateStart( fieldmlInfo, "CylinderMesh.trilinear_lagrange", basisNumberTrilinear, err )
  CALL CMISSBasis_QuadratureNumberOfGaussXiSet( basisNumberTrilinear, gaussQuadrature, err )
  IF(calculateElementFacesFlag) THEN
    CALL CMISSBasis_QuadratureLocalFaceGaussEvaluateSet (basisNumberTrilinear,calculateElementFacesFlag,err)
  END IF

  CALL CMISSFieldML_InputBasisCreateStart( fieldmlInfo, "CylinderMesh.triquadratic_lagrange", basisNumberTriquadratic, err )
  CALL CMISSBasis_QuadratureNumberOfGaussXiSet( basisNumberTriquadratic, gaussQuadrature, err )
  IF(calculateElementFacesFlag) THEN
    CALL CMISSBasis_QuadratureLocalFaceGaussEvaluateSet (basisNumberTriquadratic,calculateElementFacesFlag,err)
  END IF
  CALL CMISSBasis_CreateFinish( basisNumberTriquadratic, err )  

  CALL CMISSBasis_CreateFinish( basisNumberTrilinear, err )

  !================================================================================================================================
  ! Mesh
  !================================================================================================================================

  meshComponentCount = 2

  CALL CMISSFieldML_InputMeshCreateStart( fieldmlInfo, "CylinderMesh.mesh.argument", Mesh, MeshUserNumber, Region, err )
  CALL CMISSMesh_NumberOfComponentsSet( Mesh, meshComponentCount, err )

  CALL CMISSFieldML_InputCreateMeshComponent( fieldmlInfo, RegionUserNumber, MeshUserNumber, 1, &
    & "CylinderMesh.template.triquadratic", err )
  CALL CMISSFieldML_InputCreateMeshComponent( fieldmlInfo, RegionUserNumber, MeshUserNumber, 2, &
    & "CylinderMesh.template.trilinear", err )

  MESH_COMPONENT_NUMBER_SPACE = 1
  MESH_COMPONENT_NUMBER_VELOCITY = 1
  MESH_COMPONENT_NUMBER_PRESSURE = 2

  !Finish the creation of the mesh
  CALL CMISSMesh_CreateFinish(Mesh, err )

  !================================================================================================================================
  !Decomposition
  !================================================================================================================================

  !Create a decomposition
  CALL CMISSDecomposition_Initialise(Decomposition,Err)
  CALL CMISSDecomposition_CreateStart(DecompositionUserNumber,Mesh,Decomposition,Err)
  !Set the decomposition to be a general decomposition with the specified number of domains
  CALL CMISSDecomposition_TypeSet(Decomposition,CMISS_DECOMPOSITION_CALCULATED_TYPE,Err)
  CALL CMISSDecomposition_NumberOfDomainsSet(Decomposition,NumberOfComputationalNodes,Err)
  CALL CMISSDecomposition_CalculateFacesSet(Decomposition,calculateElementFacesFlag,Err)
  !Finish the decomposition
  CALL CMISSDecomposition_CreateFinish(Decomposition,Err)
  
  !================================================================================================================================
  !Geometric Field
  !================================================================================================================================

  CALL CMISSFieldML_InputFieldCreateStart( fieldmlInfo, Region, Decomposition, GeometricFieldUserNumber, GeometricField, &
    & CMISS_FIELD_U_VARIABLE_TYPE, "CylinderMesh.coordinates", err )
  CALL CMISSField_CreateFinish( RegionUserNumber, GeometricFieldUserNumber, err )
  CALL CMISSFieldML_InputFieldParametersUpdate( fieldmlInfo, GeometricField, "CylinderMesh.node.coordinates", &
    & CMISS_FIELD_U_VARIABLE_TYPE, CMISS_FIELD_VALUES_SET_TYPE, err )
  CALL CMISSField_ParameterSetUpdateStart( GeometricField, CMISS_FIELD_U_VARIABLE_TYPE, CMISS_FIELD_VALUES_SET_TYPE, err )
  CALL CMISSField_ParameterSetUpdateFinish( GeometricField, CMISS_FIELD_U_VARIABLE_TYPE, CMISS_FIELD_VALUES_SET_TYPE, err )
  CALL CMISSFieldMLIO_Finalise( fieldmlInfo, err )

  !================================================================================================================================
  !Equations Sets
  !================================================================================================================================

  !Create the equations set for transient SUPG Navier-Stokes
  CALL CMISSEquationsSet_Initialise(EquationsSetNavierStokes, err )
  CALL CMISSField_Initialise(EquationsSetField,Err)
  CALL CMISSEquationsSet_CreateStart(EquationsSetUserNumberNavierStokes,Region,GeometricField, &
    & CMISS_EQUATIONS_SET_FLUID_MECHANICS_CLASS,CMISS_EQUATIONS_SET_NAVIER_STOKES_EQUATION_TYPE, &
    & EquationsSetSubtype,EquationsSetFieldUserNumber,EquationsSetField,EquationsSetNavierStokes,Err)   
  
  !Finish creating the equations set
  CALL CMISSEquationsSet_CreateFinish(EquationsSetNavierStokes, err) 

  !================================================================================================================================
  !Dependent Fields
  !================================================================================================================================

  !Create the equations set dependent field variables for transient Navier-Stokes
  CALL CMISSField_Initialise(DependentFieldNavierStokes, err )
  CALL CMISSEquationsSet_DependentCreateStart(EquationsSetNavierStokes,DependentFieldUserNumberNavierStokes, & 
    & DependentFieldNavierStokes, err )

  !Set the mesh component to be used by the field components.

  ! Component 1: Quadratic Mesh
  DO COMPONENT_NUMBER=1,numberOfDimensions
    ! Velocity vector (U Variable type)
    CALL CMISSField_ComponentMeshComponentSet(DependentFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, & 
      & MESH_COMPONENT_NUMBER_VELOCITY, err )
    ! dU/dN
    CALL CMISSField_ComponentMeshComponentSet(DependentFieldNavierStokes,CMISS_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, & 
      & MESH_COMPONENT_NUMBER_VELOCITY, err )
  ENDDO
  ! Component 2: Linear Mesh: Pressure
  COMPONENT_NUMBER=numberOfDimensions+1
  CALL CMISSField_ComponentMeshComponentSet(DependentFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,COMPONENT_NUMBER, & 
    & MESH_COMPONENT_NUMBER_PRESSURE, err )
  CALL CMISSField_ComponentMeshComponentSet(DependentFieldNavierStokes,CMISS_FIELD_DELUDELN_VARIABLE_TYPE,COMPONENT_NUMBER, & 
    & MESH_COMPONENT_NUMBER_PRESSURE, err )

  !Finish the equations set dependent field variables
  CALL CMISSEquationsSet_DependentCreateFinish(EquationsSetNavierStokes, err )

  !Initialise velocity field
  DO COMPONENT_NUMBER=1,numberOfDimensions
    CALL CMISSField_ComponentValuesInitialise(DependentFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
      & COMPONENT_NUMBER,INITIAL_FIELD_NAVIER_STOKES(COMPONENT_NUMBER), err )
  ENDDO

  !================================================================================================================================
  !Materials Field
  !================================================================================================================================

  !Create the equations set materials field variables for transient Navier-Stokes
  CALL CMISSField_Initialise(MaterialsFieldNavierStokes, err )
  CALL CMISSEquationsSet_MaterialsCreateStart(EquationsSetNavierStokes,MaterialsFieldUserNumberNavierStokes, & 
    & MaterialsFieldNavierStokes, err )
  !Finish the equations set materials field variables
  CALL CMISSEquationsSet_MaterialsCreateFinish(EquationsSetNavierStokes, err )
  CALL CMISSField_ComponentValuesInitialise(MaterialsFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & MaterialsFieldUserNumberNavierStokesMu,MU_PARAM_NAVIER_STOKES, err )
  CALL CMISSField_ComponentValuesInitialise(MaterialsFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & MaterialsFieldUserNumberNavierStokesRho,RHO_PARAM_NAVIER_STOKES, err )

  !================================================================================================================================
  !Analytic Field - apply a sinusoidal inlet boundary condition
  !================================================================================================================================
  !Create the equations set analytic field variables
  CALL CMISSField_Initialise(AnalyticFieldNavierStokes,Err)
  CALL CMISSEquationsSet_AnalyticCreateStart(EquationsSetNavierStokes,CMISS_EQUATIONS_SET_NAVIER_STOKES_EQUATION_Sinusoid, &
    & AnalyticFieldUserNumber,AnalyticFieldNavierStokes,Err)
  !Finish the equations set analytic field variables
  CALL CMISSEquationsSet_AnalyticCreateFinish(EquationsSetNavierStokes,Err)
  !Set analytic field variables
  ! inlet boundary normals
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 1,-normalValueInlet(1),Err)
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 2,-normalValueInlet(2),Err)
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 3,-normalValueInlet(3),Err)
  ! Period
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 4,2.0_CMISSDP,Err)
  ! amplitude
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 5,1.0_CMISSDP,Err)
  ! offset
  CALL CMISSField_ComponentValuesInitialise(AnalyticFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE, & 
    & 6,1.0_CMISSDP,Err)


  !================================================================================================================================
  !  C e l l M L    M o d e l    M a p s
  !================================================================================================================================

  IF (cellmlFlag) THEN

    !Create the CellML environment
    CALL CMISSCellML_Initialise(CellML,Err)
    CALL CMISSCellML_CreateStart(CellMLUserNumber,Region,CellML,Err)

    IF(windkesselFlag) THEN
      CALL CMISSCellML_ModelImport(CellML, &
       & "/hpc/dlad004/opencmiss/examples/FluidMechanics/NavierStokes/CellMLModels/Windkessel/WindkesselMain.cellml", &
       & CellMLModelIndex,Err)
    ELSE
      CALL CMISSCellML_ModelImport(CellML, &
       & "/hpc/dlad004/opencmiss/examples/FluidMechanics/NavierStokes/CellMLModels/Resistance/resistance.xml", &
!       & "/hpc/dlad004/opencmiss/examples/FluidMechanics/NavierStokes/CellMLModels/Resistance/p100.xml", &
       & CellMLModelIndex,Err)
    ENDIF
    ! - known (to OpenCMISS) variables 
    CALL CMISSCellML_VariableSetAsKnown(CellML,CellMLModelIndex,"interface/FlowRate",Err)
    !   - to get from the CellML side
    CALL CMISSCellML_VariableSetAsWanted(CellML,CellMLModelIndex,"interface/Pressure",Err)
    CALL CMISSCellML_CreateFinish(CellML,Err)
    !Start the creation of CellML <--> OpenCMISS field maps
    CALL CMISSCellML_FieldMapsCreateStart(CellML,Err)
    !Now we can set up the field variable component <--> CellML model variable mappings.
    !Map the CMISS boundary flow rate values --> CellML
    CALL CMISSCellML_CreateFieldToCellMLMap(CellML,EquationsSetField,CMISS_FIELD_U_VARIABLE_TYPE,13, &
      & CMISS_FIELD_VALUES_SET_TYPE,CellMLModelIndex,"interface/FlowRate",CMISS_FIELD_VALUES_SET_TYPE,Err)
    !Map the returned pressure values from CellML --> CMISS
    CALL CMISSCellML_CreateCellMLToFieldMap(CellML,CellMLModelIndex,"interface/Pressure",CMISS_FIELD_VALUES_SET_TYPE, &
      & DependentFieldNavierStokes,CMISS_FIELD_U_VARIABLE_TYPE,4,CMISS_FIELD_VALUES_SET_TYPE,Err)

    !Finish the creation of CellML <--> OpenCMISS field maps
    CALL CMISSCellML_FieldMapsCreateFinish(CellML,Err)

    ! Initialise flow rate field to 0
    CALL CMISSField_ComponentValuesInitialise(EquationsSetField,CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,13, &
     & 0.0_CMISSDP,Err)

    !Create the CellML models field --- only 1 model here
    CALL CMISSField_Initialise(CellMLModelsField,Err)
    CALL CMISSCellML_ModelsFieldCreateStart(CellML,CellMLModelsFieldUserNumber,CellMLModelsField,Err)
    CALL CMISSCellML_ModelsFieldCreateFinish(CellML,Err)

    IF (windkesselFlag) THEN 
      !Start the creation of the CellML state field
      CALL CMISSField_Initialise(CellMLStateField,Err)
      CALL CMISSCellML_StateFieldCreateStart(CellML,CellMLStateFieldUserNumber,CellMLStateField,Err)
      !Finish the creation of the CellML state field
      CALL CMISSCellML_StateFieldCreateFinish(CellML,Err)
    ENDIF

    !Create the CellML parameters field --- will be the Flow rate
    CALL CMISSField_Initialise(CellMLParametersField,Err)
    CALL CMISSCellML_ParametersFieldCreateStart(CellML,CellMLParametersFieldUserNumber,CellMLParametersField,Err)
    CALL CMISSCellML_ParametersFieldCreateFinish(CellML,Err)

    !Create the CellML intermediate field --- will be the pressure value returned from CellML to be mapped to the dependent field
    CALL CMISSField_Initialise(CellMLIntermediateField,Err)
    CALL CMISSCellML_IntermediateFieldCreateStart(CellML,CellMLIntermediateFieldUserNumber,CellMLIntermediateField,Err)
    CALL CMISSCellML_IntermediateFieldCreateFinish(CellML,Err)
 
  ENDIF ! cellml flag

  !================================================================================================================================
  !Equations
  !================================================================================================================================

  !Create the equations set equations
  CALL CMISSEquations_Initialise(EquationsNavierStokes, err )
  CALL CMISSEquationsSet_EquationsCreateStart(EquationsSetNavierStokes,EquationsNavierStokes, err )
  !Set the equations matrices sparsity type
  CALL CMISSEquations_SparsityTypeSet(EquationsNavierStokes,CMISS_EQUATIONS_SPARSE_MATRICES, err )
  !Set the equations set output
  CALL CMISSEquations_OutputTypeSet(EquationsNavierStokes,EQUATIONS_NAVIER_STOKES_OUTPUT, err )
  !Finish the equations set equations
  CALL CMISSEquationsSet_EquationsCreateFinish(EquationsSetNavierStokes, err )

  !================================================================================================================================
  !Problems
  !================================================================================================================================

  !Start the creation of a problem.
  CALL CMISSProblem_Initialise(Problem, err )
  CALL CMISSControlLoop_Initialise(ControlLoop, err )
  CALL CMISSProblem_CreateStart(ProblemUserNumber,Problem, err )
  !Set the problem to be a transient Navier-Stokes problem
  CALL CMISSProblem_SpecificationSet(Problem,CMISS_PROBLEM_FLUID_MECHANICS_CLASS,CMISS_PROBLEM_NAVIER_STOKES_EQUATION_TYPE, &
    & ProblemSubtype,Err)
  !Finish the creation of a problem.
  CALL CMISSProblem_CreateFinish(Problem, err )
  !Start the creation of the problem control loop
  CALL CMISSProblem_ControlLoopCreateStart(Problem, err )
  !Get the control loop
  CALL CMISSProblem_ControlLoopGet(Problem,CMISS_CONTROL_LOOP_NODE,ControlLoop,Err)
  !Set the times
  CALL CMISSControlLoop_TimesSet(ControlLoop,DYNAMIC_SOLVER_NAVIER_STOKES_START_TIME,DYNAMIC_SOLVER_NAVIER_STOKES_STOP_TIME, & 
    & DYNAMIC_SOLVER_NAVIER_STOKES_TIME_INCREMENT,Err)
  !Set the output timing
  CALL CMISSControlLoop_TimeOutputSet(ControlLoop,DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_FREQUENCY,Err)
  !Finish creating the problem control loop
  CALL CMISSProblem_ControlLoopCreateFinish(Problem, err )

  !================================================================================================================================
  !Solvers
  !================================================================================================================================

  CALL CMISSProblem_SolversCreateStart(Problem, err )

  ! CellML DAE solver
  IF (windkesselFlag) THEN
    CALL CMISSSolver_Initialise(CellMLSolver,Err)
    CALL CMISSProblem_SolverGet(Problem,CMISS_CONTROL_LOOP_NODE,SolverDAEUserNumber,CellMLSolver,Err)
    CALL CMISSSolver_DAETimeStepSet(CellMLSolver,0.001_CMISSDP,Err)
    CALL CMISSSolver_OutputTypeSet(CellMLSolver,CMISS_SOLVER_NO_OUTPUT,Err)
  ENDIF

  ! 3D Navier-Stokes solver
  !Start the creation of the problem solvers
  CALL CMISSSolver_Initialise(DynamicSolverNavierStokes,Err)
  CALL CMISSSolver_Initialise(NonlinearSolverNavierStokes, err )
  CALL CMISSSolver_Initialise(LinearSolverNavierStokes, err )
  !Get the dynamic solver
  CALL CMISSProblem_SolverGet(Problem,CMISS_CONTROL_LOOP_NODE,SolverNavierStokesUserNumber,DynamicSolverNavierStokes,Err)
  !Set the output type
  CALL CMISSSolver_OutputTypeSet(DynamicSolverNavierStokes,DYNAMIC_SOLVER_NAVIER_STOKES_OUTPUT_TYPE,Err)
  !Set theta
  CALL CMISSSolver_DynamicThetaSet(DynamicSolverNavierStokes,DYNAMIC_SOLVER_NAVIER_STOKES_THETA,Err)
  !Get the dynamic nonlinear solver
  CALL CMISSSolver_DynamicNonlinearSolverGet(DynamicSolverNavierStokes,NonlinearSolverNavierStokes,Err)
  !Set the nonlinear Jacobian type
  CALL CMISSSolver_NewtonJacobianCalculationTypeSet(NonlinearSolverNavierStokes, &
    & CMISS_SOLVER_NEWTON_JACOBIAN_EQUATIONS_CALCULATED,Err)
  CALL CMISSSolver_OutputTypeSet(NonlinearSolverNavierStokes,NONLINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE,Err)
  !Set the solver settings
  CALL CMISSSolver_NewtonAbsoluteToleranceSet(NonlinearSolverNavierStokes,ABSOLUTE_TOLERANCE,Err)
  CALL CMISSSolver_NewtonRelativeToleranceSet(NonlinearSolverNavierStokes,RELATIVE_TOLERANCE,Err)
  CALL CMISSSolver_NewtonSolutionToleranceSet(NonlinearSolverNavierStokes,SOLUTION_TOLERANCE,Err)
  CALL CMISSSolver_NewtonMaximumIterationsSet(NonlinearSolverNavierStokes,MAXIMUM_ITERATIONS,Err)
  !Get the dynamic nonlinear linear solver
  CALL CMISSSolver_NewtonLinearSolverGet(NonlinearSolverNavierStokes,LinearSolverNavierStokes,Err)
  !Set the output type
  CALL CMISSSolver_OutputTypeSet(LinearSolverNavierStokes,LINEAR_SOLVER_NAVIER_STOKES_OUTPUT_TYPE,Err)
  !Set the solver settings
  IF(LINEAR_SOLVER_NAVIER_STOKES_DIRECTFlag) THEN
    CALL CMISSSolver_LinearTypeSet(LinearSolverNavierStokes,CMISS_SOLVER_LINEAR_DIRECT_SOLVE_TYPE,Err)
    CALL CMISSSolver_LibraryTypeSet(LinearSolverNavierStokes,CMISS_SOLVER_MUMPS_LIBRARY,Err)
  ELSE
    CALL CMISSSolver_LinearTypeSet(LinearSolverNavierStokes,CMISS_SOLVER_LINEAR_ITERATIVE_SOLVE_TYPE,Err)
    CALL CMISSSolver_LinearIterativeMaximumIterationsSet(LinearSolverNavierStokes,MAXIMUM_ITERATIONS,Err)
    CALL CMISSSolver_LinearIterativeDivergenceToleranceSet(LinearSolverNavierStokes,DIVERGENCE_TOLERANCE,Err)
    CALL CMISSSolver_LinearIterativeRelativeToleranceSet(LinearSolverNavierStokes,RELATIVE_TOLERANCE,Err)
    CALL CMISSSolver_LinearIterativeAbsoluteToleranceSet(LinearSolverNavierStokes,ABSOLUTE_TOLERANCE,Err)
    CALL CMISSSolver_LinearIterativeGMRESRestartSet(LinearSolverNavierStokes,RESTART_VALUE,Err)
  ENDIF
  !Finish the creation of the problem solver
  CALL CMISSProblem_SolversCreateFinish(Problem,Err)

  !================================================================================================================================
  !Solver Equations
  !================================================================================================================================

  IF(cellmlFlag) THEN
    !Start the creation of the problem solver CellML equations
    CALL CMISSSolver_Initialise(CellMLSolver,Err)
    CALL CMISSCellMLEquations_Initialise(CellMLEquations,Err)
    CALL CMISSProblem_CellMLEquationsCreateStart(Problem,Err)
    IF(windkesselFlag) THEN
      !Get the DAE solver  
      CALL CMISSProblem_SolverGet(Problem,CMISS_CONTROL_LOOP_NODE,SolverDAEUserNumber,CellMLSolver,Err)
    ELSE
      CALL CMISSSolver_NewtonCellMLSolverGet(DynamicSolverNavierStokes,CellMLSolver,Err) 
    ENDIF
    !Get the CellML equations
    CALL CMISSSolver_CellMLEquationsGet(CellMLSolver,CellMLEquations,Err)
    !Add in the CellML environement
    CALL CMISSCellMLEquations_CellMLAdd(CellMLEquations,CellML,CellMLModelIndex,Err)
    !Finish the creation of the problem solver CellML equations
    CALL CMISSProblem_CellMLEquationsCreateFinish(Problem,Err)
  ENDIF

  !Start the creation of the problem solver equations
  CALL CMISSSolver_Initialise(DynamicSolverNavierStokes,Err)
  CALL CMISSSolverEquations_Initialise(SolverEquationsNavierStokes, err )
  CALL CMISSProblem_SolverEquationsCreateStart(Problem, err )
  !Get the dynamic solver equations
  CALL CMISSProblem_SolverGet(Problem,CMISS_CONTROL_LOOP_NODE,SolverNavierStokesUserNumber,DynamicSolverNavierStokes,Err)
  CALL CMISSSolver_SolverEquationsGet(DynamicSolverNavierStokes,SolverEquationsNavierStokes,Err)
  !Set the solver equations sparsity
  CALL CMISSSolverEquations_SparsityTypeSet(SolverEquationsNavierStokes,CMISS_SOLVER_SPARSE_MATRICES,Err)
  !Add in the equations set
  CALL CMISSSolverEquations_EquationsSetAdd(SolverEquationsNavierStokes,EquationsSetNavierStokes,EquationsSetIndex,Err)
  !Finish the creation of the problem solver equations
  CALL CMISSProblem_SolverEquationsCreateFinish(Problem,Err)

  !================================================================================================================================
  !Boundary Conditions
  !================================================================================================================================

  !Start the creation of the equations set boundary conditions for Navier-Stokes
  CALL CMISSBoundaryConditions_Initialise(BoundaryConditionsNavierStokes, err )
  CALL CMISSSolverEquations_BoundaryConditionsCreateStart(SolverEquationsNavierStokes,BoundaryConditionsNavierStokes, err )
  
  !Set inlet boundary conditions
  IF(boundaryNodesInletFlag) THEN
    DO nodeIdx=1,numberOfBoundaryNodesInlet
      nodeNumber=boundaryNodesInlet(nodeIdx)
      CALL CMISSDecomposition_NodeDomainGet(Decomposition,nodeNumber,MESH_COMPONENT_NUMBER_VELOCITY,BoundaryNodeDomain,Err)
      IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
!        CONDITION=CMISS_BOUNDARY_CONDITION_FIXED
        CONDITION=CMISS_BOUNDARY_CONDITION_FIXED_INLET
        DO COMPONENT_NUMBER=1,numberOfDimensions
          VALUE=BOUNDARY_CONDITIONS_NAVIER_STOKES(COMPONENT_NUMBER)
          CALL CMISSBoundaryConditions_SetNode(BoundaryConditionsNavierStokes,DependentFieldNavierStokes, &
            & CMISS_FIELD_U_VARIABLE_TYPE,1,CMISS_NO_GLOBAL_DERIV,nodeNumber,COMPONENT_NUMBER,CONDITION,VALUE, err )
        ENDDO
      ENDIF
    ENDDO
  ENDIF
  ! Set boundary normals, boundary number, Inlet boundary
  IF(boundaryElementsInletFlag) THEN
    DO elementCounter=1,numberOfBoundaryElementsInlet
      elementNumber=elementCounter
      CALL CMISSDecomposition_ElementDomainGet(Decomposition,elementNumber,BoundaryElementDomain,Err)
      IF(BoundaryElementDomain==ComputationalNodeNumber) THEN
        ! normal values
        DO COMPONENT_NUMBER=1,numberOfDimensions
          VALUE=normalValueInlet(COMPONENT_NUMBER)
          !Note: first 4 values in equations set field reserved for SUPG parameters
          CALL CMISSField_ParameterSetUpdateElement(EquationsSetField, &
            & CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,elementNumber,COMPONENT_NUMBER+4,VALUE,err)
        ENDDO ! component_number
        ! boundary identifier
        boundaryID=1.0_CMISSDP
        CALL CMISSField_ParameterSetUpdateElement(EquationsSetField, &
          & CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,elementNumber,8,boundaryID,err)
        CALL CMISSField_ParameterSetUpdateElement(EquationsSetField, &
          & CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,elementNumber,10,resistanceProximal,err)
      ENDIF
    ENDDO
  ENDIF

  !Set fixed wall nodes
  IF(fixedWallNodesFlag) THEN
    DO nodeIdx=1,numberOfFixedWallNodes
      nodeNumber=fixedWallNodes(nodeIdx)
      CALL CMISSDecomposition_NodeDomainGet(Decomposition,nodeNumber,MESH_COMPONENT_NUMBER_VELOCITY,BoundaryNodeDomain,Err)
      IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
        CONDITION=CMISS_BOUNDARY_CONDITION_FIXED
        DO COMPONENT_NUMBER=1,numberOfDimensions
          VALUE=0.0_CMISSDP
          CALL CMISSBoundaryConditions_SetNode(BoundaryConditionsNavierStokes,DependentFieldNavierStokes, &
            & CMISS_FIELD_U_VARIABLE_TYPE,1,CMISS_NO_GLOBAL_DERIV,nodeNumber,COMPONENT_NUMBER,CONDITION,VALUE, err )
        ENDDO
      ENDIF
    ENDDO
  ENDIF

  !Set Outlet boundary conditions
  IF(setOutletPressureOutletFlag) THEN
    VALUE=OutletPressureValue
    COMPONENT_NUMBER=numberOfDimensions+1
    DO nodeIdx=1,numberOfLinearBoundaryNodesOutlet
      nodeNumber=linearBoundaryNodesOutlet(nodeIdx)
      CALL CMISSDecomposition_NodeDomainGet(Decomposition,nodeNumber,1,BoundaryNodeDomain,Err)
      IF(BoundaryNodeDomain==ComputationalNodeNumber) THEN
        CALL CMISSBoundaryConditions_SetNode(BoundaryConditionsNavierStokes,DependentFieldNavierStokes, &
          & CMISS_FIELD_U_VARIABLE_TYPE,1,CMISS_NO_GLOBAL_DERIV,nodeNumber,COMPONENT_NUMBER,CONDITION,VALUE, err )
      ENDIF
    ENDDO
  ENDIF
  ! Set boundary normals and boundary number for elements on Outlet boundary
  IF(boundaryElementsOutletFlag) THEN
    DO elementCounter=1,numberOfBoundaryElementsOutlet
      elementNumber=boundaryElementsOutlet(elementCounter)
      CALL CMISSDecomposition_ElementDomainGet(Decomposition,elementNumber,BoundaryElementDomain,Err)
      IF(BoundaryElementDomain==ComputationalNodeNumber) THEN
        ! normal values
        DO COMPONENT_NUMBER=1,numberOfDimensions
          VALUE=normalValueOutlet(COMPONENT_NUMBER)
          !Note: first 4 values in equations set field reserved for SUPG parameters
          CALL CMISSField_ParameterSetUpdateElement(EquationsSetField, &
            & CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,elementNumber,COMPONENT_NUMBER+4,VALUE,err)
        ENDDO
        boundaryID=2.0_CMISSDP
        CALL CMISSField_ParameterSetUpdateElement(EquationsSetField, &
          & CMISS_FIELD_U_VARIABLE_TYPE,CMISS_FIELD_VALUES_SET_TYPE,elementNumber,8,boundaryID,err)
      ENDIF
    ENDDO
  ENDIF

  !Finish the creation of the equations set boundary conditions
  CALL CMISSSolverEquations_BoundaryConditionsCreateFinish(SolverEquationsNavierStokes, err )

  !================================================================================================================================
  ! RUN SOLVERS
  !================================================================================================================================

  !Solve the problem
  WRITE(*,'(A)') "Solving problem..."
  CALL CMISSProblem_Solve(Problem, err )
  WRITE(*,'(A)') "Problem solved!"


  !================================================================================================================================
  ! OUTPUT
  !================================================================================================================================
  
  !Finialise CMISS
  CALL CMISSFinalise(Err)

  WRITE(*,'(A)') "Program successfully completed."
  
  STOP
  
END PROGRAM Coupled3D0D
