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

# -----------------------------------------------
#  Set up general problem
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

# Set diagnostics
#CMISS.DiagnosticsSetOn(CMISS.DiagnosticTypes.IN,[1,2,3,4,5],'',["SOLVER_NEWTON_LINESEARCH_SOLVE"])

# Creation a RC coordinate system
coordinateSystem = CMISS.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.dimension = 3
coordinateSystem.CreateFinish()

# Create a region
region = CMISS.Region()
region.CreateStart(regionUserNumber,CMISS.WorldRegion)
region.label = "GeneratedHex"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Create a biquadratic lagrange basis
quadraticBasis = CMISS.Basis()
quadraticBasis.CreateStart(quadraticBasisUserNumber)
quadraticBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
quadraticBasis.numberOfXi = 3
quadraticBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*3
quadraticBasis.quadratureNumberOfGaussXi = [3]*3
quadraticBasis.quadratureLocalFaceGaussEvaluate = True
quadraticBasis.CreateFinish()

# Create a bilinear lagrange basis
linearBasis = CMISS.Basis()
linearBasis.CreateStart(linearBasisUserNumber)
linearBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
linearBasis.numberOfXi = 3
linearBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*3
linearBasis.quadratureNumberOfGaussXi = [3]*3
linearBasis.quadratureLocalFaceGaussEvaluate = True
linearBasis.CreateFinish()

# Create a generated mesh
numberOfMeshComponents=2
meshComponent1 = 1
meshComponent2 = 2
meshDimensions = [1.0,1.0,2.0]
meshResolution = [3,3,4]
numberOfElements = meshResolution[0]*meshResolution[1]*meshResolution[2]
meshName = "GeneratedHex"
generatedMesh = CMISS.GeneratedMesh()
generatedMesh.CreateStart(generatedMeshUserNumber,region)
generatedMesh.type = CMISS.GeneratedMeshTypes.REGULAR
generatedMesh.basis = [quadraticBasis,quadraticBasis]
generatedMesh.extent = meshDimensions
generatedMesh.numberOfElements = meshResolution

mesh = CMISS.Mesh()
generatedMesh.CreateFinish(meshUserNumber,mesh)

# Create a decomposition for the mesh
decomposition = CMISS.Decomposition()
decomposition.CreateStart(decompositionUserNumber,mesh)
decomposition.type = CMISS.DecompositionTypes.CALCULATED
decomposition.numberOfDomains = numberOfComputationalNodes
decomposition.CalculateFacesSet(True)
decomposition.CreateFinish()

# Create a field for the geometry
geometricField = CMISS.Field()
geometricField.CreateStart(geometricFieldUserNumber,region)
geometricField.meshDecomposition = decomposition
geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
geometricField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,1)
geometricField.CreateFinish()

# Set geometry from the generated mesh
generatedMesh.GeometricParametersCalculate(geometricField)

# -----------------------------------------------
#  Solve problem with provided settings
# -----------------------------------------------
def solveProblem(transient,viscosity,density,offset,amplitude,period):
    """ Sets up the problem and solve with the provided parameter values


        Oscillatory flow through a rigid cylinder

                                     u=0
                  ------------------------------------------- R = 0.5
                                             >
                                             ->  
        p = offset + A*cos(2*pi*(t/period))  --> u(r,t)        p = 0
                                             ->
                                             >
                  ------------------------------------------- L = 10
                                     u=0
    """
    startTime = time.time()
    angularFrequency = 2.0*math.pi/period
    womersley = 0.5*math.sqrt(angularFrequency*density/viscosity)
    if computationalNodeNumber == 0:
        print("-----------------------------------------------")
        print("Setting up problem for Womersley number: " + str(womersley))
        print("-----------------------------------------------")

    # Create standard Navier-Stokes equations set
    equationsSetField = CMISS.Field()
    equationsSet = CMISS.EquationsSet()
    equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
            CMISS.EquationsSetClasses.FLUID_MECHANICS,
            CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
            CMISS.EquationsSetSubtypes.TRANSIENT_SUPG_NAVIER_STOKES,
            equationsSetFieldUserNumber, equationsSetField)
    equationsSet.CreateFinish()
    # Set beta: boundary retrograde flow stabilisation scaling factor (default 1.0)
    equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                  CMISS.FieldParameterSetTypes.VALUES, 
                                                  1,1.0)
    # Set max CFL number (default 1.0)
    equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                  CMISS.FieldParameterSetTypes.VALUES,
                                                  2,1000.0)
    # Set time increment (default 0.0)
    equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                  CMISS.FieldParameterSetTypes.VALUES,
                                                  3,transient[2])
    # Set constant C1
    equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                  CMISS.FieldParameterSetTypes.VALUES,
                                                  4,12.0)

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
        equationsSet.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.FlowrateSinusoid,analyticFieldUserNumber,analyticField)
        equationsSet.AnalyticCreateFinish()
        # Initialise analytic field parameters: (1-4) Dependent params, 5 amplitude, 6 offset, 7 period
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,3,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,4,1.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,5,amplitude)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,6,offset)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,7,period)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,8,transient[0])
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,9,transient[1])

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
                             CMISS.ProblemSubTypes.TRANSIENT_SUPG_NAVIER_STOKES)
    problem.CreateFinish()

    # Create control loops
    problem.ControlLoopCreateStart()
    controlLoop = CMISS.ControlLoop()
    problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],controlLoop)
    controlLoop.TimesSet(transient[0],transient[1],transient[2])
    controlLoop.TimeOutputSet(transient[3])
    problem.ControlLoopCreateFinish()

    # Create problem solver
    dynamicSolver = CMISS.Solver()
    problem.SolversCreateStart()
    problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver)
    dynamicSolver.outputType = CMISS.SolverOutputTypes.NONE
    dynamicSolver.dynamicTheta = [0.5]
    nonlinearSolver = CMISS.Solver()
    dynamicSolver.DynamicNonlinearSolverGet(nonlinearSolver)
    nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
    nonlinearSolver.outputType = CMISS.SolverOutputTypes.NONE
    nonlinearSolver.newtonAbsoluteTolerance = 1.0E-8
    nonlinearSolver.newtonRelativeTolerance = 1.0E-8
    nonlinearSolver.newtonSolutionTolerance = 1.0E-8
    nonlinearSolver.newtonMaximumFunctionEvaluations = 10000
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
    # Wall boundary nodes u = 0 (no-slip)
    value=0.0
    boundaryTolerance = 1.0e-6
    nodes = CMISS.Nodes()
    region.NodesGet(nodes)
    for node in range(nodes.numberOfNodes):    
        nodeId = node + 1
        nodeNumber = nodes.UserNumberGet(nodeId)
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,1)
        if (nodeDomain == computationalNodeNumber):
            xLocation = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,1)
            yLocation = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,2)
            zLocation = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,3)
            # rigid wall (left,right,bottom,top) conditions: velocity=0
            if (xLocation < boundaryTolerance or 
                meshDimensions[0]-xLocation < boundaryTolerance or
                yLocation < boundaryTolerance or
                meshDimensions[1]-yLocation < boundaryTolerance):
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,0.0)
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,0.0)
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,3,CMISS.BoundaryConditionsTypes.FIXED,0.0)
                #print("Wall node: " + str(nodeNumber))
            # inlet conditions: sinusoidal pressure
            if (zLocation < boundaryTolerance):
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.FIXED_INLET,0.0)
                print("Inlet node: " + str(nodeNumber))
            # outlet conditions: 0 pressure
            elif (meshDimensions[2]-zLocation < boundaryTolerance):
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,
                                           1,CMISS.GlobalDerivativeConstants.NO_GLOBAL_DERIV,
                                           nodeNumber,4,CMISS.BoundaryConditionsTypes.FIXED_OUTLET,0.0)
                print("Outlet node: " + str(nodeNumber))

    solverEquations.BoundaryConditionsCreateFinish()

    for element in range(numberOfElements):    
        elementId = element + 1
        elementNumber = elementId
        elementDomain=decomposition.ElementDomainGet(elementNumber)
        if (elementDomain == computationalNodeNumber):
            # Inlet elements
            if (elementNumber <= meshResolution[0]*meshResolution[1]):
                print("Inlet element: " + str(elementNumber))
                boundaryID = 2.0
                normal = [0.0,0.0,-1.0]
                # Boundary ID
                equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                              elementNumber,8,boundaryID)
                # Boundary normal
                for component in range(len(meshDimensions)):
                    componentId = component + 5
                    value = normal[component]
                    equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                  elementNumber,componentId,value)
            # Outlet elements
            elif (elementNumber > numberOfElements - meshResolution[0]*meshResolution[1]):
                print("Outlet element: " + str(elementNumber))
                boundaryID = 3.0
                normal = [0.0,0.0,1.0]
                # Boundary ID
                equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                              elementNumber,8,boundaryID)
                # Boundary normal
                for component in range(len(meshDimensions)):
                    componentId = component + 5
                    value = normal[component]
                    equationsSetField.ParameterSetUpdateElementDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                                  elementNumber,componentId,value)
                

    # Solve the problem
    print("solving problem...")
    problem.Solve()
    print("Finished. Time to solve (seconds): " + str(time.time()-startTime))    

    # Clear fields so can run in batch mode on this region
    materialsField.Destroy()
    dependentField.Destroy()
    analyticField.Destroy()
    equationsSet.Destroy()
    problem.Destroy()


#==========================================================
# P r o b l e m     C o n t r o l
#==========================================================

# Problem parameters
offset = 0.0
density = 1.0
amplitude = 1.0
period = math.pi/2.
timeIncrements = [period/100.]
womersleyNumbers = [10.0]
startTime = 0.0
stopTime = 2.0*period + 0.000001
outputFrequency = 1

for timeIncrement in timeIncrements:
    
    transient = [startTime,stopTime,timeIncrement,outputFrequency]
    for w in womersleyNumbers:
        # determine w using viscosity:density ratio (fixed angular frequency and radius)
        viscosity = density/(w**2.0)
        # make a new output directory if necessary
        outputDirectory = "./output/Wom" + str(w) + 'Dt' + str(round(timeIncrement,5)) + meshName + "/"
        try:
            os.makedirs(outputDirectory)
        except OSError, e:
            if e.errno != 17:
                raise   
        # change to new directory and solve problem (note will return to original directory on exit)
        with ChangeDirectory(outputDirectory):
            solveProblem(transient,viscosity,density,offset,amplitude,period)

CMISS.Finalise()





