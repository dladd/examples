#!/usr/bin/env python

#> \file
#> \author David Ladd
#> \brief This is an example script to solve a Navier-Stokes lid driven cavity benchmark problem using openCMISS calls in python.
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

#> \example FluidMechanics/NavierStokes/LidDrivenCavity/LidDriven.py
## Example script to solve a Navier-Stokes lid driven cavity benchmark problem using openCMISS calls in python.
## \par Latest Builds:
#<


# Add Python bindings directory to PATH
import sys, os
sys.path.append(os.sep.join((os.environ['OPENCMISS_ROOT'],'cm','bindings','python')))

import numpy
import gzip
import pylab
import time
import matplotlib.pyplot as plt

# Intialise OpenCMISS
from opencmiss import CMISS

# Get the computational nodes information
numberOfComputationalNodes = CMISS.ComputationalNumberOfNodesGet()
computationalNodeNumber = CMISS.ComputationalNodeNumberGet()

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
 equationsSetUserNumber,
 problemUserNumber) = range(1,14)

# Creation a RC coordinate system
coordinateSystem = CMISS.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.dimension = 2
coordinateSystem.CreateFinish()

# Create a region
region = CMISS.Region()
region.CreateStart(regionUserNumber,CMISS.WorldRegion)
region.label = "Cavity"
region.coordinateSystem = coordinateSystem
region.CreateFinish()

# Create a biquadratic lagrange basis
quadraticBasis = CMISS.Basis()
quadraticBasis.CreateStart(quadraticBasisUserNumber)
quadraticBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
quadraticBasis.numberOfXi = 2
quadraticBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE]*2
quadraticBasis.quadratureNumberOfGaussXi = [2]*2
quadraticBasis.CreateFinish()

# Create a bilinear lagrange basis
linearBasis = CMISS.Basis()
linearBasis.CreateStart(linearBasisUserNumber)
linearBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
linearBasis.numberOfXi = 2
linearBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*2
linearBasis.quadratureNumberOfGaussXi = [2]*2
linearBasis.CreateFinish()


def LidDriven(numberOfElements,cavityDimensions,lidVelocity,viscosity,density,outputFilename,transient):
    """ Sets up the lid driven cavity problem and solves with the provided parameter values

          Square Lid-Driven Cavity

                  v=1
               >>>>>>>>>>
             1|          |
              |          |
         v=0  |          |  v=0
              |          |
              |          |
              ------------
             0    v=0    1
    """

    # Create a generated mesh
    generatedMesh = CMISS.GeneratedMesh()
    generatedMesh.CreateStart(generatedMeshUserNumber,region)
    generatedMesh.type = CMISS.GeneratedMeshTypes.REGULAR
    generatedMesh.basis = [quadraticBasis,linearBasis]
    generatedMesh.extent = cavityDimensions
    generatedMesh.numberOfElements = numberOfElements

    mesh = CMISS.Mesh()
    generatedMesh.CreateFinish(meshUserNumber,mesh)

    # Create a decomposition for the mesh
    decomposition = CMISS.Decomposition()
    decomposition.CreateStart(decompositionUserNumber,mesh)
    decomposition.type = CMISS.DecompositionTypes.CALCULATED
    decomposition.numberOfDomains = numberOfComputationalNodes
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

    # Create standard Navier-Stokes equations set
    equationsSetField = CMISS.Field()
    equationsSet = CMISS.EquationsSet()
    if transient:
        equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
                CMISS.EquationsSetClasses.FLUID_MECHANICS,
                CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                CMISS.EquationsSetSubtypes.TRANSIENT_SUPG_NAVIER_STOKES,
                equationsSetFieldUserNumber, equationsSetField)
    else:
        equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
                CMISS.EquationsSetClasses.FLUID_MECHANICS,
                CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                CMISS.EquationsSetSubtypes.STATIC_SUPG_NAVIER_STOKES,
                equationsSetFieldUserNumber, equationsSetField)
    equationsSet.CreateFinish()

    # Create dependent field
    dependentField = CMISS.Field()
    equationsSet.DependentCreateStart(dependentFieldUserNumber,dependentField)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,1,1)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,2,1)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.U,3,2)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,1,1)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,2,1)
    dependentField.ComponentMeshComponentSet(CMISS.FieldVariableTypes.DELUDELN,3,2)
    dependentField.DOFOrderTypeSet(CMISS.FieldVariableTypes.U,CMISS.FieldDOFOrderTypes.SEPARATED)
    dependentField.DOFOrderTypeSet(CMISS.FieldVariableTypes.DELUDELN,CMISS.FieldDOFOrderTypes.SEPARATED)
    equationsSet.DependentCreateFinish()
    # Initialise dependent field
    dependentField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,0.0)

    # Create materials field
    materialsField = CMISS.Field()
    equationsSet.MaterialsCreateStart(materialsFieldUserNumber,materialsField)
    equationsSet.MaterialsCreateFinish()
    # Initialise materials field parameters
    materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,viscosity)
    materialsField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,density)

    # Create equations
    equations = CMISS.Equations()
    equationsSet.EquationsCreateStart(equations)
    equations.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
    equations.outputType = CMISS.EquationsOutputTypes.NONE
    equationsSet.EquationsCreateFinish()

    # Create Navier-Stokes problem
    problem = CMISS.Problem()
    problem.CreateStart(problemUserNumber)
    if transient:
        problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                                 CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                                 CMISS.ProblemSubTypes.TRANSIENT_SUPG_NAVIER_STOKES)
    else:
        problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                                 CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                                 CMISS.ProblemSubTypes.STATIC_NAVIER_STOKES)
    problem.CreateFinish()

    # Create control loops
    problem.ControlLoopCreateStart()
    if transient:
        controlLoop = CMISS.ControlLoop()
        problem.ControlLoopGet([CMISS.ControlLoopIdentifiers.NODE],controlLoop)
        controlLoop.TimesSet(transient[0],transient[1],transient[2])
        controlLoop.TimeOutputSet(transient[3])
    problem.ControlLoopCreateFinish()

    # Create problem solver
    if transient:
        dynamicSolver = CMISS.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,dynamicSolver)
        dynamicSolver.outputType = CMISS.SolverOutputTypes.NONE
        dynamicSolver.dynamicTheta = [1.0]
        nonlinearSolver = CMISS.Solver()
        dynamicSolver.DynamicNonlinearSolverGet(nonlinearSolver)
        nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
        nonlinearSolver.outputType = CMISS.SolverOutputTypes.PROGRESS
        nonlinearSolver.newtonAbsoluteTolerance = 1.0E-7
        nonlinearSolver.newtonRelativeTolerance = 1.0E-7
        nonlinearSolver.newtonSolutionTolerance = 1.0E-7
        nonlinearSolver.newtonMaximumFunctionEvaluations = 10000
        linearSolver = CMISS.Solver()
        nonlinearSolver.NewtonLinearSolverGet(linearSolver)
        linearSolver.outputType = CMISS.SolverOutputTypes.NONE
        linearSolver.linearType = CMISS.LinearSolverTypes.DIRECT
        linearSolver.libraryType = CMISS.SolverLibraries.MUMPS
        problem.SolversCreateFinish()
    else:
        nonlinearSolver = CMISS.Solver()
        linearSolver = CMISS.Solver()
        problem.SolversCreateStart()
        problem.SolverGet([CMISS.ControlLoopIdentifiers.NODE],1,nonlinearSolver)
        nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
        nonlinearSolver.outputType = CMISS.SolverOutputTypes.NONE
        nonlinearSolver.newtonAbsoluteTolerance = 1.0E-8
        nonlinearSolver.newtonRelativeTolerance = 1.0E-8
        nonlinearSolver.newtonSolutionTolerance = 1.0E-8
        nonlinearSolver.newtonMaximumFunctionEvaluations = 10000
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
    # Set values for boundary 
    firstNodeNumber=1
    nodes = CMISS.Nodes()
    region.NodesGet(nodes)
    print("Total # of nodes: " + str(nodes.numberOfNodes))
    boundaryTolerance = 1.0e-6
    # Currently issues with getting generated mesh surfaces through python so easier to just loop over all nodes
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
            # rigid wall (left,right,bottom) conditions: v=0
            if (xLocation < boundaryTolerance or 
                cavityDimensions[0]-xLocation < boundaryTolerance or
                yLocation < boundaryTolerance):
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,0.0)
                boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,0.0)
            # lid (top) conditions: v=v
            elif (cavityDimensions[1]-yLocation < boundaryTolerance):
                if not (xLocation < boundaryTolerance or 
                        cavityDimensions[0]-xLocation < boundaryTolerance):
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,lidVelocity[0])
                    boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,lidVelocity[1])
    solverEquations.BoundaryConditionsCreateFinish()

    # Solve the problem
    problem.Solve()

    print("exporting CMGUI data")
    # Export results
    fields = CMISS.Fields()
    fields.CreateRegion(region)
    fields.NodesExport(outputFilename,"FORTRAN")
    fields.ElementsExport(outputFilename,"FORTRAN")
    fields.Finalise()

    # Clear fields so can run in batch mode on this region
    generatedMesh.Destroy()
    nodes.Destroy()
    mesh.Destroy()
    geometricField.Destroy()
    equationsSetField.Destroy()
    dependentField.Destroy()
    materialsField.Destroy()
    equationsSet.Destroy()
    problem.Destroy()


# Problem defaults
dimensions = [1.0,1.0]
elementResolution = [10,10]#[60,60]
ReynoldsNumbers = [100]#[100,400,1000,2500,3200,5000]
lidVelocity = [1.0,0.0]
viscosity = 1.0
density = 1.0

for Re in ReynoldsNumbers:
    viscosity = 1.0/Re

    # High Re problems susceptible to mode switching using direct iteration- solve steady state using transient
    # solver instead to dampen out instabilities.
    if Re > 500:
        # transient parameters: startTime,stopTime,timeIncrement,outputFrequency (for static, leave list empty)
        transient = [0.0,300.0001,0.5,100]
    else:
        transient = []

    outputDirectory = "./output/Re" + str(Re) + "Elem" +str(elementResolution[0])+"x" +str(elementResolution[1]) + "/"
    try:
        os.makedirs(outputDirectory)
    except OSError, e:
        if e.errno != 17:
            raise   

    outputFile = outputDirectory +"LidDrivenCavity"
    LidDriven(elementResolution,dimensions,lidVelocity,viscosity,density,outputFile,transient)
    print('Finished solving Re ' + str(Re))




CMISS.Finalise()





