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
import time
import exfile

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
 analyticFieldUserNumber,
 problemUserNumber) = range(1,15)

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
quadraticBasis.quadratureNumberOfGaussXi = [3]*2
quadraticBasis.CreateFinish()

# Create a bilinear lagrange basis
linearBasis = CMISS.Basis()
linearBasis.CreateStart(linearBasisUserNumber)
linearBasis.type = CMISS.BasisTypes.LAGRANGE_HERMITE_TP
linearBasis.numberOfXi = 2
linearBasis.interpolationXi = [CMISS.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*2
linearBasis.quadratureNumberOfGaussXi = [3]*2
linearBasis.CreateFinish()


def LidDriven(numberOfElements,cavityDimensions,lidVelocity,viscosity,density,
              outputFilename,transient,supg,fdJacobian,referencePressure,initialiseFromFile,analytic):
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
    if supg:
        generatedMesh.basis = [quadraticBasis,linearBasis]
#        generatedMesh.basis = [quadraticBasis,quadraticBasis]
#        generatedMesh.basis = [linearBasis,linearBasis]
    else:
        generatedMesh.basis = [quadraticBasis,linearBasis]
#        generatedMesh.basis = [quadraticBasis,quadraticBasis]
#        generatedMesh.basis = [linearBasis,linearBasis]
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
    if supg:
        equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
                CMISS.EquationsSetClasses.FLUID_MECHANICS,
                CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                CMISS.EquationsSetSubtypes.TRANSIENT_SUPG_NAVIER_STOKES,
                equationsSetFieldUserNumber, equationsSetField)
    else:
        equationsSet.CreateStart(equationsSetUserNumber,region,geometricField,
                CMISS.EquationsSetClasses.FLUID_MECHANICS,
                CMISS.EquationsSetTypes.NAVIER_STOKES_EQUATION,
                CMISS.EquationsSetSubtypes.TRANSIENT_NAVIER_STOKES,
                equationsSetFieldUserNumber, equationsSetField)
    equationsSet.CreateFinish()

    if supg:
        # Set beta: boundary retrograde flow stabilisation scaling factor (default 1.0)
        equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                      CMISS.FieldParameterSetTypes.VALUES,1,0.0)
        # Set max CFL number (default 1.0)
        equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                      CMISS.FieldParameterSetTypes.VALUES,2,1.0E20)
        # Set time increment (default 0.0)
        equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                      CMISS.FieldParameterSetTypes.VALUES,3,transient[2])
        # Set C1 
        equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                      CMISS.FieldParameterSetTypes.VALUES,4,12.0)
        # Set stabilisation type (default 1.0)
        equationsSetField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.V,
                                                      CMISS.FieldParameterSetTypes.VALUES,5,1.0)

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

    # If specified, create analytic field (allows for time-dependent calculation of sinusoidal waveform during solve)
    if analytic:
        #(offset + amplitude*cos(2*pi*(CURRENT_TIME/(period))))
        amplitude = -0.5*lidVelocity[0]
        offset = 0.5*lidVelocity[0]
        period = 20.0
        startSine = 0.0
        stopSine = period/2.0
        analyticField = CMISS.Field()
        equationsSet.AnalyticCreateStart(CMISS.NavierStokesAnalyticFunctionTypes.FlowrateSinusoid,analyticFieldUserNumber,analyticField)
        equationsSet.AnalyticCreateFinish()
        # Initialise analytic field parameters: (1-4) Dependent params, 5 amplitude, 6 offset, 7 period
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,1,1.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,2,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,3,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,4,0.0)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,5,amplitude)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,6,offset)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,7,period)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,8,startSine)
        analyticField.ComponentValuesInitialiseDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,9,stopSine)

    # Create equations
    equations = CMISS.Equations()
    equationsSet.EquationsCreateStart(equations)
    equations.sparsityType = CMISS.EquationsSparsityTypes.SPARSE
    equations.outputType = CMISS.EquationsOutputTypes.NONE
    equationsSet.EquationsCreateFinish()

    # Create Navier-Stokes problem
    problem = CMISS.Problem()
    problem.CreateStart(problemUserNumber)
    if supg:
        problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                                 CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                                 CMISS.ProblemSubTypes.TRANSIENT_SUPG_NAVIER_STOKES)
    else:
        problem.SpecificationSet(CMISS.ProblemClasses.FLUID_MECHANICS,
                                 CMISS.ProblemTypes.NAVIER_STOKES_EQUATION,
                                 CMISS.ProblemSubTypes.TRANSIENT_NAVIER_STOKES)
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
    if fdJacobian:
        nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.FD
    else:
        nonlinearSolver.newtonJacobianCalculationType = CMISS.JacobianCalculationTypes.EQUATIONS
    nonlinearSolver.outputType = CMISS.SolverOutputTypes.NONE
    nonlinearSolver.newtonAbsoluteTolerance = 1.0E-8
    nonlinearSolver.newtonRelativeTolerance = 1.0E-9
    nonlinearSolver.newtonSolutionTolerance = 1.0E-9
    nonlinearSolver.newtonMaximumFunctionEvaluations = 10000
    nonlinearSolver.newtonLineSearchType = CMISS.NewtonLineSearchTypes.QUADRATIC
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
    # Set values for boundary 
    firstNodeNumber=1
    nodes = CMISS.Nodes()
    region.NodesGet(nodes)
    print("Total # of nodes: " + str(nodes.numberOfNodes))
    if initialiseFromFile:
        dependentField.ParameterSetCreate(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES)
        initFile = "./output/Re" + str(int(1/viscosity)) + "Elem" +str(numberOfElements[0])+"x" +str(numberOfElements[1]) + "_SUPG/init/Initialise.exnode"
        print("Reading initial data from: " + initFile)
        initialData = exfile.Exnode(initFile)
    boundaryTolerance = 1.0e-6

    # Currently issues with getting generated mesh surfaces through python so easier to just loop over all nodes
    for node in range(nodes.numberOfNodes):
        nodeId = node + 1
        nodeNumber = nodes.UserNumberGet(nodeId)
        # print('node number: '+ str(nodeNumber))
        # Velocity nodes
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
                    if analytic:
                        boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED_INLET,0.0)
                        boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED_INLET,0.0)
                    else:
                        boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,1,CMISS.BoundaryConditionsTypes.FIXED,lidVelocity[0])
                        boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,2,CMISS.BoundaryConditionsTypes.FIXED,lidVelocity[1])

                    #print('lid node: '+str(nodeNumber))                    

            # Init values from file
            if initialiseFromFile:
                for component in range(1, 3):                
                    component_name = str(component)
                    value = initialData.node_value("U", component_name, nodeNumber, 1)
                    dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.VALUES,
                                                    1,1,nodeNumber,component,value)
                    dependentField.ParameterSetUpdateNodeDP(CMISS.FieldVariableTypes.U,CMISS.FieldParameterSetTypes.PREVIOUS_VALUES,
                                                    1,1,nodeNumber,component,value)

    # Pressure node
    if (referencePressure):
        nodeNumber = 1 #numberOfElements[0] + 1
        nodeDomain=decomposition.NodeDomainGet(nodeNumber,2)
        if (nodeDomain == computationalNodeNumber):
            xLocation = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,1)
            yLocation = geometricField.ParameterSetGetNodeDP(CMISS.FieldVariableTypes.U,
                                                             CMISS.FieldParameterSetTypes.VALUES,
                                                             1,1,nodeNumber,2)
            # bottom center node - reference pressure: p=0
            boundaryConditions.SetNode(dependentField,CMISS.FieldVariableTypes.U,1,1,nodeNumber,3,CMISS.BoundaryConditionsTypes.FIXED,0.0)
            print('pressure node: '+str(nodeNumber))

    solverEquations.BoundaryConditionsCreateFinish()
    # Need to fix using MUMPS 5.0?
    #linearSolver.MumpsSetIcntl(14,60)

    # Solve the problem
    print("solving...")
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
    if supg:
        equationsSetField.Destroy()
    if analytic:
        analyticField.Destroy()
    dependentField.Destroy()
    materialsField.Destroy()
    equationsSet.Destroy()
    problem.Destroy()


# Problem defaults
dimensions = [1.0,1.0]
elementResolutions = [20]#,20,30,40,60]
#elementResolution = [20,20]
ReynoldsNumbers = [100,400,1000,2500,3200,5000]
lidVelocity = [1.0,0.0]
density = 1.0
supgTypes = [True]
fdJacobian = False
referencePressure = True
initialiseFromFile = False
analyticLidVelocity = True

# # blood: 1.06x10^-3 kg/cm^3
# density = 0.00106
# # blood: 3.5x10^-5 kg/cm^3
# viscosity = 0.000035

for elemRes in elementResolutions:
    for Re in ReynoldsNumbers:
        for supg in supgTypes:

            elementResolution = [elemRes,elemRes]
            viscosity = density*lidVelocity[0]/Re

            #lidVelocity[0] = viscosity*Re/density

            # High Re problems susceptible to mode switching using direct iteration- solve steady state using transient
            # solver instead to dampen out instabilities.
            # transient parameters: startTime,stopTime,timeIncrement,outputFrequency
            transient = [0.0,300.000001,0.1,1000000]#000000]
            if supg:    
                outputDirectory = "./output/Re" + str(Re) + "Elem" +str(elementResolution[0])+"x" +str(elementResolution[1]) + "_SUPG/"
            else:
                outputDirectory = "./output/Re" + str(Re) + "Elem" +str(elementResolution[0])+"x" +str(elementResolution[1]) + "_GFEM/"

            try:
                os.makedirs(outputDirectory)
            except OSError, e:
                if e.errno != 17:
                    raise   
            outputFile = outputDirectory +"LidDrivenCavity"
            # Display solve info
            if computationalNodeNumber == 0:
                print('Starting solve with parameters: ')
                print('---------------------------------------------------------------------')
                print('    Reynolds number (Re): ' + str(Re))
                print('    ElementResolution   : ' + str(elementResolution))
                print('    SUPG                : ' + str(supg))
                print('    Transient parameters: ' + str(transient))
                print('    FD Jacobian: ' + str(fdJacobian))
            start = time.time()
            LidDriven(elementResolution,dimensions,lidVelocity,viscosity,density,
                      outputFile,transient,supg,fdJacobian,referencePressure,initialiseFromFile,analyticLidVelocity)
            end = time.time()
            runtime = end - start
            if computationalNodeNumber == 0:
                print('Finished solve with parameters: ')
                print('---------------------------------------------------------------------')
                print('    Reynolds number (Re): ' + str(Re))
                print('    ElementResolution   : ' + str(elementResolution))
                print('    SUPG                : ' + str(supg))
                print('    Transient parameters: ' + str(transient))
                print('    FD Jacobian         : ' + str(fdJacobian))
                print('    Runtime             : ' + str(runtime))

CMISS.Finalise()





