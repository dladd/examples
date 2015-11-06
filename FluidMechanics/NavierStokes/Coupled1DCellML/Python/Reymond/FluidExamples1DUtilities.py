#> \file
#> \author David Ladd
#> \brief This is a utility script, with routines for reading data associated with 1D Navier-Stokes problems, analysing the mesh, and outputting results.
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
#> Contributor(s): Soroush Safaei
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
#> OpenCMISS/examples/FluidMechanics/NavierStokes/Coupled1DCellML/Python/Reymond/1DFluidUtilities.py
#>

import csv
import math
import numpy as np
from numpy import linalg  

def GetNumberOfNodes(filename):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine gets the number of nodes for array allocation.
    '''
    # Read the node file
    try:
        with open(filename,'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rownum = 0
            for row in reader:
                if (rownum == 0):
                    # Read the header row
                    header = row
                else:
                    # Read the number of nodes
                    if (rownum == 1):
                        numberOfNodes = int(row[5])
                    else:
                        break
                rownum+=1
            return(numberOfNodes);
    except IOError:
        print ('Could not open Node csv file: ' + filename)

def CsvNodeReader(filename,inputNodeNumbers,bifurcationNodeNumbers,trifurcationNodeNumbers,coupledNodeNumbers,nodeCoordinates,arteryLabels):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in nodal data and returns it to the user.
    '''
    numberOfInputNodes     = 0
    numberOfBifurcations   = 0
    numberOfTrifurcations  = 0
    numberOfTerminalNodes  = 0
    # Read the node file
    try:
        with open(filename,'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rownum = -1
            for row in reader:
                if (rownum == -1):
                    # Read the header row
                    header = row
                else:
                    # Read the number of nodes
                    if (rownum == 0):
                        numberOfNodesSpace = int(row[5])
                        totalNumberOfNodes = numberOfNodesSpace*3
                        xValues = np.empty([numberOfNodesSpace,4])
                        yValues = np.empty([numberOfNodesSpace,4])
                        zValues = np.empty([numberOfNodesSpace,4])
                        xValues.fill(np.nan)
                        yValues.fill(np.nan)
                        zValues.fill(np.nan)
                    # Initialise the coordinates
                    arteryLabels.append(row[0])
                    xValues[rownum,0] = float(row[1])
                    yValues[rownum,0] = float(row[2])
                    zValues[rownum,0] = float(row[3])
                    # Read the input nodes
                    if (row[4] == 'input'):
                        inputNodeNumbers.append(rownum+1)
                        numberOfInputNodes = numberOfInputNodes+1
                    # Read the bifurcation nodes
                    elif (row[4] == 'bifurcation'):
                        numberOfBifurcations+=1
                        bifurcationNodeNumbers.append(rownum+1)
                        xValues[rownum,1] = float(row[1])
                        yValues[rownum,1] = float(row[2])
                        zValues[rownum,1] = float(row[3])
                        xValues[rownum,2] = float(row[1])
                        yValues[rownum,2] = float(row[2])
                        zValues[rownum,2] = float(row[3])
                    # Read the trifurcation nodes
                    elif (row[4] == 'trifurcation'):
                        numberOfTrifurcations+=1
                        trifurcationNodeNumbers.append(rownum+1)
                        xValues[rownum,1] = float(row[1])
                        yValues[rownum,1] = float(row[2])
                        zValues[rownum,1] = float(row[3])
                        xValues[rownum,2] = float(row[1])
                        yValues[rownum,2] = float(row[2])
                        zValues[rownum,2] = float(row[3])
                        xValues[rownum,3] = float(row[1])
                        yValues[rownum,3] = float(row[2])
                        zValues[rownum,3] = float(row[3])
                    # Read the terminal nodes
                    elif (row[4] == 'terminal'):
                        coupledNodeNumbers.append(rownum+1)
                        numberOfTerminalNodes = numberOfTerminalNodes+1
                # Next line
                rownum+=1      
            # Create coordinates numpy array - init to NaN
            nodeCoordinates[:,:,0] = xValues[:,:]
            nodeCoordinates[:,:,1] = yValues[:,:]
            nodeCoordinates[:,:,2] = zValues[:,:]
    except IOError:
        print ('Could not open Node csv file: ' + filename)


def CsvNodeReader2(filename,inputNodeNumbers,branchNodeNumbers,terminalNodeNumbers,nodeCoordinates,
                   branchNodeElements,terminalArteryNames,RCRParameters):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in nodal data and returns it to the user.
    '''
    # Read the node file
    try:
        with open(filename,'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            rownum = -1
            for row in reader:
                if (rownum == -1):
                    # Read the header row
                    header = row
                else:
                    nodeCoordinates.append([float(row[1]),float(row[2]),float(row[3])])
                    if row[4] == 'inlet':
                        inputNodeNumbers.append(int(row[0]))
                        #inputArteryNames.append(row[5])
                    elif row[4] == 'terminal':                    
                        terminalNodeNumbers.append(int(row[0]))
                        terminalArteryNames.append(row[5])             
                        RCRParameters.append([float(row[6]),float(row[7]),float(row[8])])
                    elif row[4] == '':
                        continue
                    else:
                        branchNodeNumbers.append(int(row[0]))
                        thisNodeElements = []
                        for element in range(4,len(row)):
                            if row[element] == '':
                                break
                            else:
                                thisNodeElements.append(int(row[element]))
                        branchNodeElements.append(thisNodeElements)

                # Next line
                rownum+=1      
    except IOError:
        print ('Could not open Node csv file: ' + filename)


def CsvElementReader(filename,elementNodes,bifurcationElements,trifurcationElements,numberOfBifurcations,numberOfTrifurcations):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in element data and returns it to the user.
    '''
    try:
        # Read the element file
        with open(filename,'rb') as csvfile:
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
                        #elementNodes          = (totalNumberOfElements+1)*[3*[0]]
                    # Read the element nodes
                    #elementNodes[rownum] = [int(row[1]),int(row[2]),int(row[3])]
                    elementNodes.append([int(row[1]),int(row[2]),int(row[3])])
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
    except IOError:
        print ('Could not open Element csv file: ' + filename)

def CsvElementReader2(filename,elementNodes,elementArteryNames):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in element data and returns it to the user.
    '''
    try:
        # Read the element file
        with open(filename,'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            rownum = 0
            for row in reader:
                if (rownum == 0):
                    # Read the header row
                    header = row
                else:
                    elementNodes.append([int(row[1]),int(row[2]),int(row[3])])
                    elementArteryNames.append(row[4])
                # Next line
                rownum+=1
    except IOError:
        print ('Could not open Element csv file: ' + filename)

def CsvMaterialReader(filename,A0,E,H):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in material data and returns it to the user.
    '''
    try:
        # Read the element file
        with open(filename,'rb') as csvfile:
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
    except IOError:
        print ('Could not open Material csv file: ' + filename)

def CsvMaterialReader2(filename,A0,E,H):
    ''' 1D Navier-Stokes problems are currently described using csv files- this routine reads in material data and returns it to the user.
    '''
    try:
        # Read the element file
        with open(filename,'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            rownum = 0
            for row in reader:
                if (rownum == 0):
                    # Read the header row
                    header = row
                else:
                    nodeA0=[]
                    nodeE=[]
                    nodeH=[]
                    for version in range(len(row)/3):
                        if row[version*3+1] == '':
                            break
                        else:
                            nodeA0.append(float(row[1+version*3]))
                            nodeE.append(float(row[2+version*3]))
                            nodeH.append(float(row[3+version*3]))
                    A0.append(nodeA0)
                    E.append(nodeE)
                    H.append(nodeH)
                rownum+=1
    except IOError:
        print ('Could not open Material csv file: ' + filename)

def WriteCellMLRCRModels(terminalArteryNames,RCRParameters,terminalPressure,modelDirectory):
    ''' Writes the terminal parameter values to RCR Windkessel models
    '''    

    prefix = '''<?xml version="1.0"?>
<model xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cmeta="http://www.cellml.org/metadata/1.0#" cmeta:id="ModelRCR" name="ModelRCR">

<units base_units="no" name="UnitP"><unit units="gram"/><unit exponent="-1" prefix="milli" units="metre"/><unit exponent="-2" prefix="milli" units="second"/></units>
<units base_units="no" name="UnitQ"><unit exponent="-1" prefix="milli" units="second"/><unit exponent="3" prefix="milli" units="metre"/></units>
<units base_units="no" name="UnitR"><unit units="UnitP"/><unit exponent="-3" prefix="milli" units="metre"/><unit prefix="milli" units="second"/></units>
<units base_units="no" name="UnitC"><unit exponent="-1" units="UnitP"/><unit exponent="3" prefix="milli" units="metre"/></units>
<units base_units="no" name="UnitT"><unit prefix="milli" units="second"/></units>

'''
    suffix = '''
<component name="environment">
<variable initial_value="0.0" name="t" public_interface="out" units="UnitT"/></component>

<component name="Circuit">
<variable initial_value="0.0" name="Pout" private_interface="in" public_interface="out" units="UnitP"/>
<variable initial_value="0.0" name="Qin" private_interface="out" public_interface="in" units="UnitQ"/></component>

<component name="RC">
<variable initial_value="0.0" name="Pi" public_interface="out" units="UnitP"/>
<variable name="Po" public_interface="in" units="UnitP"/>
<variable name="Qi" public_interface="in" units="UnitQ"/>
<variable name="Qo" public_interface="out" units="UnitQ"/>
<variable name="R" public_interface="in" units="UnitR"/>
<variable name="C" public_interface="in" units="UnitC"/>
<variable name="t" public_interface="in" units="second"/>
    
<math xmlns="http://www.w3.org/1998/Math/MathML" id="RC equations"><apply><eq/><apply><divide/><apply><minus/><ci>Qi</ci><ci>Qo</ci></apply><ci>C</ci></apply><apply><diff/><bvar><ci>t</ci></bvar><ci>Pi</ci></apply></apply><apply><eq/><apply><divide/><apply><minus/><ci>Pi</ci><ci>Po</ci></apply><ci>R</ci></apply><ci>Qo</ci></apply>
</math></component>

<component name="Rp">
<variable name="Pi" public_interface="out" units="UnitP"/>
<variable name="Po" public_interface="in" units="UnitP"/>
<variable name="Qi" public_interface="in" units="UnitQ"/>
<variable name="Qo" public_interface="out" units="UnitQ"/>
<variable name="R" public_interface="in" units="UnitR"/>

<math xmlns="http://www.w3.org/1998/Math/MathML" id="R equations"><apply><eq/><ci>Pi</ci><apply><plus/><ci>Po</ci><apply><times/><ci>R</ci><ci>Qi</ci></apply></apply></apply><apply><eq/><ci>Qo</ci><ci>Qi</ci></apply>
</math></component>

<connection><map_components component_1="environment" component_2="RC"/>
<map_variables variable_1="t" variable_2="t"/></connection>

<connection><map_components component_1="Circuit" component_2="Rp"/>
<map_variables variable_1="Pout" variable_2="Pi"/>
<map_variables variable_1="Qin" variable_2="Qi"/></connection>

<connection><map_components component_1="Rp" component_2="RC"/>
<map_variables variable_1="Po" variable_2="Pi"/>
<map_variables variable_1="Qo" variable_2="Qi"/></connection>

<connection><map_components component_1="ParameterValues" component_2="Rp"/>
<map_variables variable_1="ResistanceProximal" variable_2="R"/></connection>

<connection><map_components component_1="ParameterValues" component_2="RC"/>
<map_variables variable_1="ResistanceDistal" variable_2="R"/>
<map_variables variable_1="Capacitance" variable_2="C"/>
<map_variables variable_1="PressureTerminal" variable_2="Po"/></connection>

<group>
<relationship_ref relationship="encapsulation"/>
<component_ref component="Circuit">
<component_ref component="environment"/>
<component_ref component="Rp"/>
<component_ref component="RC"/>
<component_ref component="ParameterValues"/></component_ref>
</group></model>
'''

    for arteryIdx in range(len(terminalArteryNames)):
        arteryName = terminalArteryNames[arteryIdx]
        filename = modelDirectory  + arteryName + '.cellml'
        print('Creating CellML model: '+ filename)
        print('    R1, R2, C, pterm = '+ str(RCRParameters[arteryIdx]) + '  ' + str(terminalPressure))
        f = open(filename,'w')
        f.write(prefix) 
        parameterLine = '\n<component name="ParameterValues">\n'
        f.write(parameterLine)
        parameterLine = '<variable initial_value="'+str(RCRParameters[arteryIdx][0])+'" name="ResistanceProximal" public_interface="out" units="UnitR"></variable>\n'
        f.write(parameterLine)
        parameterLine = '<variable initial_value="'+str(RCRParameters[arteryIdx][1])+'" name="ResistanceDistal" public_interface="out" units="UnitR"></variable>\n'
        f.write(parameterLine)
        parameterLine = '<variable initial_value="'+str(RCRParameters[arteryIdx][2])+'" name="Capacitance" public_interface="out" units="UnitC"></variable>\n'
        f.write(parameterLine)
        parameterLine = '<variable initial_value="'+str(terminalPressure)+'" name="PressureTerminal" public_interface="out" units="UnitP"/>\n</component>\n'
        f.write(parameterLine)
        f.write(suffix) 
        f.close()
        


def GetMaxStableTimestep(elementNodes,QMax,nodeCoordinates,H,E,A0,Rho):
    ''' Indicates the min/max timestep for the provided mesh
    '''
    maxTimestep = 0.0
    numberOfElements = len(elementNodes)-1
    numberOfNodes = len(nodeCoordinates)
    # Check the element length
    elementNumber = [0]*(numberOfElements+1)
    elementLength = [0]*(numberOfElements+1)
    #numberOfNodes = len(nodeCoordinates[:,0,0])
    eig  = [0]*(numberOfNodes+1)
    for i in range(1,numberOfElements+1):
        Node1 = elementNodes[i][0]
        Node2 = elementNodes[i][1]
        Node3 = elementNodes[i][2]
        #Length1 = linalg.norm(nodeCoordinates[Node1-1,0,:]-nodeCoordinates[Node2-1,0,:])
        #Length2 = linalg.norm(nodeCoordinates[Node2-1,0,:]-nodeCoordinates[Node3-1,0,:])
        Length1 = linalg.norm(np.array(nodeCoordinates[Node1-1])-np.array(nodeCoordinates[Node2-1]))
        Length2 = linalg.norm(np.array(nodeCoordinates[Node2-1])-np.array(nodeCoordinates[Node3-1]))
        elementNumber[i] = i
        elementLength[i] = Length1 + Length2
        elementLength[0] = elementLength[i]
        # print "Element %1.0f" %elementNumber[i], 
        # print "Length: %1.1f" %elementLength[i],
        # print "Length1: %1.1f" %Length1,
        # print "Length2: %1.1f" %Length2
    maxElementLength = max(elementLength)
    minElementLength = min(elementLength)
    print("Max Element Length: %1.3f" % maxElementLength)
    print("Min Element Length: %1.3f" % minElementLength)
               
    # Check the timestep
    dt   = [0]*(numberOfNodes)
    for i in range(numberOfNodes):
        # beta   = (3.0*math.sqrt(math.pi)*H[i,0]*E[i,0])/(4.0*A0[i,0])
        # eig[i] = QMax/A0[i,0] + (A0[i,0]**0.25)*(math.sqrt(beta/(2.0*Rho)))
        beta   = (3.0*math.sqrt(math.pi)*H[i][0]*E[i][0])/(4.0*A0[i][0])
        eig[i] = QMax/A0[i][0] + (A0[i][0]**0.25)*(math.sqrt(beta/(2.0*Rho)))
        dt[i]  = ((3.0**(0.5))/3.0)*minElementLength/eig[i]
        #dt[0]  = dt[i]
    maxTimestep = min(dt)
    print("Max allowable timestep: %3.5f" % maxTimestep )
    return(maxTimestep);

def ExnodeInitReader(filename,Q,A,dQ,dA):
    # Reads Q,A,dQ,dA data from exnode file (for initilisation)
    print('reading initialisation data from: '+ filename)
    inputFile = open(filename)
    lines = inputFile.readlines()
    for i in range(0,len(lines)):
        if 'Node:' in lines[i]:
            nodeNumber = [int(s) for s in lines[i].split() if s.isdigit()][0]
            # Extract the variables from output files
            for version in range(len(lines[i+1].split())):
                Q[nodeNumber-1,version] = float(lines[i+4].split()[version])
                A[nodeNumber-1,version] = float(lines[i+5].split()[version])
                dQ[nodeNumber-1,version] = float(lines[i+6].split()[version])
                dA[nodeNumber-1,version] = float(lines[i+7].split()[version])
    inputFile.close()
